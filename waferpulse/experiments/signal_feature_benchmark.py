"""Leakage-safe signal-processing benchmark for all EquipmentData traces.

Feature construction uses only each wafer's upstream sensor sequence. Model
selection evidence is generated with manufacturing-lot-grouped out-of-fold
predictions; lot/wafer identifiers and response/class columns are never used as
predictors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from waferpulse.tools.equipment_data import load_equipment_trace_dataset


SEED = 42


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=np.abs(denominator) > 1e-12,
    )


def build_signal_features(
    traces: np.ndarray,
    sensor_names: Tuple[str, ...],
) -> pd.DataFrame:
    values = np.asarray(traces, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected wafer-by-sensor-by-time traces, found {values.shape}")
    if not np.isfinite(values).all():
        median = np.nanmedian(values, axis=(0, 2), keepdims=True)
        values = np.where(np.isfinite(values), values, median)

    blocks: List[np.ndarray] = []
    names: List[str] = []

    mean = values.mean(axis=2)
    centered = values - mean[:, :, None]
    std = values.std(axis=2)
    scale = np.where(std > 1e-12, std, 1.0)
    standardized = centered / scale[:, :, None]

    time_statistics: Mapping[str, np.ndarray] = {
        "mean": mean,
        "std": std,
        "min": values.min(axis=2),
        "max": values.max(axis=2),
        "median": np.median(values, axis=2),
        "q01": np.quantile(values, 0.01, axis=2),
        "q05": np.quantile(values, 0.05, axis=2),
        "q25": np.quantile(values, 0.25, axis=2),
        "q75": np.quantile(values, 0.75, axis=2),
        "q95": np.quantile(values, 0.95, axis=2),
        "q99": np.quantile(values, 0.99, axis=2),
        "rms": np.sqrt(np.mean(values * values, axis=2)),
        "skew": np.mean(standardized**3, axis=2),
        "kurtosis": np.mean(standardized**4, axis=2) - 3.0,
        "first": values[:, :, 0],
        "last": values[:, :, -1],
        "delta": values[:, :, -1] - values[:, :, 0],
        "argmin": np.argmin(values, axis=2) / float(values.shape[2] - 1),
        "argmax": np.argmax(values, axis=2) / float(values.shape[2] - 1),
        "mean_crossing_rate": np.mean(
            standardized[:, :, :-1] * standardized[:, :, 1:] < 0, axis=2
        ),
    }
    difference = np.diff(values, axis=2)
    time_statistics = {
        **time_statistics,
        "diff_mean": difference.mean(axis=2),
        "diff_std": difference.std(axis=2),
        "diff_abs_mean": np.abs(difference).mean(axis=2),
        "diff_abs_max": np.abs(difference).max(axis=2),
    }
    for statistic, block in time_statistics.items():
        blocks.append(block)
        names.extend(f"{sensor}__{statistic}" for sensor in sensor_names)

    for lag in (1, 2, 4, 8, 16, 32):
        numerator = np.sum(centered[:, :, :-lag] * centered[:, :, lag:], axis=2)
        denominator = np.sqrt(
            np.sum(centered[:, :, :-lag] ** 2, axis=2)
            * np.sum(centered[:, :, lag:] ** 2, axis=2)
        )
        blocks.append(_safe_divide(numerator, denominator))
        names.extend(f"{sensor}__autocorr_lag_{lag}" for sensor in sensor_names)

    frequency = np.abs(np.fft.rfft(centered, axis=2)) ** 2
    frequency[:, :, 0] = 0.0
    total_power = frequency.sum(axis=2)
    frequency_indices = np.arange(frequency.shape[2], dtype=np.float64)
    spectral_centroid = _safe_divide(
        np.sum(frequency * frequency_indices[None, None, :], axis=2), total_power
    )
    dominant_frequency = np.argmax(frequency, axis=2).astype(np.float64)
    spectral_probability = _safe_divide(frequency, total_power[:, :, None])
    spectral_entropy = -np.sum(
        np.where(
            spectral_probability > 0,
            spectral_probability * np.log(spectral_probability + 1e-12),
            0.0,
        ),
        axis=2,
    )
    for statistic, block in (
        ("spectral_total_power", np.log1p(total_power)),
        ("spectral_centroid", spectral_centroid),
        ("spectral_dominant_frequency", dominant_frequency),
        ("spectral_entropy", spectral_entropy),
    ):
        blocks.append(block)
        names.extend(f"{sensor}__{statistic}" for sensor in sensor_names)
    for band_number, (start, stop) in enumerate(
        ((1, 3), (3, 6), (6, 11), (11, 21), (21, 41), (41, frequency.shape[2])),
        start=1,
    ):
        band = frequency[:, :, start:stop].sum(axis=2)
        blocks.append(_safe_divide(band, total_power))
        names.extend(f"{sensor}__spectral_band_{band_number}" for sensor in sensor_names)

    for window_number, indices in enumerate(np.array_split(np.arange(values.shape[2]), 16)):
        for statistic, block in (
            ("mean", values[:, :, indices].mean(axis=2)),
            ("std", values[:, :, indices].std(axis=2)),
        ):
            blocks.append(block)
            names.extend(
                f"{sensor}__window_{window_number + 1:02d}_{statistic}"
                for sensor in sensor_names
            )

    # Pairwise sensor correlation captures synchronous equipment behaviour.
    # It is computed separately within each wafer and therefore cannot import
    # information from another wafer, lot, or response label.
    sensor_count = values.shape[1]
    upper = np.triu_indices(sensor_count, k=1)
    correlation = np.einsum(
        "nst,nmt->nsm", standardized, standardized, optimize=True
    ) / float(values.shape[2])
    blocks.append(correlation[:, upper[0], upper[1]])
    names.extend(
        f"corr__{sensor_names[left]}__{sensor_names[right]}"
        for left, right in zip(*upper)
    )

    matrix = np.concatenate(blocks, axis=1).astype(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    if matrix.shape[1] != len(names):
        raise AssertionError("Signal feature names do not match feature matrix")
    return pd.DataFrame(matrix, columns=names)


def _pipeline(estimator: Any, *, k: int, classification: bool, scale: bool) -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
        ("selector", SelectKBest(f_classif if classification else f_regression, k=k)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _candidate_models(feature_count: int) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    k = min(500, feature_count)
    regressors: Dict[str, Any] = {
        "ridge": _pipeline(Ridge(alpha=100.0), k=k, classification=False, scale=True),
        "extra_trees": _pipeline(
            ExtraTreesRegressor(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
            classification=False,
            scale=False,
        ),
    }
    classifiers: Dict[str, Any] = {
        "logistic": _pipeline(
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            ),
            k=k,
            classification=True,
            scale=True,
        ),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
            classification=True,
            scale=False,
        ),
        "extra_trees": _pipeline(
            ExtraTreesClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
            classification=True,
            scale=False,
        ),
    }
    return regressors, classifiers


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> Mapping[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def run_benchmark(data_dir: Path, output_dir: Path) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    started = time.perf_counter()
    dataset = load_equipment_trace_dataset(
        data_dir, stage_mode="both", max_invalid_rate=1.0
    )
    X = build_signal_features(dataset.traces, dataset.sensor_names)
    response = dataset.response.to_numpy(float)
    bad = dataset.bad_label.to_numpy(int)
    groups = dataset.groups.to_numpy(str)
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            X, bad, groups
        )
    )
    regressors, classifiers = _candidate_models(X.shape[1])
    rows: List[Dict[str, Any]] = []
    predictions: Dict[str, np.ndarray] = {}

    for name, template in regressors.items():
        prediction = np.full(len(response), np.nan)
        model_started = time.perf_counter()
        for train, validation in folds:
            fitted = clone(template).fit(X.iloc[train], response[train])
            prediction[validation] = np.asarray(
                fitted.predict(X.iloc[validation])
            ).reshape(-1)
        row = {
            "task": "direct_regression",
            "model": name,
            **_metrics(response, prediction),
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "seconds": time.perf_counter() - model_started,
        }
        rows.append(row)
        predictions[f"regression__{name}"] = prediction
        print(json.dumps(row), flush=True)

    for name, template in classifiers.items():
        probability = np.full(len(response), np.nan)
        hurdle = np.full(len(response), np.nan)
        model_started = time.perf_counter()
        for train, validation in folds:
            fitted = clone(template).fit(X.iloc[train], bad[train])
            fold_probability = fitted.predict_proba(X.iloc[validation])[:, 1]
            probability[validation] = fold_probability
            good_mean = float(response[train][bad[train] == 0].mean())
            bad_mean = float(response[train][bad[train] == 1].mean())
            hurdle[validation] = good_mean + fold_probability * (bad_mean - good_mean)
        row = {
            "task": "probability_hurdle",
            "model": name,
            **_metrics(response, hurdle),
            "roc_auc": float(roc_auc_score(bad, probability)),
            "pr_auc": float(average_precision_score(bad, probability)),
            "seconds": time.perf_counter() - model_started,
        }
        rows.append(row)
        predictions[f"hurdle__{name}"] = hurdle
        predictions[f"probability__{name}"] = probability
        print(json.dumps(row), flush=True)

    result = pd.DataFrame(rows).sort_values("r2", ascending=False)
    ledger = dataset.identity.copy()
    ledger["response"] = response
    ledger["bad"] = bad
    for name, values in predictions.items():
        ledger[f"prediction__{name}"] = values
    metadata = {
        "validation": "5-fold StratifiedGroupKFold by manufacturing lot",
        "wafers": int(len(response)),
        "lots": int(pd.Series(groups).nunique()),
        "bad_wafers": int(bad.sum()),
        "raw_trace_shape": [int(value) for value in dataset.traces.shape],
        "signal_features": int(X.shape[1]),
        "feature_families": [
            "time statistics",
            "derivatives",
            "autocorrelation",
            "frequency-domain power",
            "sixteen temporal windows",
            "within-wafer cross-sensor correlation",
        ],
        "feature_exclusions": ["lot", "wafer", "response", "class"],
        "best_candidate": result.iloc[0].to_dict(),
        "r2_gate": 0.8,
        "r2_gate_passed": bool(float(result.iloc[0]["r2"]) > 0.8),
        "selection_status": "exploratory candidate comparison; no production replacement",
        "seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "signal_feature_benchmark.csv", index=False)
    ledger.to_csv(output_dir / "signal_feature_oof_predictions.csv", index=False)
    (output_dir / "signal_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result, metadata


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result, metadata = run_benchmark(
        root / "EquipmentData", root / "output" / "equipment_quality"
    )
    print("\nSIGNAL FEATURE RESULTS\n" + result.to_string(index=False), flush=True)
    print("\n" + json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
