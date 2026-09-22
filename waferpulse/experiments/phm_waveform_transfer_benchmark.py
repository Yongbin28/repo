"""Leakage-safe PHM-to-EquipmentData temporal waveform transfer benchmark.

The PHM16 response labels are deliberately never loaded.  Its upstream CMP
sensor sequences provide only a generic, response-free 1-D waveform basis.
For every EquipmentData outer fold, imputation and the EquipmentData portion of
the PCA basis are fit on training manufacturing lots only.  Held-out lots are
transformed once, then scored using the existing StratifiedGroupKFold contract.

This experiment tests whether the alternate open-source dataset can improve
the original EquipmentData goal without mixing incompatible targets or using
target/class leakage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
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

from waferpulse.experiments.phm2016_cmp_benchmark import (
    CRITICAL_SIGNALS,
    KEY_COLUMNS,
    _discover_dataset_root,
    _load_traces,
)
from waferpulse.tools.equipment_data import load_equipment_trace_dataset


SEED = 42


def _interpolate_vector(values: np.ndarray, points: int) -> np.ndarray | None:
    vector = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vector)
    if not finite.any():
        return None
    source = np.linspace(0.0, 1.0, len(vector), dtype=np.float64)
    if finite.sum() == 1:
        filled = np.full(len(vector), vector[finite][0], dtype=np.float64)
    else:
        filled = np.interp(source, source[finite], vector[finite])
    target = np.linspace(0.0, 1.0, points, dtype=np.float64)
    return np.interp(target, source, filled)


def normalize_waveforms(waveforms: np.ndarray) -> np.ndarray:
    values = np.asarray(waveforms, dtype=np.float64)
    center = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return (values - center) / scale


def extract_external_waveforms(
    traces: pd.DataFrame,
    *,
    points: int,
    signals: Sequence[str] = CRITICAL_SIGNALS,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Convert PHM upstream groups to normalized response-free waveforms."""

    retained = tuple(signal for signal in signals if signal in traces.columns)
    if not retained:
        raise ValueError("No requested PHM upstream signals are present")
    rows: List[np.ndarray] = []
    ordered = traces.sort_values(KEY_COLUMNS + ["TIMESTAMP", "_source_index"])
    for _, group in ordered.groupby(KEY_COLUMNS, sort=False):
        for signal in retained:
            waveform = _interpolate_vector(group[signal].to_numpy(), points)
            if waveform is not None:
                rows.append(waveform)
    if not rows:
        raise ValueError("No finite PHM upstream waveforms were extracted")
    return normalize_waveforms(np.vstack(rows)), retained


def load_external_waveform_bank(
    root: Path, *, points: int, maximum: int = 30_000
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    # The label path is intentionally ignored: this experiment never opens it.
    _label_path, trace_dir = _discover_dataset_root(root)
    traces, paths = _load_traces(trace_dir)
    waveforms, signals = extract_external_waveforms(traces, points=points)
    extracted = len(waveforms)
    if len(waveforms) > maximum:
        indices = np.linspace(0, len(waveforms) - 1, maximum).astype(np.int64)
        waveforms = waveforms[indices]
    return waveforms, {
        "source": "PHM Society 2016 CMP upstream traces",
        "trace_files": len(paths),
        "trace_rows": int(len(traces)),
        "signals": list(signals),
        "waveforms_extracted": int(extracted),
        "waveforms_used": int(len(waveforms)),
        "response_labels_loaded": False,
    }


def load_supervised_external_projection(
    root: Path,
    *,
    points: int,
    components: int = 12,
    maximum: int = 30_000,
    max_valid_mrr: float = 300.0,
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    """Fit a frozen waveform projection using PHM labels only."""

    label_path, trace_dir = _discover_dataset_root(root)
    traces, paths = _load_traces(trace_dir)
    labels = pd.read_csv(label_path)
    labels["STAGE"] = labels["STAGE"].astype(str).str.strip()
    labels["WAFER_ID"] = pd.to_numeric(labels["WAFER_ID"], errors="raise")
    labels["AVG_REMOVAL_RATE"] = pd.to_numeric(
        labels["AVG_REMOVAL_RATE"], errors="raise"
    )
    labels = labels.loc[labels["AVG_REMOVAL_RATE"].le(max_valid_mrr)]
    target_by_key = labels.set_index(KEY_COLUMNS)["AVG_REMOVAL_RATE"]
    signals = tuple(signal for signal in CRITICAL_SIGNALS if signal in traces.columns)
    rows: List[np.ndarray] = []
    targets: List[float] = []
    ordered = traces.sort_values(KEY_COLUMNS + ["TIMESTAMP", "_source_index"])
    for key, group in ordered.groupby(KEY_COLUMNS, sort=False):
        if key not in target_by_key.index:
            continue
        target = float(target_by_key.loc[key])
        for signal in signals:
            waveform = _interpolate_vector(group[signal].to_numpy(), points)
            if waveform is not None:
                rows.append(waveform)
                targets.append(target)
    waveforms = normalize_waveforms(np.vstack(rows))
    y = np.asarray(targets, dtype=np.float64)
    extracted = len(waveforms)
    if len(waveforms) > maximum:
        indices = np.linspace(0, len(waveforms) - 1, maximum).astype(np.int64)
        waveforms = waveforms[indices]
        y = y[indices]
    pls = PLSRegression(
        n_components=min(components, waveforms.shape[1]),
        scale=False,
        max_iter=1000,
    ).fit(waveforms, y)
    projection = np.asarray(pls.x_rotations_, dtype=np.float64)
    return projection, {
        "source": "PHM Society 2016 CMP upstream traces and clean MRR labels",
        "trace_files": len(paths),
        "trace_rows": int(len(traces)),
        "signals": list(signals),
        "waveforms_extracted": int(extracted),
        "waveforms_used": int(len(waveforms)),
        "response_labels_loaded": True,
        "response_labels_source": "external PHM only",
        "max_valid_mrr": float(max_valid_mrr),
        "projection": "PLS x_rotations_ frozen before EquipmentData validation",
        "components": int(projection.shape[1]),
    }


def fit_trace_imputer(train_traces: np.ndarray) -> np.ndarray:
    median = np.nanmedian(np.asarray(train_traces, dtype=np.float64), axis=0)
    fallback = float(np.nanmedian(train_traces))
    if not np.isfinite(fallback):
        fallback = 0.0
    return np.where(np.isfinite(median), median, fallback)


def apply_trace_imputer(traces: np.ndarray, median: np.ndarray) -> np.ndarray:
    values = np.asarray(traces, dtype=np.float64)
    return np.where(np.isfinite(values), values, median[None, :, :])


def _deterministic_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).astype(np.int64)
    return values[indices]


def encode_equipment_traces(
    traces: np.ndarray, pca: PCA
) -> np.ndarray:
    """Encode every equipment sensor using amplitude and shared shape features."""

    values = np.asarray(traces, dtype=np.float64)
    wafers, sensors, points = values.shape
    flat = values.reshape(wafers * sensors, points)
    normalized = normalize_waveforms(flat)
    scores = pca.transform(normalized)
    reconstruction = pca.inverse_transform(scores)
    reconstruction_rmse = np.sqrt(np.mean((normalized - reconstruction) ** 2, axis=1))
    amplitude = np.column_stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            flat.min(axis=1),
            flat.max(axis=1),
            flat[:, -1] - flat[:, 0],
            np.sqrt(np.mean(flat**2, axis=1)),
        ]
    )
    per_sensor = np.column_stack([amplitude, scores, reconstruction_rmse])
    return per_sensor.reshape(wafers, sensors * per_sensor.shape[1])


def encode_equipment_traces_with_projection(
    traces: np.ndarray, projection: np.ndarray
) -> np.ndarray:
    values = np.asarray(traces, dtype=np.float64)
    wafers, sensors, points = values.shape
    flat = values.reshape(wafers * sensors, points)
    normalized = normalize_waveforms(flat)
    scores = normalized @ projection
    reconstruction = scores @ np.linalg.pinv(projection)
    reconstruction_rmse = np.sqrt(np.mean((normalized - reconstruction) ** 2, axis=1))
    amplitude = np.column_stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            flat.min(axis=1),
            flat.max(axis=1),
            flat[:, -1] - flat[:, 0],
            np.sqrt(np.mean(flat**2, axis=1)),
        ]
    )
    per_sensor = np.column_stack([amplitude, scores, reconstruction_rmse])
    return per_sensor.reshape(wafers, sensors * per_sensor.shape[1])


def build_fold_features(
    train_traces: np.ndarray,
    validation_traces: np.ndarray,
    external_waveforms: np.ndarray,
    *,
    components: int = 12,
    maximum_equipment_waveforms: int = 30_000,
    fixed_projection: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    median = fit_trace_imputer(train_traces)
    train_clean = apply_trace_imputer(train_traces, median)
    validation_clean = apply_trace_imputer(validation_traces, median)
    if fixed_projection is not None:
        return (
            encode_equipment_traces_with_projection(train_clean, fixed_projection),
            encode_equipment_traces_with_projection(
                validation_clean, fixed_projection
            ),
            {
                "components": int(fixed_projection.shape[1]),
                "external_basis_waveforms": None,
                "training_equipment_basis_waveforms": 0,
                "explained_variance_ratio": None,
                "basis_frozen_before_equipment_fold": True,
            },
        )
    equipment_bank = normalize_waveforms(train_clean.reshape(-1, train_clean.shape[-1]))
    equipment_bank = _deterministic_sample(
        equipment_bank, maximum_equipment_waveforms
    )
    basis_bank = (
        np.vstack([external_waveforms, equipment_bank])
        if len(external_waveforms)
        else equipment_bank
    )
    pca = PCA(
        n_components=min(components, basis_bank.shape[1]),
        svd_solver="randomized",
        random_state=SEED,
    ).fit(basis_bank)
    return (
        encode_equipment_traces(train_clean, pca),
        encode_equipment_traces(validation_clean, pca),
        {
            "components": int(pca.n_components_),
            "external_basis_waveforms": int(len(external_waveforms)),
            "training_equipment_basis_waveforms": int(len(equipment_bank)),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        },
    )


def _regressors() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=100.0)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=600,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        n_jobs=1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=5.0,
                        random_state=SEED,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        pass
    return models


def _classifiers() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=5.0,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        pass
    return models


def _response_metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def run_benchmark(
    equipment_dir: Path,
    phm_dir: Path,
    output_dir: Path,
    *,
    basis: str = "phm",
) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    started = time.perf_counter()
    dataset = load_equipment_trace_dataset(
        equipment_dir, stage_mode="both", max_invalid_rate=1.0
    )
    response = dataset.response.to_numpy(dtype=np.float64)
    bad = dataset.bad_label.to_numpy(dtype=np.int8)
    groups = dataset.groups.to_numpy(dtype=str)
    if basis == "phm":
        external, external_metadata = load_external_waveform_bank(
            phm_dir, points=dataset.traces.shape[-1]
        )
        output_stem = "phm_waveform_transfer"
        fixed_projection = None
    elif basis == "equipment_only":
        external = np.empty((0, dataset.traces.shape[-1]), dtype=np.float64)
        external_metadata = {
            "source": None,
            "waveforms_used": 0,
            "response_labels_loaded": False,
        }
        output_stem = "equipment_only_waveform_pca_control"
        fixed_projection = None
    elif basis == "phm_supervised_pls":
        fixed_projection, external_metadata = load_supervised_external_projection(
            phm_dir, points=dataset.traces.shape[-1]
        )
        external = np.empty((0, dataset.traces.shape[-1]), dtype=np.float64)
        output_stem = "phm_supervised_pls_transfer"
    else:
        raise ValueError(f"Unsupported basis: {basis!r}")
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            dataset.traces, bad, groups
        )
    )

    fold_features = []
    fold_metadata = []
    for fold, (train, validation) in enumerate(folds, start=1):
        train_features, validation_features, metadata = build_fold_features(
            dataset.traces[train],
            dataset.traces[validation],
            external,
            fixed_projection=fixed_projection,
        )
        fold_features.append((train, validation, train_features, validation_features))
        fold_metadata.append({"fold": fold, **metadata})

    rows: List[Dict[str, Any]] = []
    predictions: Dict[str, np.ndarray] = {}
    for name, template in _regressors().items():
        prediction = np.full(len(response), np.nan)
        model_started = time.perf_counter()
        for train, validation, train_features, validation_features in fold_features:
            fitted = clone(template).fit(train_features, response[train])
            prediction[validation] = np.asarray(
                fitted.predict(validation_features)
            ).reshape(-1)
        key = f"regression__{name}"
        predictions[key] = prediction
        rows.append(
            {
                "task": "direct_regression",
                "model": name,
                **_response_metrics(response, prediction),
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "seconds": time.perf_counter() - model_started,
            }
        )

    for name, template in _classifiers().items():
        probability = np.full(len(response), np.nan)
        hurdle = np.full(len(response), np.nan)
        model_started = time.perf_counter()
        for train, validation, train_features, validation_features in fold_features:
            fitted = clone(template).fit(train_features, bad[train])
            fold_probability = fitted.predict_proba(validation_features)[:, 1]
            probability[validation] = fold_probability
            good_mean = float(response[train][bad[train] == 0].mean())
            bad_mean = float(response[train][bad[train] == 1].mean())
            hurdle[validation] = good_mean + fold_probability * (bad_mean - good_mean)
        key = f"probability_hurdle__{name}"
        predictions[key] = hurdle
        predictions[f"bad_probability__{name}"] = probability
        rows.append(
            {
                "task": "probability_hurdle",
                "model": name,
                **_response_metrics(response, hurdle),
                "roc_auc": float(roc_auc_score(bad, probability)),
                "pr_auc": float(average_precision_score(bad, probability)),
                "seconds": time.perf_counter() - model_started,
            }
        )

    result = pd.DataFrame(rows).sort_values("r2", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / f"{output_stem}_benchmark.csv", index=False)
    ledger = dataset.identity[["lot", "wafer"]].copy()
    ledger["response"] = response
    ledger["bad"] = bad
    for key, value in predictions.items():
        ledger[f"prediction__{key}"] = value
    ledger.to_csv(output_dir / f"{output_stem}_oof.csv", index=False)
    metadata = {
        "purpose": (
            "response-free PHM upstream waveform transfer to EquipmentData"
            if basis == "phm"
            else (
                "external-label-supervised PHM waveform transfer to EquipmentData"
                if basis == "phm_supervised_pls"
                else "EquipmentData-only shared waveform PCA control"
            )
        ),
        "basis": basis,
        "validation": "5-fold StratifiedGroupKFold by EquipmentData manufacturing lot",
        "equipment_wafers": int(len(response)),
        "equipment_lots": int(pd.Series(groups).nunique()),
        "equipment_trace_shape": [int(value) for value in dataset.traces.shape],
        "external": external_metadata,
        "target_or_class_used_in_transfer_basis": basis == "phm_supervised_pls",
        "equipment_target_or_class_used_in_transfer_basis": False,
        "equipment_validation_lots_used_in_fold_basis": False,
        "fold_basis": fold_metadata,
        "best": result.iloc[0].to_dict(),
        "r2_gate": 0.8,
        "r2_gate_passed": bool(float(result.iloc[0]["r2"]) > 0.8),
        "seconds": time.perf_counter() - started,
    }
    (output_dir / f"{output_stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equipment-dir", type=Path, default=Path("EquipmentData"))
    parser.add_argument("--phm-dir", type=Path, default=Path("data/phm2016_cmp"))
    parser.add_argument(
        "--basis",
        choices=("phm", "equipment_only", "phm_supervised_pls"),
        default="phm",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/equipment_quality")
    )
    args = parser.parse_args()
    result, metadata = run_benchmark(
        args.equipment_dir, args.phm_dir, args.output_dir, basis=args.basis
    )
    print(result.to_string(index=False))
    print(json.dumps({"best": metadata["best"], "gate": metadata["r2_gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
