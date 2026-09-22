"""Search raw sensor/time-point signal under strict unseen-lot validation.

The existing benchmark mainly used 49 policy-approved sensors. This experiment
uses all 56 deterministically recovered channels and fold-local supervised
selection to test whether localized trace points encode continuous severity.
True-class rows are diagnostics only and are never deployable candidates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from waferpulse.tools.equipment_data import (
    load_equipment_dataset,
    load_equipment_trace_dataset,
)


SEED = 42


def _align(data_dir: Path) -> Tuple[Any, np.ndarray, List[str]]:
    engineered = load_equipment_dataset(data_dir, stage_mode="both")
    traces = load_equipment_trace_dataset(
        data_dir,
        stage_mode="both",
        max_invalid_rate=1.0,
    )
    trace_index = pd.MultiIndex.from_frame(traces.identity[["lot", "wafer"]])
    target_index = pd.MultiIndex.from_frame(engineered.identity[["lot", "wafer"]])
    positions = trace_index.get_indexer(target_index)
    if (positions < 0).any():
        raise ValueError("Trace and target identities do not align")
    names = [
        f"{sensor}__t{timestamp:03d}"
        for sensor in traces.sensor_names
        for timestamp in range(traces.traces.shape[2])
    ]
    raw = traces.traces[positions].reshape(len(engineered.response), -1).astype(np.float64)
    return engineered, raw, names


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def _pipeline(model: Any, k: int, *, scale: bool = True) -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        # Center first to avoid catastrophic cancellation in f_regression for
        # high-magnitude equipment channels. This transformer is fold-local.
        ("selection_scale", StandardScaler()),
        ("select", SelectKBest(f_regression, k=k)),
    ]
    steps.append(("model", model))
    return Pipeline(steps)


def candidates() -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for k in (25, 50, 100, 200, 500):
        for alpha in (10.0, 100.0, 1000.0, 10000.0):
            rows[f"raw56_ridge_k{k}_a{alpha:g}"] = _pipeline(Ridge(alpha=alpha), k)
    for k, components in ((50, 2), (100, 2), (100, 5), (200, 5), (500, 10)):
        rows[f"raw56_pls_k{k}_c{components}"] = _pipeline(
            PLSRegression(n_components=components, scale=False, max_iter=1000), k
        )
    for k, c, epsilon in (
        (50, 0.1, 0.05),
        (100, 0.1, 0.05),
        (200, 0.1, 0.05),
        (100, 0.3, 0.05),
        (200, 0.3, 0.05),
        (500, 0.3, 0.10),
        (200, 1.0, 0.10),
    ):
        rows[f"raw56_svr_k{k}_c{c}_e{epsilon}"] = _pipeline(
            SVR(C=c, epsilon=epsilon, gamma="scale"), k
        )
    for k in (50, 100, 200, 500):
        rows[f"raw56_extratrees_k{k}"] = _pipeline(
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=1,
                random_state=SEED,
            ),
            k,
            scale=False,
        )
    for k in (50, 100, 200):
        rows[f"raw56_histgradient_k{k}"] = _pipeline(
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.04,
                max_leaf_nodes=7,
                l2_regularization=2.0,
                random_state=SEED,
            ),
            k,
            scale=False,
        )
    for k, alpha, ratio in ((100, 0.001, 0.1), (200, 0.001, 0.5), (500, 0.01, 0.1)):
        rows[f"raw56_elastic_k{k}_a{alpha}_l{ratio}"] = _pipeline(
            ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=5000, random_state=SEED), k
        )
    return rows


def _correlation_audit(
    raw: np.ndarray,
    response: np.ndarray,
    bad: np.ndarray,
    names: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, mask in (("all", np.ones(len(response), dtype=bool)), ("good", bad == 0), ("bad", bad == 1)):
        values = raw[mask].astype(np.float64)
        target = response[mask].astype(np.float64)
        values -= values.mean(axis=0)
        target -= target.mean()
        denominator = np.sqrt(np.square(values).sum(axis=0) * np.square(target).sum())
        correlation = np.divide(
            values.T @ target,
            denominator,
            out=np.zeros(values.shape[1], dtype=float),
            where=denominator > 0,
        )
        order = np.argsort(np.abs(correlation))[::-1][:100]
        rows.extend(
            {
                "subset": label,
                "rank": rank,
                "feature": names[index],
                "pearson_r": float(correlation[index]),
                "abs_pearson_r": float(abs(correlation[index])),
            }
            for rank, index in enumerate(order, start=1)
        )
    return pd.DataFrame(rows)


def run_benchmark(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    dataset, raw, names = _align(data_dir)
    response = dataset.response.to_numpy(float)
    bad = dataset.bad_label.to_numpy(int)
    groups = dataset.groups.to_numpy(str)
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            raw, bad, groups
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    correlation = _correlation_audit(raw, response, bad, names)
    correlation.to_csv(output_dir / "raw56_localized_correlation_audit.csv", index=False)

    rows: List[Dict[str, Any]] = []
    for number, (name, template) in enumerate(candidates().items(), start=1):
        started = time.perf_counter()
        prediction = np.full(len(response), np.nan)
        fold_r2: List[float] = []
        status = "OK"
        error = ""
        try:
            for train, validation in folds:
                fitted = clone(template).fit(raw[train], response[train])
                fold_prediction = np.asarray(fitted.predict(raw[validation])).reshape(-1)
                prediction[validation] = fold_prediction
                fold_r2.append(float(r2_score(response[validation], fold_prediction)))
            metrics = _metrics(response, prediction)
        except Exception as exc:
            metrics = {"r2": np.nan, "rmse": np.nan, "mae": np.nan}
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "experiment": name,
            **metrics,
            "fold_r2_min": min(fold_r2) if fold_r2 else np.nan,
            "fold_r2_max": max(fold_r2) if fold_r2 else np.nan,
            "seconds": time.perf_counter() - started,
            "status": status,
            "error": error,
        }
        rows.append(row)
        print(f"[{number}/{len(candidates())}] {json.dumps(row)}", flush=True)

    # Diagnostic ceiling: true class is deliberately supplied at validation.
    oracle_prediction = np.full(len(response), np.nan)
    for train, validation in folds:
        good_mean = float(response[train][bad[train] == 0].mean())
        bad_mean = float(response[train][bad[train] == 1].mean())
        oracle_prediction[validation] = np.where(bad[validation] == 1, bad_mean, good_mean)
    rows.append(
        {
            "experiment": "TRUE_CLASS_FOLD_MEAN_ORACLE_NOT_DEPLOYABLE",
            **_metrics(response, oracle_prediction),
            "fold_r2_min": np.nan,
            "fold_r2_max": np.nan,
            "seconds": 0.0,
            "status": "DIAGNOSTIC_ONLY",
            "error": "Uses true validation class",
        }
    )
    results = pd.DataFrame(rows).sort_values("r2", ascending=False, na_position="last")
    results.to_csv(output_dir / "raw56_localized_benchmark.csv", index=False)
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run_benchmark(root / "EquipmentData", root / "output" / "equipment_quality")
    print("\nRAW LOCALIZED RESULTS\n" + result.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
