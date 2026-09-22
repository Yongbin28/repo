"""Compare raw traces, engineered features, models and parameters fairly.

Every candidate uses the same StratifiedGroupKFold allocation by manufacturing
lot. The resulting CSV is an experiment ledger, not a replacement for the
production metrics until a candidate is integrated and independently rerun.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from waferpulse.tools.equipment_data import (
    _inspect_equipment,
    _read_semicolon_csv,
    load_equipment_dataset,
)
from waferpulse.tools.equipment_modeling import QualityHurdleRegressor


SEED = 42
ENGINEERED_NINE_ALGORITHMS = (
    "Ridge",
    "SVR-RBF",
    "QualityHurdleSVM",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
)


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    model: str
    parameters: str
    estimator: Any


def _raw_stage_matrix(data_dir: Path, equipment: str) -> pd.DataFrame:
    raw = _read_semicolon_csv(data_dir / f"{equipment}.csv")
    numeric, _, _ = _inspect_equipment(raw, equipment, max_invalid_rate=0.01)
    sensors = [column for column in numeric if column.startswith("sensor_")]
    wide = numeric.set_index(["lot", "wafer", "_timestamp_number"])[sensors].unstack(
        "_timestamp_number"
    )
    wide = wide.sort_index(axis=1, level=[0, 1])
    wide.columns = [
        f"{equipment}__{sensor}__t{int(timestamp):03d}" for sensor, timestamp in wide.columns
    ]
    return wide


def build_feature_sets(data_dir: Path) -> Tuple[Dict[str, pd.DataFrame], Any]:
    dataset = load_equipment_dataset(data_dir, stage_mode="both")
    identity = pd.MultiIndex.from_frame(dataset.identity[["lot", "wafer"]])
    raw = _raw_stage_matrix(data_dir, "equipment1").join(
        _raw_stage_matrix(data_dir, "equipment2"), how="inner", validate="one_to_one"
    )
    raw = raw.reindex(identity)
    if raw.isna().any().any():
        raise ValueError("The aligned reliable raw-trace matrix contains missing readings")
    raw = raw.reset_index(drop=True)

    sensor_names: List[str] = []
    for column in raw.columns[::176]:
        sensor_names.append(column.rsplit("__t", 1)[0])
    cube = raw.to_numpy(dtype=float).reshape(len(raw), len(sensor_names), 176)
    windows = np.array_split(np.arange(176), 16)
    blocks: List[np.ndarray] = []
    names: List[str] = []
    for statistic in ("mean", "std"):
        for window_number, indices in enumerate(windows, start=1):
            values = (
                cube[:, :, indices].mean(axis=2)
                if statistic == "mean"
                else cube[:, :, indices].std(axis=2)
            )
            blocks.append(values)
            names.extend(
                f"{sensor}__window_{window_number:02d}_{statistic}" for sensor in sensor_names
            )
    difference = np.diff(cube, axis=2)
    for statistic, values in (
        ("diff_mean", difference.mean(axis=2)),
        ("diff_std", difference.std(axis=2)),
        ("diff_abs_mean", np.abs(difference).mean(axis=2)),
        ("diff_abs_max", np.abs(difference).max(axis=2)),
    ):
        blocks.append(values)
        names.extend(f"{sensor}__{statistic}" for sensor in sensor_names)
    trace_shape = pd.DataFrame(np.concatenate(blocks, axis=1), columns=names)
    rich = pd.concat([dataset.features.reset_index(drop=True), trace_shape], axis=1)
    return {
        "raw_trace_no_hand_features": raw,
        "engineered_key_numbers": dataset.features.reset_index(drop=True),
        "engineered_rich_trace_shape": rich,
    }, dataset


def _preprocessed(model: Any, k: int | str = "all", scale: bool = False) -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
    ]
    if k != "all":
        steps.append(("selector", SelectKBest(f_regression, k=k)))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _forest(kind: str, **overrides: Any) -> Any:
    model_class = RandomForestRegressor if kind == "RF" else ExtraTreesRegressor
    parameters = {
        "n_estimators": 180,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": 1,
        "random_state": SEED,
    }
    parameters.update(overrides)
    return model_class(**parameters)


def candidates() -> List[Candidate]:
    rows: List[Candidate] = [
        Candidate("engineered_key_numbers", "DummyMean", "strategy=mean", _preprocessed(DummyRegressor())),
    ]

    for alpha in (1.0, 10.0, 100.0, 1000.0):
        rows.append(
            Candidate(
                "raw_trace_no_hand_features",
                "Ridge",
                f"alpha={alpha}",
                _preprocessed(Ridge(alpha=alpha), scale=True),
            )
        )
    for c in (0.03, 0.1, 0.3, 1.0):
        rows.append(
            Candidate(
                "raw_trace_no_hand_features",
                "SVR-RBF",
                f"C={c},epsilon=0.1,gamma=scale",
                _preprocessed(SVR(C=c, epsilon=0.1, gamma="scale"), scale=True),
            )
        )
    for c, gamma in ((0.1, 0.00005), (0.3, 0.0001), (0.3, 0.0002), (0.5, 0.0002)):
        rows.append(
            Candidate(
                "raw_trace_no_hand_features",
                "QualityHurdleSVM",
                f"C={c},gamma={gamma}",
                QualityHurdleRegressor(c=c, gamma=gamma, random_state=SEED),
            )
        )
    rows.extend(
        [
            Candidate(
                "raw_trace_no_hand_features",
                "RandomForest",
                "k=500,depth=12,leaf=2,max_features=sqrt",
                _preprocessed(_forest("RF"), k=500),
            ),
            Candidate(
                "raw_trace_no_hand_features",
                "ExtraTrees",
                "k=500,depth=12,leaf=2,max_features=sqrt",
                _preprocessed(_forest("ET"), k=500),
            ),
            Candidate(
                "raw_trace_no_hand_features",
                "PCA+Ridge",
                "pca=64,alpha=100",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=64, svd_solver="randomized", random_state=SEED)),
                        ("model", Ridge(alpha=100.0)),
                    ]
                ),
            ),
        ]
    )

    engineered_forest_parameters = (
        ("RF", "depth=12,leaf=2,max_features=sqrt", {}),
        ("RF", "depth=8,leaf=3,max_features=sqrt", {"max_depth": 8, "min_samples_leaf": 3}),
        ("RF", "depth=None,leaf=1,max_features=sqrt", {"max_depth": None, "min_samples_leaf": 1}),
        ("RF", "depth=12,leaf=2,max_features=0.3", {"max_features": 0.3}),
        ("ET", "depth=12,leaf=2,max_features=sqrt", {}),
        ("ET", "depth=None,leaf=1,max_features=sqrt", {"max_depth": None, "min_samples_leaf": 1}),
        ("ET", "depth=12,leaf=2,max_features=0.3", {"max_features": 0.3}),
    )
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        rows.append(
            Candidate(
                "engineered_key_numbers",
                "Ridge",
                f"k=120,alpha={alpha}",
                _preprocessed(Ridge(alpha=alpha), k=120, scale=True),
            )
        )
    for kind, description, parameters in engineered_forest_parameters:
        rows.append(
            Candidate(
                "engineered_key_numbers",
                "RandomForest" if kind == "RF" else "ExtraTrees",
                f"k=120,{description}",
                _preprocessed(_forest(kind, **parameters), k=120),
            )
        )
    for c in (0.03, 0.1, 0.3, 1.0):
        rows.append(
            Candidate(
                "engineered_key_numbers",
                "SVR-RBF",
                f"C={c},epsilon=0.1,gamma=scale",
                _preprocessed(SVR(C=c, epsilon=0.1, gamma="scale"), k=120, scale=True),
            )
        )
    for c, gamma in ((0.1, 0.001), (0.3, 0.001), (0.3, 0.002), (0.5, 0.001), (0.5, 0.002)):
        rows.append(
            Candidate(
                "engineered_key_numbers",
                "QualityHurdleSVM",
                f"C={c},gamma={gamma}",
                QualityHurdleRegressor(c=c, gamma=gamma, random_state=SEED),
            )
        )
    rows.extend(
        [
            Candidate(
                "engineered_key_numbers",
                "GradientBoosting",
                "k=120,n=200,depth=2,lr=0.03,loss=huber",
                _preprocessed(
                    GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=2,
                        learning_rate=0.03,
                        loss="huber",
                        random_state=SEED,
                    ),
                    k=120,
                ),
            ),
        ]
    )

    # Optional boosting libraries are imported only for this experiment.
    try:
        from xgboost import XGBRegressor

        for depth, leaves in ((2, 8), (3, 8)):
            rows.append(
                Candidate(
                    "engineered_key_numbers",
                    "XGBoost",
                    f"k=200,depth={depth},min_child_weight={leaves},n=400,lr=0.025",
                    _preprocessed(
                        XGBRegressor(
                            n_estimators=400,
                            learning_rate=0.025,
                            max_depth=depth,
                            min_child_weight=leaves,
                            subsample=0.85,
                            colsample_bytree=0.75,
                            reg_lambda=2.0,
                            objective="reg:squarederror",
                            n_jobs=1,
                            random_state=SEED,
                        ),
                        k=200,
                    ),
                )
            )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        for leaves in (7, 15):
            rows.append(
                Candidate(
                    "engineered_key_numbers",
                    "LightGBM",
                    f"k=200,num_leaves={leaves},n=400,lr=0.025",
                    _preprocessed(
                        LGBMRegressor(
                            n_estimators=400,
                            learning_rate=0.025,
                            num_leaves=leaves,
                            max_depth=5,
                            min_child_samples=20,
                            subsample=0.85,
                            colsample_bytree=0.75,
                            reg_lambda=2.0,
                            verbosity=-1,
                            n_jobs=1,
                            random_state=SEED,
                        ),
                        k=200,
                    ),
                )
            )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor

        rows.append(
            Candidate(
                "engineered_key_numbers",
                "CatBoost",
                "k=200,depth=4,n=400,lr=0.025",
                _preprocessed(
                    CatBoostRegressor(
                        iterations=400,
                        depth=4,
                        learning_rate=0.025,
                        loss_function="RMSE",
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=1,
                        random_seed=SEED,
                    ),
                    k=200,
                ),
            )
        )
    except ImportError:
        pass

    rows.extend(
        [
            Candidate(
                "engineered_rich_trace_shape",
                "RandomForest",
                "k=120,depth=12,leaf=2,max_features=sqrt",
                _preprocessed(_forest("RF"), k=120),
            ),
            Candidate(
                "engineered_rich_trace_shape",
                "ExtraTrees",
                "k=200,depth=12,leaf=2,max_features=sqrt",
                _preprocessed(_forest("ET"), k=200),
            ),
            Candidate(
                "engineered_rich_trace_shape",
                "SVR-RBF",
                "k=500,C=0.1,epsilon=0.1,gamma=scale",
                _preprocessed(SVR(C=0.1, epsilon=0.1), k=500, scale=True),
            ),
            Candidate(
                "engineered_rich_trace_shape",
                "QualityHurdleSVM",
                "C=0.3,gamma=0.0005",
                QualityHurdleRegressor(c=0.3, gamma=0.0005, random_state=SEED),
            ),
        ]
    )
    return rows


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def run_benchmark(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    feature_sets, dataset = build_feature_sets(data_dir)
    y = dataset.response.to_numpy(dtype=float)
    bad = dataset.bad_label.to_numpy(dtype=int)
    groups = dataset.groups.to_numpy(dtype=str)
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            dataset.features, bad, groups
        )
    )
    rows: List[Dict[str, Any]] = []
    all_candidates = candidates()
    for number, candidate in enumerate(all_candidates, start=1):
        X = feature_sets[candidate.feature_set]
        started = time.perf_counter()
        prediction = np.full(len(y), np.nan, dtype=float)
        fold_r2: List[float] = []
        status = "OK"
        error = ""
        try:
            for train_index, validation_index in splits:
                fitted = clone(candidate.estimator).fit(X.iloc[train_index], y[train_index])
                fold_prediction = fitted.predict(X.iloc[validation_index])
                prediction[validation_index] = fold_prediction
                fold_r2.append(float(r2_score(y[validation_index], fold_prediction)))
            result = _metrics(y, prediction)
        except Exception as exc:
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
            result = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
        row = {
            "feature_set": candidate.feature_set,
            "feature_count": int(X.shape[1]),
            "model": candidate.model,
            "parameters": candidate.parameters,
            **result,
            "fold_r2_min": min(fold_r2) if fold_r2 else np.nan,
            "fold_r2_max": max(fold_r2) if fold_r2 else np.nan,
            "seconds": time.perf_counter() - started,
            "status": status,
            "error": error,
        }
        rows.append(row)
        print(f"[{number}/{len(all_candidates)}] {json.dumps(row, default=str)}", flush=True)

    result_frame = pd.DataFrame(rows).sort_values("r2", ascending=False, na_position="last")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_dir / "r2_experiment_benchmark.csv", index=False)
    engineered_nine = result_frame.loc[
        result_frame["feature_set"].eq("engineered_key_numbers")
        & result_frame["model"].isin(ENGINEERED_NINE_ALGORITHMS)
        & result_frame["status"].eq("OK")
    ]
    best_engineered_nine = (
        engineered_nine.sort_values("r2", ascending=False)
        .drop_duplicates("model")
        .set_index("model")
        .reindex(ENGINEERED_NINE_ALGORITHMS)
        .reset_index()
    )
    best_engineered_nine.to_csv(
        output_dir / "r2_engineered_nine_algorithm_summary.csv", index=False
    )
    metadata = {
        "validation": "5-fold StratifiedGroupKFold by manufacturing lot",
        "random_state": SEED,
        "wafers": int(len(y)),
        "lots": int(pd.Series(groups).nunique()),
        "feature_sets": {key: int(value.shape[1]) for key, value in feature_sets.items()},
        "candidate_count": int(len(rows)),
        "engineered_algorithm_families": list(ENGINEERED_NINE_ALGORITHMS),
    }
    (output_dir / "r2_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result_frame


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = run_benchmark(root / "EquipmentData", root / "output" / "equipment_quality")
    print("\nTOP RESULTS\n" + results.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
