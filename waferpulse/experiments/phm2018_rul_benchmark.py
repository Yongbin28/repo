"""Chronological, failure-cycle-held-out PHM 2018 ion-mill RUL benchmark.

This module uses the public NASA final-validation sensor stream and released
ground truth for tool 01_M02.  It is a retrospective benchmark, not a replay of
the original hidden challenge leaderboard.

Only current and past sensor information is used.  Absolute timestamp, tool
identity, wafer/lot identity, and all TTF columns are excluded from predictors.
Models train on earlier complete failure cycles and predict the next cycle.
Consequently a fault with fewer than two observed cycles is marked not
evaluable instead of being assigned a misleading within-cycle R².
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SEED = 42
TTF_COLUMNS = (
    "TTF_FlowCool Pressure Dropped Below Limit",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "TTF_Flowcool leak",
)
IDENTITY_COLUMNS = {"time", "Tool", "Lot"}
CATEGORICAL_COLUMNS = ("stage", "recipe", "recipe_step")
HEALTH_SIGNALS = (
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "ACTUALROTATIONANGLE",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
)


def failure_cycle_ids(labels: pd.Series) -> pd.Series:
    """Number observed TTF countdowns without using future rows."""

    previous = labels.shift(1)
    reset = labels.notna() & previous.notna() & (labels > previous)
    return reset.cumsum().astype("int32")


def _validate_alignment(sensors: pd.DataFrame, labels: pd.DataFrame) -> None:
    if len(sensors) != len(labels):
        raise ValueError(
            f"PHM sensor/ground-truth row mismatch: {len(sensors)} != {len(labels)}"
        )
    if not np.array_equal(sensors["time"].to_numpy(), labels["time"].to_numpy()):
        raise ValueError("PHM sensor and ground-truth times are not row-aligned")


def load_sampled_dataset(
    sensor_path: Path,
    groundtruth_path: Path,
    stride: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load the released stream and retain an evenly sampled causal sequence."""

    if stride < 1:
        raise ValueError("stride must be at least 1")
    labels = pd.read_csv(groundtruth_path)
    absent_labels = set(("time", *TTF_COLUMNS)).difference(labels.columns)
    if absent_labels:
        raise ValueError(f"PHM TTF columns missing: {sorted(absent_labels)}")

    sensors = pd.read_csv(sensor_path)
    absent_signals = set(("time", *CATEGORICAL_COLUMNS, *HEALTH_SIGNALS)).difference(
        sensors.columns
    )
    if absent_signals:
        raise ValueError(f"PHM sensor columns missing: {sorted(absent_signals)}")
    _validate_alignment(sensors, labels)

    cycle_columns: Dict[str, pd.Series] = {}
    preserve = np.zeros(len(labels), dtype=bool)
    preserve[::stride] = True
    for index, target in enumerate(TTF_COLUMNS):
        cycle = failure_cycle_ids(labels[target])
        cycle_columns[target] = cycle
        reset = cycle.diff().fillna(0).ne(0)
        endpoint = labels[target].eq(0)
        preserve |= reset.to_numpy() | endpoint.to_numpy()

    sensors = sensors.loc[preserve].reset_index(drop=True)
    sampled_labels = labels.loc[preserve].reset_index(drop=True)
    sampled_cycles = pd.DataFrame(
        {
            target: cycle.loc[preserve].to_numpy()
            for target, cycle in cycle_columns.items()
        }
    )

    raw_numeric = [
        column
        for column in sensors.columns
        if column not in IDENTITY_COLUMNS.union(CATEGORICAL_COLUMNS)
        and pd.api.types.is_numeric_dtype(sensors[column])
    ]
    features = sensors[raw_numeric].astype("float32").copy()
    category_source = sensors[list(CATEGORICAL_COLUMNS)].astype("string")
    category_features = pd.get_dummies(
        category_source, prefix=[f"category__{name}" for name in CATEGORICAL_COLUMNS],
        dtype=float,
    )

    selected_health = [column for column in HEALTH_SIGNALS if column in features]
    causal_parts = [features, category_features]
    timeline = sensors["time"].astype(float)
    causal_parts.append(
        pd.DataFrame(
            {
                "causal__time_since_previous_sample": timeline.diff().fillna(0.0),
                "causal__runnum_change": sensors["runnum"].astype(float).diff().fillna(0.0),
            }
        )
    )
    for lag in (1, 5):
        difference = features[selected_health].diff(lag)
        difference.columns = [f"causal__{column}__diff_{lag}" for column in selected_health]
        causal_parts.append(difference)
    for window in (5, 25):
        rolling = features[selected_health].rolling(window=window, min_periods=1)
        mean = rolling.mean()
        mean.columns = [f"causal__{column}__mean_{window}" for column in selected_health]
        std = rolling.std().fillna(0.0)
        std.columns = [f"causal__{column}__std_{window}" for column in selected_health]
        causal_parts.extend([mean, std])

    feature_frame = pd.concat(causal_parts, axis=1).replace([np.inf, -np.inf], np.nan)
    sufficiently_observed = feature_frame.notna().sum().ge(20)
    feature_frame = feature_frame.loc[:, sufficiently_observed]
    metadata = {
        "raw_rows": int(len(labels)),
        "sampled_rows": int(len(feature_frame)),
        "stride": int(stride),
        "feature_count": int(feature_frame.shape[1]),
        "time_start": int(labels["time"].min()),
        "time_end": int(labels["time"].max()),
        "faults": {
            target: {
                "labeled_rows": int(labels[target].notna().sum()),
                "observed_cycles": int(cycle_columns[target][labels[target].notna()].nunique()),
                "observed_failures": int(labels[target].eq(0).sum()),
            }
            for target in TTF_COLUMNS
        },
    }
    target_frame = pd.concat(
        [sampled_labels[["time", *TTF_COLUMNS]], sampled_cycles.add_prefix("cycle__")],
        axis=1,
    )
    return feature_frame, target_frame, metadata


def model_candidates(feature_count: int) -> Mapping[str, Any]:
    selected = max(1, min(60, feature_count))

    def pipeline(model: Any, scaled: bool = False) -> Pipeline:
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_regression, k=selected)),
        ]
        if scaled:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)

    return {
        "dummy_median": pipeline(DummyRegressor(strategy="median")),
        "ridge": pipeline(Ridge(alpha=20.0), scaled=True),
        "knn": pipeline(
            KNeighborsRegressor(n_neighbors=15, weights="distance"), scaled=True
        ),
        "random_forest": pipeline(
            RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=3,
                max_features=0.7,
                n_jobs=1,
                random_state=SEED,
            )
        ),
        "extra_trees": pipeline(
            ExtraTreesRegressor(
                n_estimators=150,
                min_samples_leaf=3,
                max_features=0.7,
                n_jobs=1,
                random_state=SEED,
            )
        ),
        "gradient_boosting": pipeline(
            GradientBoostingRegressor(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=5,
                random_state=SEED,
            )
        ),
        "adaboost": pipeline(
            AdaBoostRegressor(
                n_estimators=180,
                learning_rate=0.04,
                loss="square",
                random_state=SEED,
            )
        ),
        "xgboost": pipeline(
            XGBRegressor(
                n_estimators=250,
                learning_rate=0.035,
                max_depth=4,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=3.0,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=SEED,
            )
        ),
        "lightgbm": pipeline(
            LGBMRegressor(
                n_estimators=250,
                learning_rate=0.035,
                num_leaves=15,
                min_child_samples=20,
                reg_lambda=3.0,
                verbosity=-1,
                n_jobs=1,
                random_state=SEED,
            )
        ),
        "catboost": pipeline(
            CatBoostRegressor(
                iterations=250,
                depth=5,
                learning_rate=0.035,
                l2_leaf_reg=5.0,
                loss_function="RMSE",
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
                random_seed=SEED,
            )
        ),
    }


def expanding_cycle_splits(cycles: np.ndarray) -> Sequence[tuple[np.ndarray, np.ndarray]]:
    """Train on earlier cycles and test the immediately following cycle."""

    ordered = np.sort(np.unique(cycles))
    minimum_training_cycles = 3 if len(ordered) >= 5 else 1
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for position in range(minimum_training_cycles, len(ordered)):
        train_cycles = ordered[:position]
        test_cycle = ordered[position]
        train_index = np.flatnonzero(np.isin(cycles, train_cycles))
        test_index = np.flatnonzero(cycles == test_cycle)
        if train_index.size and test_index.size:
            splits.append((train_index, test_index))
    return splits


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def benchmark(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Dict[str, Any]]]:
    results: list[Dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    unevaluable: list[Dict[str, Any]] = []
    for target_name in TTF_COLUMNS:
        valid = targets[target_name].notna().to_numpy()
        target_features = features.loc[valid].reset_index(drop=True)
        actual = targets.loc[valid, target_name].to_numpy(dtype=float)
        cycles = targets.loc[valid, f"cycle__{target_name}"].to_numpy(dtype=int)
        times = targets.loc[valid, "time"].to_numpy(dtype=np.int64)
        splits = expanding_cycle_splits(cycles)
        if not splits:
            unevaluable.append(
                {
                    "target": target_name,
                    "reason": "fewer than two observed labeled failure cycles",
                    "labeled_rows_after_sampling": int(len(actual)),
                    "observed_cycles": int(len(np.unique(cycles))),
                }
            )
            continue
        for model_name, model in model_candidates(target_features.shape[1]).items():
            predicted = np.full(actual.shape, np.nan, dtype=float)
            for train_index, test_index in splits:
                fitted = clone(model)
                fitted.fit(target_features.iloc[train_index], actual[train_index])
                predicted[test_index] = fitted.predict(target_features.iloc[test_index])
            evaluated = np.isfinite(predicted)
            results.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "validation": "expanding_failure_cycle",
                    "rows": int(evaluated.sum()),
                    "evaluated_cycles": int(len(splits)),
                    **_metrics(actual[evaluated], predicted[evaluated]),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "target": target_name,
                        "model": model_name,
                        "time": times[evaluated],
                        "cycle": cycles[evaluated],
                        "actual": actual[evaluated],
                        "predicted": predicted[evaluated],
                    }
                )
            )
    result_frame = pd.DataFrame(results)
    prediction_frame = (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    )
    return result_frame, prediction_frame, unevaluable


def run(
    data_root: Path,
    output: Path,
    stride: int = 100,
) -> pd.DataFrame:
    sensor_path = data_root / "01_M02_DC_score.csv"
    groundtruth_path = data_root / "01_M02_DC_groundtruth.csv"
    missing = [str(path) for path in (sensor_path, groundtruth_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PHM 2018 files: {missing}")
    features, targets, metadata = load_sampled_dataset(
        sensor_path, groundtruth_path, stride=stride
    )
    results, predictions, unevaluable = benchmark(features, targets)
    output.mkdir(parents=True, exist_ok=True)
    if not results.empty:
        results = results.sort_values(["target", "r2"], ascending=[True, False])
    results.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "cycle_holdout_predictions.csv", index=False)
    summary = {
        "source": "NASA-released PHM 2018 final validation stream, tool 01_M02",
        "validation": (
            "Expanding chronological failure-cycle holdout; early cycles train, "
            "the immediately following cycle tests."
        ),
        "predictor_policy": (
            "Current sensors/settings and causal past-only lag/rolling features. "
            "Absolute time, Tool, Lot, and all TTF values excluded."
        ),
        "metadata": metadata,
        "unevaluable": unevaluable,
        "best_by_target": (
            results.sort_values("r2", ascending=False)
            .groupby("target", as_index=False)
            .first()
            .to_dict(orient="records")
            if not results.empty
            else []
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/phm2018_ion_mill"))
    parser.add_argument("--stride", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    results = run(args.data_root, args.output, stride=args.stride)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
