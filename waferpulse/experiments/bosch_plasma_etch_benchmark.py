"""Lot-held-out BOSCH plasma-etch virtual-metrology benchmark.

The public BOSCH data joins 5 Hz equipment traces with downstream 89-point
oxide/silicon etch maps.  Five-fold cross-validation holds out complete
manufacturing lots, so measurements from a test lot never influence its fitted
model.

Two tasks are reported deliberately:

* ``map_point`` predicts each downstream map point from upstream process
  summaries plus the known measurement coordinate.
* ``wafer_mean`` predicts wafer-average etch from upstream process summaries
  only.  This is the stricter equipment-to-quality test.

Identifiers, lot number, wafer number, dates, and pre/post metrology are never
used as predictors.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SEED = 42
TARGETS = ("si_etch", "oxide_etch")

MODEL_CATEGORY_MAP: Mapping[str, str] = {
    "dummy_mean": "Baseline",
    "ridge": "Regularised Linear",
    "lasso": "Regularised Linear",
    "elastic_net": "Regularised Linear",
    "knn": "Instance-Based",
    "random_forest": "Bagging Ensemble",
    "extra_trees": "Bagging Ensemble",
    "hist_gradient_boosting": "Boosting Ensemble",
    "xgboost": "Boosting Ensemble",
    "lightgbm": "Boosting Ensemble",
}


@dataclass(frozen=True)
class BoschDataset:
    process: pd.DataFrame
    measurements: pd.DataFrame
    metadata: Mapping[str, Any]


def group_to_experiment_key(name: str) -> str:
    """Convert a NetCDF group name to the downstream measurement key."""

    match = re.fullmatch(r"Day_(\d{4})_(\d{2})_(\d{2})_Wafer_(\d+)", name)
    if match is None:
        raise ValueError(f"Unexpected BOSCH process group: {name}")
    year, month, day, wafer = match.groups()
    return f"{year}-{month}-{day}_{int(wafer):02d}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _slope(values: np.ndarray) -> float:
    if values.size < 2 or np.all(values == values[0]):
        return 0.0
    x = np.linspace(-1.0, 1.0, values.size)
    return float(np.polyfit(x, values, 1)[0])


def extract_process_features(
    process_path: Path,
    dictionary_path: Path,
) -> pd.DataFrame:
    """Decode and aggregate each wafer's equipment trace using ml_compute_statistic."""

    from netCDF4 import Dataset
    from ml_compute_statistic import col_stats

    with Dataset(dictionary_path) as dictionary:
        decoder = np.asarray(dictionary.variables["data"][:], dtype=float)

    rows: list[Dict[str, Any]] = []
    with Dataset(process_path) as source:
        for group_name, group in source.groups.items():
            encoded = np.asarray(group.variables["data"][:], dtype=np.uint16)
            values = decoder[encoded]
            names = np.asarray(group.variables["feature"][:], dtype=str)
            times = np.asarray(group.variables["times"][:], dtype=float)
            row: Dict[str, Any] = {
                "experiment_key": group_to_experiment_key(group_name),
                "process__sample_count": int(times.size),
                "process__duration": float(times[-1] - times[0]),
            }
            for index, name in enumerate(names):
                signal = values[:, index]
                signal = signal[np.isfinite(signal)]
                prefix = f"signal__{_safe_name(name)}"
                if signal.size == 0:
                    for statistic in (
                        "mean", "std", "median", "iqr", "min", "max",
                        "missing_rate", "outlier_rate", "p01", "p05", "p25", "p50", "p75", "p95", "p99",
                        "q25", "q75", "first", "last", "slope",
                    ):
                        row[f"{prefix}__{statistic}"] = np.nan
                    continue
                s_series = pd.Series(signal)
                stats = col_stats(s_series)
                for stat_k, stat_v in stats.items():
                    row[f"{prefix}__{stat_k}"] = stat_v
                # Preserve q25 and q75 for backward compatibility
                row[f"{prefix}__q25"] = stats.get("p25", np.nan)
                row[f"{prefix}__q75"] = stats.get("p75", np.nan)
                row[f"{prefix}__first"] = float(signal[0])
                row[f"{prefix}__last"] = float(signal[-1])
                row[f"{prefix}__slope"] = _slope(signal)
            rows.append(row)
    return pd.DataFrame(rows).sort_values("experiment_key").reset_index(drop=True)


def load_dataset(root: Path, refresh_cache: bool = False) -> BoschDataset:
    required = {
        "process": root / "Process_data.nc",
        "dictionary": root / "Dictionary_process.nc",
        "measurements": root / "Si_Oxide_etch_89_points.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing BOSCH files: {missing}")

    cache = root / "bosch_process_wafer_features.parquet"
    if cache.is_file() and not refresh_cache:
        process = pd.read_parquet(cache)
    else:
        process = extract_process_features(required["process"], required["dictionary"])
        process.to_parquet(cache, index=False)
    numeric_columns = process.columns.difference(["experiment_key"])
    sufficiently_observed = [
        column for column in numeric_columns if process[column].notna().sum() >= 20
    ]
    process = process[["experiment_key", *sufficiently_observed]]

    measurements = pd.read_csv(required["measurements"])
    needed = {
        "experiment_key", "lot_number", "wafer_number", "X", "Y", *TARGETS,
    }
    absent = needed.difference(measurements.columns)
    if absent:
        raise ValueError(f"BOSCH measurement columns missing: {sorted(absent)}")
    matched = set(process["experiment_key"]).intersection(measurements["experiment_key"])
    if not matched:
        raise ValueError("No BOSCH process traces match downstream measurements")
    metadata = {
        "process_traces": int(len(process)),
        "measured_wafers": int(measurements["experiment_key"].nunique()),
        "matched_wafers": int(len(matched)),
        "lots": int(measurements["lot_number"].nunique()),
        "measurement_rows": int(len(measurements)),
    }
    return BoschDataset(process=process, measurements=measurements, metadata=metadata)


def coordinate_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame["X"].astype(float) / 100_000.0
    y = frame["Y"].astype(float) / 100_000.0
    radius2 = x.pow(2) + y.pow(2)
    angle = np.arctan2(y, x)
    return pd.DataFrame(
        {
            "coordinate__x": x,
            "coordinate__y": y,
            "coordinate__radius": np.sqrt(radius2),
            "coordinate__radius2": radius2,
            "coordinate__xy": x * y,
            "coordinate__x2_minus_y2": x.pow(2) - y.pow(2),
            "coordinate__sin_angle": np.sin(angle),
            "coordinate__cos_angle": np.cos(angle),
        },
        index=frame.index,
    )


def model_candidates(feature_count: int) -> Mapping[str, Pipeline]:
    selected = max(1, min(80, feature_count))
    preprocessing = [
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(f_regression, k=selected)),
    ]
    ridge_pipe = Pipeline(
        [
            *preprocessing,
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=20.0, random_state=SEED)),
        ]
    )
    return {
        "dummy_mean": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor())]
        ),
        # 1. Regularised Linear Models (Report Table 3-5)
        "ridge": ridge_pipe,
        "ridge_selected": ridge_pipe,  # Alias for backward compatibility
        "lasso": Pipeline(
            [
                *preprocessing,
                ("scale", StandardScaler()),
                ("model", Lasso(alpha=0.01, random_state=SEED)),
            ]
        ),
        "elastic_net": Pipeline(
            [
                *preprocessing,
                ("scale", StandardScaler()),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED)),
            ]
        ),
        # 2. Instance-Based Model (Report Table 3-5)
        "knn": Pipeline(
            [
                *preprocessing,
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=15, weights="distance")),
            ]
        ),
        # 3. Bagging Ensemble Models (Report Table 3-5)
        "random_forest": Pipeline(
            [
                *preprocessing,
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        min_samples_leaf=2,
                        max_features=0.7,
                        n_jobs=1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                *preprocessing,
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=120,
                        min_samples_leaf=2,
                        max_features=0.7,
                        n_jobs=1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        # 4. Boosting Models (Report Table 3-5)
        "hist_gradient_boosting": Pipeline(
            [
                *preprocessing,
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=150,
                        learning_rate=0.05,
                        max_depth=4,
                        min_samples_leaf=5,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                *preprocessing,
                (
                    "model",
                    XGBRegressor(
                        n_estimators=220,
                        learning_rate=0.04,
                        max_depth=4,
                        min_child_weight=5,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        reg_lambda=2.0,
                        objective="reg:squarederror",
                        n_jobs=1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                *preprocessing,
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=220,
                        learning_rate=0.04,
                        num_leaves=15,
                        min_child_samples=15,
                        reg_lambda=2.0,
                        verbosity=-1,
                        n_jobs=1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def _group_oof_predictions(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Group cross-validation requires at least two distinct groups")
    splits = min(5, unique_groups.size)
    splitter = GroupKFold(n_splits=splits)
    predictions = np.empty_like(target, dtype=float)
    for train_index, test_index in splitter.split(features, target, groups):
        fitted = clone(estimator).fit(
            features.iloc[train_index], target[train_index]
        )
        predictions[test_index] = fitted.predict(features.iloc[test_index])
    return predictions


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def benchmark(dataset: BoschDataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    process_columns = [column for column in dataset.process if column != "experiment_key"]
    points = dataset.measurements.merge(
        dataset.process, on="experiment_key", how="inner", validate="many_to_one"
    ).reset_index(drop=True)
    coordinates = coordinate_features(points).reset_index(drop=True)
    process_at_points = points[process_columns].reset_index(drop=True)
    feature_sets = {
        "coordinate_only": coordinates,
        "process_only": process_at_points,
        "process_plus_coordinate": pd.concat([process_at_points, coordinates], axis=1),
    }

    results: list[Dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    point_groups = points["lot_number"].to_numpy()
    for target_name in TARGETS:
        actual = points[target_name].to_numpy(dtype=float)
        for feature_set, features in feature_sets.items():
            candidates = model_candidates(features.shape[1])
            if feature_set != "process_plus_coordinate" or target_name != "si_etch":
                candidates = {
                    name: candidates[name]
                    for name in ("dummy_mean", "ridge", "extra_trees")
                    if name in candidates
                }
            for model_name, model in candidates.items():
                if model_name == "ridge_selected":
                    continue  # Skip redundant alias in full benchmark loop
                predicted = _group_oof_predictions(
                    model, features, actual, point_groups
                )
                results.append(
                    {
                        "task": "map_point",
                        "target": target_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "rows": int(len(actual)),
                        "groups": int(len(np.unique(point_groups))),
                        **_metrics(actual, predicted),
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "task": "map_point",
                            "target": target_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "experiment_key": points["experiment_key"],
                            "lot_number": point_groups,
                            "actual": actual,
                            "predicted": predicted,
                        }
                    )
                )

    wafer_targets = (
        points.groupby(["experiment_key", "lot_number"], as_index=False)[list(TARGETS)]
        .mean()
        .merge(dataset.process, on="experiment_key", validate="one_to_one")
    )
    wafer_features = wafer_targets[process_columns]
    wafer_groups = wafer_targets["lot_number"].to_numpy()
    for target_name in TARGETS:
        actual = wafer_targets[target_name].to_numpy(dtype=float)
        candidates = model_candidates(wafer_features.shape[1])
        if target_name != "si_etch":
            candidates = {
                name: candidates[name]
                for name in ("dummy_mean", "ridge", "extra_trees")
                if name in candidates
            }
        for model_name, model in candidates.items():
            if model_name == "ridge_selected":
                continue
            predicted = _group_oof_predictions(
                model, wafer_features, actual, wafer_groups
            )
            results.append(
                {
                    "task": "wafer_mean",
                    "target": target_name,
                    "feature_set": "process_only",
                    "model": model_name,
                    "rows": int(len(actual)),
                    "groups": int(len(np.unique(wafer_groups))),
                    **_metrics(actual, predicted),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "task": "wafer_mean",
                        "target": target_name,
                        "feature_set": "process_only",
                        "model": model_name,
                        "experiment_key": wafer_targets["experiment_key"],
                        "lot_number": wafer_groups,
                        "actual": actual,
                        "predicted": predicted,
                    }
                )
            )

    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def run(root: Path, output: Path, refresh_cache: bool = False) -> pd.DataFrame:
    dataset = load_dataset(root, refresh_cache=refresh_cache)
    results, predictions = benchmark(dataset)
    output.mkdir(parents=True, exist_ok=True)

    # Attach algorithm category from Report Table 3-5
    results["category"] = results["model"].map(MODEL_CATEGORY_MAP).fillna("Other")

    results = results.sort_values(
        ["task", "target", "r2"], ascending=[True, True, False]
    ).reset_index(drop=True)

    # 1. Backward-compatible CSV outputs
    results.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "oof_predictions.csv", index=False)

    # 2. Clear, direct benchmark summary CSV
    wafer_si = results.loc[
        results["task"].eq("wafer_mean")
        & results["target"].eq("si_etch")
        & results["feature_set"].eq("process_only")
    ].copy().sort_values("r2", ascending=False).reset_index(drop=True)
    wafer_si.insert(0, "rank", range(1, len(wafer_si) + 1))

    map_si = results.loc[
        results["task"].eq("map_point")
        & results["target"].eq("si_etch")
        & results["feature_set"].eq("process_plus_coordinate")
    ].copy().sort_values("r2", ascending=False).reset_index(drop=True)
    map_si.insert(0, "rank", range(1, len(map_si) + 1))

    summary_rows: list[Dict[str, Any]] = []
    for sub in (wafer_si, map_si):
        for _, r in sub.iterrows():
            summary_rows.append(
                {
                    "Rank": int(r["rank"]),
                    "Category": r["category"],
                    "Model": r["model"].replace("_", " ").title(),
                    "Task": r["task"],
                    "Target": r["target"],
                    "Feature_Set": r["feature_set"],
                    "R2": round(float(r["r2"]), 4),
                    "RMSE": round(float(r["rmse"]), 4),
                    "MAE": round(float(r["mae"]), 4),
                    "Rows": int(r["rows"]),
                    "Lots": int(r["groups"]),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output / "model_benchmark_summary.csv", index=False)

    # 3. Top 3 models CSV
    top3_df = wafer_si.head(3)[[
        "rank", "category", "model", "r2", "rmse", "mae", "rows", "groups"
    ]].copy()
    top3_df.columns = ["Rank", "Category", "Model", "R2", "RMSE", "MAE", "Wafers", "Lots"]
    top3_df["Model"] = top3_df["Model"].str.replace("_", " ").str.title()
    top3_df["R2"] = top3_df["R2"].round(4)
    top3_df["RMSE"] = top3_df["RMSE"].round(4)
    top3_df["MAE"] = top3_df["MAE"].round(4)
    top3_df.to_csv(output / "top_3_models.csv", index=False)

    # 4. One Excel with multiple tabs for benchmark tables & figures
    excel_path = output / "bosch_benchmark_report.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Tab 1: Summary
        exec_summary = pd.DataFrame(
            [
                {"Item": "Dataset Name", "Value": "Bosch Plasma-Etch Quality Dataset (Zenodo 17122442)"},
                {"Item": "Data Directory", "Value": str(root)},
                {"Item": "Validation Contract", "Value": "5-Fold Manufacturing-Lot-Held-Out GroupKFold (Zero Lot Leakage)"},
                {"Item": "Total Process Traces", "Value": dataset.metadata.get("process_traces", 0)},
                {"Item": "Matched Wafers", "Value": dataset.metadata.get("matched_wafers", 0)},
                {"Item": "Manufacturing Lots", "Value": dataset.metadata.get("lots", 0)},
                {"Item": "89-Point Measurement Rows", "Value": dataset.metadata.get("measurement_rows", 0)},
                {
                    "Item": "Primary Benchmark Champion (Wafer Mean)",
                    "Value": f"{top3_df.iloc[0]['Model']} (R² = {top3_df.iloc[0]['R2']:.4f}, RMSE = {top3_df.iloc[0]['RMSE']:.4f})",
                },
                {
                    "Item": "Spatial Metrology Champion (Map Points)",
                    "Value": f"{map_si.iloc[0]['model'].replace('_', ' ').title()} (R² = {map_si.iloc[0]['r2']:.4f})",
                },
            ]
        )
        exec_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Tab 2: Top 3 Models
        top3_df.to_excel(writer, sheet_name="Top_3_Models", index=False)

        # Tab 3: Wafer_Average_Si_Etch
        wafer_export = wafer_si[[
            "rank", "category", "model", "r2", "rmse", "mae", "rows", "groups"
        ]].copy()
        wafer_export.columns = ["Rank", "Category", "Model", "R2", "RMSE", "MAE", "Wafers", "Lots"]
        wafer_export["Model"] = wafer_export["Model"].str.replace("_", " ").str.title()
        wafer_export["R2"] = wafer_export["R2"].round(4)
        wafer_export["RMSE"] = wafer_export["RMSE"].round(4)
        wafer_export["MAE"] = wafer_export["MAE"].round(4)
        wafer_export.to_excel(writer, sheet_name="Wafer_Average_Si_Etch", index=False)

        # Tab 4: Map_Point_Si_Etch
        map_export = map_si[[
            "rank", "category", "model", "feature_set", "r2", "rmse", "mae", "rows", "groups"
        ]].copy()
        map_export.columns = ["Rank", "Category", "Model", "Feature_Set", "R2", "RMSE", "MAE", "Points", "Lots"]
        map_export["Model"] = map_export["Model"].str.replace("_", " ").str.title()
        map_export["R2"] = map_export["R2"].round(4)
        map_export["RMSE"] = map_export["RMSE"].round(4)
        map_export["MAE"] = map_export["MAE"].round(4)
        map_export.to_excel(writer, sheet_name="Map_Point_Si_Etch", index=False)

        # Tab 5: All_Metrics
        results.to_excel(writer, sheet_name="All_Metrics", index=False)

    summary = {
        "validation": "Five-fold manufacturing-lot-held-out cross-validation",
        "predictor_policy": (
            "Upstream process summaries; map task additionally uses known X/Y. "
            "No identifiers, dates, lots, wafer numbers, or metrology inputs."
        ),
        "dataset": dict(dataset.metadata),
        "best_by_task_target": (
            results.sort_values("r2", ascending=False)
            .groupby(["task", "target"], as_index=False)
            .first()
            .to_dict(orient="records")
        ),
        "top_3_wafer_models": top3_df.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _generate_bosch_shap(dataset, output)
    return results


def _generate_bosch_shap(dataset: Dataset, output: Path) -> None:
    try:
        import shap
        import matplotlib.pyplot as plt

        wafer_targets = (
            dataset.measurements.groupby(["experiment_key", "lot_number"], as_index=False)["si_etch"]
            .mean()
            .merge(dataset.process, on="experiment_key")
        )
        process_cols = [c for c in dataset.process.columns if c != "experiment_key"]
        X = wafer_targets[process_cols]
        y = wafer_targets["si_etch"].values

        pipe = model_candidates(X.shape[1])["lightgbm"]
        pipe.fit(X, y)

        X_trans = pipe[:-1].transform(X)
        selector = pipe.named_steps["selector"]
        raw_cols = np.array(process_cols)[selector.get_support()]

        def clean_bosch_name(name: str) -> str:
            return name.replace("signal__", "").replace("__", " · ")

        cleaned_cols = [clean_bosch_name(c) for c in raw_cols]
        lgbm_model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X_trans)

        plt.rcParams["font.sans-serif"] = "Segoe UI", "DejaVu Sans", "Arial"
        plt.rcParams["font.size"] = 10

        fig = plt.figure(figsize=(14, 6), dpi=300)
        shap.summary_plot(
            shap_values,
            X_trans,
            feature_names=cleaned_cols,
            max_display=10,
            plot_size=None,
            show=False,
            color_bar=True,
            alpha=0.85,
        )
        ax = plt.gca()
        ax.tick_params(axis="y", labelsize=10, pad=8)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=10.5, labelpad=8)
        plt.title("SHAP Beeswarm Summary Plot — Top 10 Feature Drivers (Bosch Etch)", fontsize=12, fontweight="bold", pad=15)
        plt.subplots_adjust(left=0.25, right=0.92, top=0.92, bottom=0.12)
        fig.savefig(output / "bosch_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        fig.savefig(output / "bosch_shap_beeswarm.svg", format="svg", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = (
        Path("dataset/bosch_plasma_etch")
        if Path("dataset/bosch_plasma_etch").is_dir()
        else Path("data/bosch_plasma_etch")
    )
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument(
        "--output", type=Path, default=Path("output/bosch_plasma_etch")
    )
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    results = run(args.data_root, args.output, args.refresh_cache)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
