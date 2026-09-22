"""Leakage-resistant PHM 2016 CMP virtual-metrology benchmark.

The public PHM Society challenge maps upstream CMP process traces to the
downstream average material-removal rate measured from wafer thickness before
and after polishing.  The official files do not expose manufacturing lots, so
this benchmark uses chronological source-file campaigns as the closest honest
batch analogue.  Model selection uses only earlier campaigns; the latest four
campaigns are an untouched prospective test set.

WAFER_ID, source-file identity, and absolute timestamps are deliberately
excluded from predictors.  They are retained only for alignment and splitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SEED = 42
KEY_COLUMNS = ["WAFER_ID", "STAGE"]
IDENTITY_COLUMNS = {
    "MACHINE_ID",
    "MACHINE_DATA",
    "TIMESTAMP",
    "WAFER_ID",
    "STAGE",
    "CHAMBER",
    "_source_index",
}
CRITICAL_SIGNALS = [
    "PRESSURIZED_CHAMBER_PRESSURE",
    "MAIN_OUTER_AIR_BAG_PRESSURE",
    "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE",
    "RIPPLE_AIR_BAG_PRESSURE",
    "EDGE_AIR_BAG_PRESSURE",
    "SLURRY_FLOW_LINE_A",
    "SLURRY_FLOW_LINE_B",
    "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION",
    "STAGE_ROTATION",
    "HEAD_ROTATION",
]


@dataclass(frozen=True)
class CMPDataset:
    features: pd.DataFrame
    target: np.ndarray
    campaign: np.ndarray
    identity: pd.DataFrame
    metadata: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_dataset_root(root: Path) -> Tuple[Path, Path]:
    labels = sorted(root.rglob("CMP-training-removalrate.csv"))
    if len(labels) != 1:
        raise ValueError(
            f"Expected one CMP-training-removalrate.csv below {root}, found {len(labels)}"
        )
    trace_dirs = [
        path
        for path in root.rglob("training")
        if any(path.glob("CMP-training-*.csv"))
    ]
    if len(trace_dirs) != 1:
        raise ValueError(
            f"Expected one directory of CMP training traces below {root}, "
            f"found {len(trace_dirs)}"
        )
    return labels[0], trace_dirs[0]


def _load_traces(trace_dir: Path) -> Tuple[pd.DataFrame, List[Path]]:
    paths = sorted(trace_dir.glob("CMP-training-[0-9][0-9][0-9].csv"))
    if len(paths) != 185:
        raise ValueError(f"Expected 185 PHM16 training files, found {len(paths)}")
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        source_index = int(path.stem.rsplit("-", 1)[-1])
        frame["_source_index"] = source_index
        frames.append(frame)
    traces = pd.concat(frames, ignore_index=True)
    required = set(KEY_COLUMNS).union(
        {"TIMESTAMP", "CHAMBER", "MACHINE_DATA"}
    )
    missing = required.difference(traces.columns)
    if missing:
        raise ValueError(f"PHM16 traces are missing columns: {sorted(missing)}")
    traces["STAGE"] = traces["STAGE"].astype(str).str.strip()
    for column in traces.columns.difference(["STAGE"]):
        traces[column] = pd.to_numeric(traces[column], errors="coerce")
    if traces[KEY_COLUMNS].isna().any().any():
        raise ValueError("PHM16 traces contain missing wafer/stage identifiers")
    return traces, paths


def _flatten_columns(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    result = frame.copy()
    result.columns = [
        prefix + "__".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else prefix + str(column)
        for column in result.columns
    ]
    return result


def _aggregate_features(traces: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = traces.sort_values(KEY_COLUMNS + ["TIMESTAMP", "_source_index"]).copy()
    groups = ordered.groupby(KEY_COLUMNS, sort=False)
    signal_columns = [
        column
        for column in ordered.columns
        if column not in IDENTITY_COLUMNS
        and pd.api.types.is_numeric_dtype(ordered[column])
    ]

    statistics = groups[signal_columns].agg(
        ["mean", "std", "min", "max", "median", "first", "last"]
    )
    statistics = _flatten_columns(statistics, "signal__")
    quantiles = pd.concat(
        [
            groups[signal_columns]
            .quantile(quantile)
            .add_suffix(f"__q{int(quantile * 100):02d}")
            for quantile in (0.25, 0.75)
        ],
        axis=1,
    ).add_prefix("signal__")

    first_timestamp = groups["TIMESTAMP"].min()
    last_timestamp = groups["TIMESTAMP"].max()
    general = pd.DataFrame(
        {
            "sample_count": groups.size(),
            "process_duration": last_timestamp - first_timestamp,
            "timestamp_gap_mean": groups["TIMESTAMP"].apply(
                lambda values: values.sort_values().diff().mean()
            ),
            "timestamp_gap_std": groups["TIMESTAMP"].apply(
                lambda values: values.sort_values().diff().std()
            ),
        }
    )

    ordered["_row_number"] = groups.cumcount()
    ordered["_group_size"] = groups["TIMESTAMP"].transform("size")
    ordered["_phase"] = np.minimum(
        (3 * ordered["_row_number"] / ordered["_group_size"].clip(lower=1)).astype(int),
        2,
    )
    phase_columns = [column for column in CRITICAL_SIGNALS if column in ordered.columns]
    phase = (
        ordered.groupby(KEY_COLUMNS + ["_phase"], sort=False)[phase_columns]
        .agg(["mean", "std"])
        .unstack("_phase")
    )
    phase = _flatten_columns(phase, "phase__")

    chamber = ordered[KEY_COLUMNS + ["CHAMBER", "TIMESTAMP"] + phase_columns].copy()
    chamber["_chamber_count"] = 1.0
    chamber_summary = chamber.groupby(KEY_COLUMNS + ["CHAMBER"], sort=False).agg(
        {"_chamber_count": "sum", "TIMESTAMP": ["min", "max"], **{
            column: "mean" for column in phase_columns
        }}
    )
    chamber_summary = _flatten_columns(chamber_summary)
    chamber_summary["chamber_duration"] = (
        chamber_summary["TIMESTAMP__max"] - chamber_summary["TIMESTAMP__min"]
    )
    chamber_summary = chamber_summary.drop(
        columns=["TIMESTAMP__min", "TIMESTAMP__max"]
    ).unstack("CHAMBER")
    chamber_summary = _flatten_columns(chamber_summary, "chamber__")

    context_frames: List[pd.DataFrame] = []
    for column in ("MACHINE_DATA", "CHAMBER"):
        values = sorted(ordered[column].dropna().unique())
        for value in values:
            fraction = groups[column].apply(lambda series, target=value: (series == target).mean())
            context_frames.append(fraction.rename(f"context__{column.lower()}_{value:g}_fraction"))
    stage_indicator = (
        ordered[KEY_COLUMNS]
        .drop_duplicates()
        .set_index(KEY_COLUMNS)
        .index.get_level_values("STAGE")
        .map({"A": 0.0, "B": 1.0})
    )
    stage_frame = pd.DataFrame(
        {"context__stage_b": np.asarray(stage_indicator, dtype=float)},
        index=ordered[KEY_COLUMNS].drop_duplicates().set_index(KEY_COLUMNS).index,
    )
    context_frames.append(stage_frame["context__stage_b"])
    context = pd.concat(context_frames, axis=1)

    if {
        "PRESSURIZED_CHAMBER_PRESSURE",
        "WAFER_ROTATION",
        "HEAD_ROTATION",
        "STAGE_ROTATION",
    }.issubset(ordered.columns):
        ordered["_preston_proxy"] = ordered["PRESSURIZED_CHAMBER_PRESSURE"] * (
            ordered["WAFER_ROTATION"].abs()
            + ordered["HEAD_ROTATION"].abs()
            + ordered["STAGE_ROTATION"].abs()
        )
        physics = groups["_preston_proxy"].agg(["mean", "std", "max"])
        physics.columns = [f"physics__preston_proxy__{column}" for column in physics.columns]
        physics["physics__pressure_speed_exposure"] = (
            physics["physics__preston_proxy__mean"] * general["process_duration"]
        )
    else:
        physics = pd.DataFrame(index=statistics.index)

    features = pd.concat(
        [statistics, quantiles, general, phase, chamber_summary, context, physics],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    features = features.loc[:, ~features.columns.duplicated()].sort_index()

    earliest_source = groups["_source_index"].min().rename("earliest_source")
    earliest_wafer_source = ordered.groupby("WAFER_ID")["_source_index"].min()
    identity = earliest_source.reset_index()
    identity["campaign"] = (
        identity["WAFER_ID"].map(earliest_wafer_source).astype(int) // 10
    )
    return features, identity


def load_cmp_dataset(
    root: Path,
    *,
    max_valid_mrr: float | None = 300.0,
) -> CMPDataset:
    label_path, trace_dir = _discover_dataset_root(root)
    traces, trace_paths = _load_traces(trace_dir)
    features, identity = _aggregate_features(traces)

    labels = pd.read_csv(label_path)
    labels["STAGE"] = labels["STAGE"].astype(str).str.strip()
    labels["WAFER_ID"] = pd.to_numeric(labels["WAFER_ID"], errors="raise")
    labels["AVG_REMOVAL_RATE"] = pd.to_numeric(
        labels["AVG_REMOVAL_RATE"], errors="raise"
    )
    if labels.duplicated(KEY_COLUMNS).any():
        raise ValueError("PHM16 labels contain duplicate wafer/stage keys")

    feature_frame = features.reset_index()
    combined = labels.merge(
        identity,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    ).merge(
        feature_frame,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(combined) != len(labels):
        raise ValueError(
            f"Only {len(combined)} of {len(labels)} removal-rate labels align to traces"
        )
    outliers = combined.loc[
        combined["AVG_REMOVAL_RATE"].ge(max_valid_mrr)
        if max_valid_mrr is not None
        else np.zeros(len(combined), dtype=bool),
        ["WAFER_ID", "STAGE", "AVG_REMOVAL_RATE", "earliest_source", "campaign"],
    ].copy()
    if max_valid_mrr is not None:
        combined = combined.loc[combined["AVG_REMOVAL_RATE"].lt(max_valid_mrr)].copy()
    combined = combined.sort_values(["earliest_source", "WAFER_ID", "STAGE"]).reset_index(
        drop=True
    )
    feature_columns = [
        column
        for column in combined.columns
        if column
        not in {
            "WAFER_ID",
            "STAGE",
            "AVG_REMOVAL_RATE",
            "earliest_source",
            "campaign",
        }
    ]
    X = combined[feature_columns].astype(float)
    y = combined["AVG_REMOVAL_RATE"].to_numpy(float)
    groups = combined["campaign"].to_numpy(int)
    metadata = {
        "dataset": "PHM Society 2016 CMP Data Challenge",
        "official_specification": "https://phmsociety.org/wp-content/uploads/2016/05/PHM16DataChallengeCFP.pdf",
        "transport_mirror_doi": "10.5281/zenodo.19803296",
        "label_sha256": _sha256(label_path),
        "trace_files": len(trace_paths),
        "trace_rows": int(len(traces)),
        "wafer_stage_samples": int(len(combined)),
        "physical_wafers": int(combined["WAFER_ID"].nunique()),
        "campaigns": int(combined["campaign"].nunique()),
        "features": int(X.shape[1]),
        "feature_exclusions": ["WAFER_ID", "source-file identity", "absolute TIMESTAMP"],
        "campaign_definition": (
            "floor(physical wafer earliest chronological source-file index / 10)"
        ),
        "mrr_quality_ceiling": max_valid_mrr,
        "mrr_outliers_removed": int(len(outliers)),
        "mrr_outlier_rows": outliers.to_dict("records"),
        "mrr_outlier_reference": "https://doi.org/10.3390/app122211478",
        "mrr_outlier_rule": (
            "Fixed before modeling from published PHM16 evidence: four labels above "
            "4000 are outliers while ordinary MRR values are below 170; ceiling 300."
            if max_valid_mrr is not None
            else "No target-quality filtering"
        ),
    }
    return CMPDataset(
        features=X,
        target=y,
        campaign=groups,
        identity=combined[["WAFER_ID", "STAGE", "earliest_source", "campaign"]],
        metadata=metadata,
    )


def _scaled_pipeline(model: Any, selected_features: int | str = "all") -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
    ]
    if selected_features != "all":
        steps.append(("selector", SelectKBest(f_regression, k=selected_features)))
    steps.extend([("scale", StandardScaler()), ("model", model)])
    return Pipeline(steps)


def _tree_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("variance", VarianceThreshold()),
            ("model", model),
        ]
    )


def _candidate_models(random_state: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "ridge_alpha_10": _scaled_pipeline(Ridge(alpha=10.0)),
        "pls_20": _scaled_pipeline(PLSRegression(n_components=20, max_iter=1000)),
        "rbf_svr": _scaled_pipeline(SVR(C=100.0, epsilon=0.5, gamma="scale"), 250),
        "random_forest": _tree_pipeline(
            RandomForestRegressor(
                n_estimators=500,
                max_features=0.7,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=random_state,
            )
        ),
        "extra_trees": _tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=500,
                max_features=0.8,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=random_state,
            )
        ),
        "gradient_boosting": _tree_pipeline(
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=3,
                loss="huber",
                random_state=random_state,
            )
        ),
        "hist_gradient_boosting": _tree_pipeline(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=random_state,
            )
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = _tree_pipeline(
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=10.0,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=random_state,
            )
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = _tree_pipeline(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=15,
                reg_lambda=5.0,
                verbosity=-1,
                n_jobs=1,
                random_state=random_state,
            )
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor

        models["catboost"] = _tree_pipeline(
            CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.03,
                loss_function="RMSE",
                l2_leaf_reg=10.0,
                verbose=False,
                allow_writing_files=False,
                random_seed=random_state,
                thread_count=1,
            )
        )
    except ImportError:
        pass
    return models


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def run_benchmark(
    data_root: Path,
    output_dir: Path,
    *,
    random_state: int = SEED,
    final_campaign_count: int = 4,
    max_valid_mrr: float | None = 300.0,
) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    started = time.perf_counter()
    dataset = load_cmp_dataset(data_root, max_valid_mrr=max_valid_mrr)
    X, y, groups = dataset.features, dataset.target, dataset.campaign
    campaigns = np.sort(np.unique(groups))
    if final_campaign_count < 1 or final_campaign_count >= len(campaigns):
        raise ValueError("final_campaign_count must leave campaigns for development")
    final_campaigns = campaigns[-final_campaign_count:]
    final_mask = np.isin(groups, final_campaigns)
    development = np.flatnonzero(~final_mask)
    final = np.flatnonzero(final_mask)
    development_groups = groups[development]
    cv = list(GroupKFold(n_splits=5).split(X.iloc[development], y[development], development_groups))
    models = _candidate_models(random_state)

    rows: List[Dict[str, Any]] = []
    development_predictions: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        prediction = np.full(len(development), np.nan)
        fold_seconds = 0.0
        try:
            for fold_number, (train_local, validation_local) in enumerate(cv, start=1):
                fold_started = time.perf_counter()
                fitted = clone(model).fit(
                    X.iloc[development[train_local]], y[development[train_local]]
                )
                prediction[validation_local] = np.asarray(
                    fitted.predict(X.iloc[development[validation_local]])
                ).reshape(-1)
                fold_seconds += time.perf_counter() - fold_started
            if not np.isfinite(prediction).all():
                raise ValueError(f"{name} produced incomplete development predictions")
        except Exception as exc:
            rows.append(
                {
                    "model": name,
                    "lane": "development_group_oof_failed",
                    "r2": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "seconds": fold_seconds,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            continue
        development_predictions[name] = prediction
        rows.append(
            {
                "model": name,
                "lane": "development_group_oof",
                **_metrics(y[development], prediction),
                "seconds": fold_seconds,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    development_rows = [
        row
        for row in rows
        if row["lane"] == "development_group_oof" and np.isfinite(row["r2"])
    ]
    selected_name = max(development_rows, key=lambda row: row["r2"])["model"]
    selected_model = clone(models[selected_name]).fit(X.iloc[development], y[development])
    final_prediction = np.asarray(selected_model.predict(X.iloc[final])).reshape(-1)
    final_metrics = _metrics(y[final], final_prediction)
    rows.append(
        {
            "model": selected_name,
            "lane": "untouched_latest_campaigns",
            **final_metrics,
            "seconds": np.nan,
        }
    )

    mean_prediction = np.full(len(final), float(y[development].mean()))
    rows.append(
        {
            "model": "development_mean",
            "lane": "untouched_latest_campaigns",
            **_metrics(y[final], mean_prediction),
            "seconds": np.nan,
        }
    )
    result = pd.DataFrame(rows)

    identity = dataset.identity.copy()
    ledger = pd.concat(
        [
            identity.iloc[development].reset_index(drop=True).assign(
                split="development_group_oof",
                actual=y[development],
                prediction=development_predictions[selected_name],
            ),
            identity.iloc[final].reset_index(drop=True).assign(
                split="untouched_latest_campaigns",
                actual=y[final],
                prediction=final_prediction,
            ),
        ],
        ignore_index=True,
    )
    metadata = {
        **dict(dataset.metadata),
        "validation_contract": (
            "Model selected by 5-fold GroupKFold on earlier chronological campaigns; "
            "one final evaluation on the latest four untouched campaigns"
        ),
        "development_campaigns": [int(value) for value in campaigns[:-final_campaign_count]],
        "final_campaigns": [int(value) for value in final_campaigns],
        "development_samples": int(len(development)),
        "final_samples": int(len(final)),
        "selected_model": selected_name,
        "selected_development_metrics": next(
            row for row in development_rows if row["model"] == selected_name
        ),
        "final_metrics": final_metrics,
        "r2_gate": 0.8,
        "r2_gate_passed": bool(final_metrics["r2"] > 0.8),
        "seconds": time.perf_counter() - started,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "phm2016_cmp_benchmark.csv", index=False)
    ledger.to_csv(output_dir / "phm2016_cmp_prediction_ledger.csv", index=False)
    if metadata["r2_gate_passed"]:
        output_model_path = output_dir / "phm2016_cmp_verified_model.joblib"
        bundled_model_path = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "phm2016_cmp_verified_model.joblib"
        )
        bundled_model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata["artifact_paths"] = {
            "evidence_model": str(output_model_path.resolve()),
            "bundled_model": str(bundled_model_path.resolve()),
        }
        final_model = clone(models[selected_name]).fit(X, y)
        bundle = {
            "model": final_model,
            "feature_columns": list(X.columns),
            "metadata": metadata,
        }
        joblib.dump(bundle, output_model_path)
        joblib.dump(bundle, bundled_model_path)
    (output_dir / "phm2016_cmp_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result, metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing the extracted PHM16 archive.",
    )
    parser.add_argument("--final-campaign-count", type=int, default=4)
    parser.add_argument(
        "--include-documented-mrr-outliers",
        action="store_true",
        help="Sensitivity lane retaining four published MRR label outliers above 4000.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root or project_root / "data" / "phm2016_cmp"
    result, metadata = run_benchmark(
        data_root,
        project_root / "output" / "phm2016_cmp",
        final_campaign_count=args.final_campaign_count,
        max_valid_mrr=None if args.include_documented_mrr_outliers else 300.0,
    )
    print("\nPHM 2016 CMP RESULTS\n" + result.to_string(index=False), flush=True)
    print("\nFINAL GATE\n" + json.dumps(metadata["final_metrics"], indent=2))


if __name__ == "__main__":
    main()
