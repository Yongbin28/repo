"""Unseen-lot benchmark using label-free complete-lot sensor context.

For each wafer, this experiment augments its 56-sensor temporal summaries with
the mean, standard deviation, deviation, and z-score of its incoming lot. Lot
aggregates use sensor inputs only, so validation-lot responses remain unseen.
The deployment contract is batch scoring after all wafers in a lot are present.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
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
from sklearn.svm import SVC

from waferpulse.tools.equipment_data import load_equipment_dataset
from waferpulse.tools.equipment_modeling import QualityHurdleForestBlendRegressor


SEED = 42


def build_lot_context(features: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    frame = features.copy()
    frame["_lot"] = groups.to_numpy()
    feature_columns = [column for column in frame if column != "_lot"]
    grouped = frame.groupby("_lot", sort=False)[feature_columns]
    lot_mean = grouped.transform("mean")
    lot_std = grouped.transform("std").fillna(0.0)
    deviation = frame[feature_columns] - lot_mean
    zscore = (deviation / lot_std.where(lot_std.abs() > 1e-12)).fillna(0.0)
    context = pd.concat(
        [
            frame[feature_columns].add_prefix("wafer__"),
            lot_mean.add_prefix("lot_mean__"),
            lot_std.add_prefix("lot_std__"),
            deviation.add_prefix("lot_delta__"),
            zscore.add_prefix("lot_z__"),
        ],
        axis=1,
    )
    return context.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")


def _classifier(model: Any, k: int | str, scale: bool = False) -> Pipeline:
    steps: List[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        ),
        ("variance", VarianceThreshold()),
    ]
    if k != "all":
        steps.append(("select", SelectKBest(f_classif, k=k)))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def candidates() -> Dict[str, Pipeline]:
    rows: Dict[str, Pipeline] = {}
    for k, c in ((200, 0.1), (200, 0.3), (500, 0.1), (500, 0.3), (800, 0.3), (800, 1.0)):
        rows[f"lot_context_svm_k{k}_c{c}"] = _classifier(
            SVC(
                C=c,
                gamma="scale",
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=SEED,
            ),
            k,
            scale=True,
        )
    for k in (200, 500, 800):
        rows[f"lot_context_extra_k{k}"] = _classifier(
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=SEED,
            ),
            k,
        )
        rows[f"lot_context_rf_k{k}"] = _classifier(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=SEED,
            ),
            k,
        )
    return rows


def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def run_benchmark(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    dataset = load_equipment_dataset(
        data_dir,
        stage_mode="both",
        max_invalid_rate=1.0,
    )
    X = build_lot_context(dataset.features, dataset.groups)
    response = dataset.response.to_numpy(float)
    bad = dataset.bad_label.to_numpy(int)
    groups = dataset.groups.to_numpy(str)
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            X, bad, groups
        )
    )
    baseline_prediction = np.full(len(response), np.nan)
    for train, validation in folds:
        baseline = QualityHurdleForestBlendRegressor(random_state=SEED)
        baseline.fit(dataset.features.iloc[train], response[train])
        baseline_prediction[validation] = baseline.predict(dataset.features.iloc[validation])

    rows: List[Dict[str, Any]] = [
        {
            "experiment": "production_baseline",
            "classifier": "production",
            **_regression_metrics(response, baseline_prediction),
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "seconds": 0.0,
        }
    ]
    candidate_count = len(candidates())
    for number, (name, template) in enumerate(candidates().items(), start=1):
        started = time.perf_counter()
        probability = np.full(len(response), np.nan)
        hurdle_by_scale = {
            scale: np.full(len(response), np.nan) for scale in (0.5, 0.75, 1.0, 1.25, 1.5)
        }
        for train, validation in folds:
            fitted = template.fit(X.iloc[train], bad[train])
            fold_probability = fitted.predict_proba(X.iloc[validation])[:, 1]
            probability[validation] = fold_probability
            good_mean = float(response[train][bad[train] == 0].mean())
            bad_mean = float(response[train][bad[train] == 1].mean())
            for scale in hurdle_by_scale:
                calibrated = np.clip(fold_probability * scale, 0.0, 1.0)
                hurdle_by_scale[scale][validation] = good_mean + calibrated * (
                    bad_mean - good_mean
                )
        roc_auc = float(roc_auc_score(bad, probability))
        pr_auc = float(average_precision_score(bad, probability))
        for scale, hurdle in hurdle_by_scale.items():
            rows.append(
                {
                    "experiment": f"{name}__hurdle_scale_{scale}",
                    "classifier": name,
                    **_regression_metrics(response, hurdle),
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "seconds": time.perf_counter() - started,
                }
            )
            for hurdle_weight in (0.25, 0.50, 0.75):
                blend = hurdle_weight * hurdle + (1.0 - hurdle_weight) * baseline_prediction
                rows.append(
                    {
                        "experiment": (
                            f"{name}__scale_{scale}__hurdle_weight_{hurdle_weight}"
                        ),
                        "classifier": name,
                        **_regression_metrics(response, blend),
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "seconds": time.perf_counter() - started,
                    }
                )
        print(
            f"[{number}/{candidate_count}] "
            + json.dumps({"classifier": name, "roc_auc": roc_auc, "pr_auc": pr_auc}),
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values("r2", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "lot_context_benchmark.csv", index=False)
    (output_dir / "lot_context_metadata.json").write_text(
        json.dumps(
            {
                "validation": "5-fold StratifiedGroupKFold by manufacturing lot",
                "deployment_contract": "Complete-lot batch sensor context; no response labels used",
                "wafers": len(response),
                "lots": int(pd.Series(groups).nunique()),
                "features": int(X.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run_benchmark(root / "EquipmentData", root / "output" / "equipment_quality")
    print("\nLOT CONTEXT RESULTS\n" + result.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
