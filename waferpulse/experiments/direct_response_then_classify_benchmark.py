"""Predict EquipmentData response first, then derive good/bad from prediction.

No classifier, true class, bad probability, or class mean is supplied to the
regression models.  Each model produces lot-grouped out-of-fold response
predictions.  The fixed documented response threshold is applied only after
prediction:

    predicted_bad = predicted_response > 0.75

This directly tests the deployment flow requested for the quality gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from waferpulse.core.equipment_contracts import DEFAULT_RESPONSE_THRESHOLD
from waferpulse.experiments.equipment_r2_benchmark import build_feature_sets


SEED = 42


def response_to_bad(
    response: np.ndarray | pd.Series,
    threshold: float = DEFAULT_RESPONSE_THRESHOLD,
) -> np.ndarray:
    """Apply the source's strict greater-than response rule."""

    return (np.asarray(response, dtype=float) > threshold).astype(int)


def _pipeline(model: Any, selected_features: int, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
        ("selector", SelectKBest(f_regression, k=selected_features)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def model_candidates(feature_count: int) -> Mapping[str, Pipeline]:
    selected_120 = min(120, feature_count)
    selected_200 = min(200, feature_count)
    return {
        "Ridge": _pipeline(Ridge(alpha=1000.0), selected_120, scale=True),
        "SVR-RBF": _pipeline(
            SVR(C=0.3, epsilon=0.1, gamma="scale"), selected_120, scale=True
        ),
        "KNN": _pipeline(
            KNeighborsRegressor(n_neighbors=15, weights="distance"),
            selected_120,
            scale=True,
        ),
        "RandomForest": _pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=1,
                random_state=SEED,
            ),
            selected_120,
        ),
        "ExtraTrees": _pipeline(
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                max_features=0.3,
                n_jobs=1,
                random_state=SEED,
            ),
            selected_120,
        ),
        "GradientBoosting": _pipeline(
            GradientBoostingRegressor(
                n_estimators=250,
                max_depth=2,
                learning_rate=0.03,
                loss="huber",
                random_state=SEED,
            ),
            selected_120,
        ),
        "XGBoost": _pipeline(
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.025,
                max_depth=2,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=SEED,
            ),
            selected_200,
        ),
        "LightGBM": _pipeline(
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.025,
                num_leaves=7,
                max_depth=5,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                verbosity=-1,
                n_jobs=1,
                random_state=SEED,
            ),
            selected_200,
        ),
        "CatBoost": _pipeline(
            CatBoostRegressor(
                iterations=400,
                depth=4,
                learning_rate=0.025,
                l2_leaf_reg=3.0,
                loss_function="RMSE",
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
                random_seed=SEED,
            ),
            selected_200,
        ),
    }


def evaluate_predictions(
    actual_response: np.ndarray,
    predicted_response: np.ndarray,
    threshold: float = DEFAULT_RESPONSE_THRESHOLD,
) -> Dict[str, float | int]:
    actual_bad = response_to_bad(actual_response, threshold)
    predicted_bad = response_to_bad(predicted_response, threshold)
    tn, fp, fn, tp = confusion_matrix(
        actual_bad, predicted_bad, labels=[0, 1]
    ).ravel()
    return {
        "r2": float(r2_score(actual_response, predicted_response)),
        "rmse": float(np.sqrt(mean_squared_error(actual_response, predicted_response))),
        "mae": float(mean_absolute_error(actual_response, predicted_response)),
        "accuracy": float(accuracy_score(actual_bad, predicted_bad)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_bad, predicted_bad)),
        "precision_bad": float(precision_score(actual_bad, predicted_bad, zero_division=0)),
        "recall_bad": float(recall_score(actual_bad, predicted_bad, zero_division=0)),
        "f1_bad": float(f1_score(actual_bad, predicted_bad, zero_division=0)),
        # Continuous predicted response is the ranking score; no classifier is fit.
        "roc_auc_from_response": float(roc_auc_score(actual_bad, predicted_response)),
        "pr_auc_from_response": float(average_precision_score(actual_bad, predicted_response)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def benchmark(
    features: pd.DataFrame,
    response: np.ndarray,
    groups: np.ndarray,
    threshold: float = DEFAULT_RESPONSE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_bad = response_to_bad(response, threshold)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(features, actual_bad, groups))
    metrics: list[Dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    candidates = model_candidates(features.shape[1])
    for number, (model_name, estimator) in enumerate(candidates.items(), start=1):
        predicted_response = np.full(len(response), np.nan, dtype=float)
        fold = np.full(len(response), -1, dtype=int)
        for fold_number, (train_index, test_index) in enumerate(splits, start=1):
            fitted = clone(estimator).fit(
                features.iloc[train_index], response[train_index]
            )
            predicted_response[test_index] = fitted.predict(features.iloc[test_index])
            fold[test_index] = fold_number
        result = {
            "model": model_name,
            "feature_set": "engineered_key_numbers",
            "features": int(features.shape[1]),
            "wafers": int(len(response)),
            "lots": int(pd.Series(groups).nunique()),
            "response_threshold": float(threshold),
            **evaluate_predictions(response, predicted_response, threshold),
        }
        metrics.append(result)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "fold": fold,
                    "lot": groups,
                    "actual_response": response,
                    "predicted_response": predicted_response,
                    "actual_class": np.where(actual_bad == 1, "bad", "good"),
                    "predicted_class": np.where(
                        response_to_bad(predicted_response, threshold) == 1,
                        "bad",
                        "good",
                    ),
                }
            )
        )
        print(f"[{number}/{len(candidates)}] {json.dumps(result)}", flush=True)
    metric_frame = pd.DataFrame(metrics).sort_values("r2", ascending=False)
    return metric_frame, pd.concat(prediction_frames, ignore_index=True)


def run(
    data_root: Path,
    output: Path,
    threshold: float = DEFAULT_RESPONSE_THRESHOLD,
) -> pd.DataFrame:
    feature_sets, dataset = build_feature_sets(data_root)
    features = feature_sets["engineered_key_numbers"]
    response = dataset.response.to_numpy(dtype=float)
    groups = dataset.groups.to_numpy(dtype=str)
    metrics, predictions = benchmark(features, response, groups, threshold)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "oof_response_then_class.csv", index=False)
    summary = {
        "flow": "equipment sensors -> predicted response -> predicted good/bad",
        "class_rule": f"bad when predicted_response > {threshold}",
        "validation": "5-fold StratifiedGroupKFold; complete lots held out",
        "regression_policy": (
            "Direct regression only. True class and class probability are never "
            "inputs to a response model."
        ),
        "wafers": int(len(response)),
        "lots": int(pd.Series(groups).nunique()),
        "good": int((response_to_bad(response, threshold) == 0).sum()),
        "bad": int(response_to_bad(response, threshold).sum()),
        "best_r2": metrics.iloc[0].to_dict(),
        "best_f1_bad": metrics.sort_values("f1_bad", ascending=False).iloc[0].to_dict(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("output/direct_response_then_classify")
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_RESPONSE_THRESHOLD
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = run(args.data_root, args.output, args.threshold)
    print("\nDIRECT RESPONSE THEN CLASS RESULTS\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
