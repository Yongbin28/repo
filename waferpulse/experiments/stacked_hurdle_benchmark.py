"""Leakage-safe nested benchmark for multi-classifier probability stacking.

Outer folds estimate unseen-lot response performance. Inside every outer
training fold, base-classifier scores are cross-fitted by lot before sigmoid
calibration and response meta-regression. Consequently, neither calibrators
nor the meta-regressor see in-sample classifier scores.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from waferpulse.tools.equipment_data import load_equipment_dataset
from waferpulse.tools.equipment_modeling import QualityHurdleForestBlendRegressor


SEED = 42


def _preprocessed_classifier(model: Any, *, k: int | str = "all", scale: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
    ]
    if k != "all":
        steps.append(("selector", SelectKBest(f_classif, k=k)))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def base_classifiers(random_state: int) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "svm": _preprocessed_classifier(
            SVC(C=0.3, gamma=0.002, kernel="rbf", probability=False, random_state=random_state),
            scale=True,
        ),
        "random_forest": _preprocessed_classifier(
            RandomForestClassifier(
                n_estimators=180,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=random_state,
            ),
            k=120,
        ),
        "extra_trees": _preprocessed_classifier(
            ExtraTreesClassifier(
                n_estimators=180,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=random_state,
            ),
            k=120,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = _preprocessed_classifier(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.025,
                max_depth=2,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=1,
                random_state=random_state,
            ),
            k=200,
        )
    except ImportError:
        pass
    return models


def meta_regressors(random_state: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for alpha in (0.01, 0.1, 1.0, 10.0):
        models[f"ridge_alpha_{alpha}"] = Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]
        )
    models["huber_epsilon_1.35"] = Pipeline(
        [("scale", StandardScaler()), ("model", HuberRegressor(epsilon=1.35, alpha=0.1))]
    )
    models["shallow_random_forest"] = RandomForestRegressor(
        n_estimators=220,
        max_depth=3,
        min_samples_leaf=20,
        max_features=0.75,
        n_jobs=1,
        random_state=random_state,
    )
    models["gradient_boosting_stumps"] = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.025,
        max_depth=1,
        min_samples_leaf=20,
        loss="huber",
        random_state=random_state,
    )
    return models


def _classifier_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    probability = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))


def _meta_features(probabilities: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            probabilities,
            probabilities.mean(axis=1),
            probabilities.std(axis=1),
            probabilities.min(axis=1),
            probabilities.max(axis=1),
        ]
    )


def _cross_fitted_probability_features(
    X_train: pd.DataFrame,
    class_train: np.ndarray,
    group_train: np.ndarray,
    X_validation: pd.DataFrame,
    *,
    random_state: int,
    inner_splits: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    base_models = base_classifiers(random_state)
    names = tuple(base_models)
    train_probability = np.full((len(X_train), len(names)), np.nan, dtype=float)
    validation_probability = np.full((len(X_validation), len(names)), np.nan, dtype=float)
    folds = list(
        StratifiedGroupKFold(
            n_splits=min(inner_splits, pd.Series(group_train).nunique()),
            shuffle=True,
            random_state=random_state,
        ).split(X_train, class_train, group_train)
    )

    for column, name in enumerate(names):
        template = base_models[name]
        cross_fitted_score = np.full(len(X_train), np.nan, dtype=float)
        for inner_train, inner_validation in folds:
            fitted = clone(template).fit(X_train.iloc[inner_train], class_train[inner_train])
            cross_fitted_score[inner_validation] = _classifier_score(
                fitted, X_train.iloc[inner_validation]
            )
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        calibrator.fit(cross_fitted_score.reshape(-1, 1), class_train)
        train_probability[:, column] = calibrator.predict_proba(
            cross_fitted_score.reshape(-1, 1)
        )[:, 1]

        fitted_full = clone(template).fit(X_train, class_train)
        validation_score = _classifier_score(fitted_full, X_validation)
        validation_probability[:, column] = calibrator.predict_proba(
            validation_score.reshape(-1, 1)
        )[:, 1]

    if not np.isfinite(train_probability).all() or not np.isfinite(validation_probability).all():
        raise ValueError("Stacked probability features contain non-finite values")
    return _meta_features(train_probability), _meta_features(validation_probability), names


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def run_seed(
    X: pd.DataFrame,
    response: np.ndarray,
    quality: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
) -> pd.DataFrame:
    outer_folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state).split(
            X, quality, groups
        )
    )
    meta_models = meta_regressors(random_state)
    predictions = {name: np.full(len(X), np.nan) for name in meta_models}
    probability_sources = (
        "logistic_c_0.1",
        "logistic_c_1.0",
        "logistic_c_10.0",
        "soft_vote_mean",
        *(f"{name}_only" for name in base_classifiers(random_state)),
    )
    hurdle_predictions = {
        f"{source}_scale_{scale}_hurdle_weight_{weight}": np.full(len(X), np.nan)
        for source in probability_sources
        for scale in (0.5, 0.75, 1.0, 1.25)
        for weight in (0.6, 0.7, 0.8, 1.0)
    }
    baseline_prediction = np.full(len(X), np.nan)
    base_names: Tuple[str, ...] = ()
    started = time.perf_counter()

    for fold_number, (train, validation) in enumerate(outer_folds, start=1):
        train_meta, validation_meta, base_names = _cross_fitted_probability_features(
            X.iloc[train],
            quality[train],
            groups[train],
            X.iloc[validation],
            random_state=random_state + fold_number,
        )
        for name, template in meta_models.items():
            fitted = clone(template).fit(train_meta, response[train])
            predictions[name][validation] = fitted.predict(validation_meta)
        baseline = QualityHurdleForestBlendRegressor(random_state=random_state)
        baseline.fit(X.iloc[train], response[train])
        baseline_prediction[validation] = baseline.predict(X.iloc[validation])
        forest_prediction = baseline.forest_model_.predict(X.iloc[validation])

        base_count = len(base_names)
        validation_probabilities = validation_meta[:, :base_count]
        stacked_probabilities: Dict[str, np.ndarray] = {}
        for c in (0.1, 1.0, 10.0):
            classifier = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
            classifier.fit(train_meta, quality[train])
            stacked_probabilities[f"logistic_c_{c}"] = classifier.predict_proba(
                validation_meta
            )[:, 1]
        stacked_probabilities["soft_vote_mean"] = validation_probabilities.mean(axis=1)
        for column, name in enumerate(base_names):
            stacked_probabilities[f"{name}_only"] = validation_probabilities[:, column]

        good_mean = float(response[train][quality[train] == 0].mean())
        bad_mean = float(response[train][quality[train] == 1].mean())
        for source, probability in stacked_probabilities.items():
            for scale in (0.5, 0.75, 1.0, 1.25):
                hurdle = good_mean + np.clip(probability * scale, 0.0, 1.0) * (
                    bad_mean - good_mean
                )
                for weight in (0.6, 0.7, 0.8, 1.0):
                    name = f"{source}_scale_{scale}_hurdle_weight_{weight}"
                    hurdle_predictions[name][validation] = (
                        weight * hurdle + (1.0 - weight) * forest_prediction
                    )
        print(
            json.dumps(
                {
                    "seed": random_state,
                    "outer_fold": fold_number,
                    "base_classifiers": base_names,
                }
            ),
            flush=True,
        )

    rows = [
        {
            "seed": random_state,
            "experiment": "production_hurdle_forest_blend",
            "base_classifiers": "production baseline",
            **_metrics(response, baseline_prediction),
            "seconds": time.perf_counter() - started,
        }
    ]
    for name, prediction in predictions.items():
        rows.append(
            {
                "seed": random_state,
                "experiment": f"stacked_sigmoid_{name}",
                "base_classifiers": "+".join(base_names),
                **_metrics(response, prediction),
                "seconds": time.perf_counter() - started,
            }
        )
    for name, prediction in hurdle_predictions.items():
        rows.append(
            {
                "seed": random_state,
                "experiment": f"stacked_sigmoid_hurdle_{name}",
                "base_classifiers": "+".join(base_names),
                **_metrics(response, prediction),
                "seconds": time.perf_counter() - started,
            }
        )
    return pd.DataFrame(rows)


def main(seeds: Iterable[int] = (SEED,)) -> None:
    root = Path(__file__).resolve().parents[2]
    dataset = load_equipment_dataset(root / "EquipmentData", stage_mode="both")
    X = dataset.features.reset_index(drop=True)
    response = dataset.response.to_numpy(float)
    quality = dataset.bad_label.to_numpy(int)
    groups = dataset.groups.to_numpy(str)
    results = pd.concat(
        [
            run_seed(
                X,
                response,
                quality,
                groups,
                random_state=int(seed),
            )
            for seed in seeds
        ],
        ignore_index=True,
    ).sort_values(["seed", "r2"], ascending=[True, False])
    output = root / "output" / "equipment_quality" / "r2_stacked_hurdle_benchmark.csv"
    results.to_csv(output, index=False)
    baseline = results.loc[results["experiment"].eq("production_hurdle_forest_blend")]
    best_stack = (
        results.loc[~results["experiment"].eq("production_hurdle_forest_blend")]
        .sort_values("r2", ascending=False)
        .groupby("seed", as_index=False)
        .head(1)
    )
    summary = pd.concat([baseline, best_stack], ignore_index=True).sort_values(
        ["seed", "r2"], ascending=[True, False]
    )
    summary.to_csv(
        root / "output" / "equipment_quality" / "r2_stacked_hurdle_summary.csv",
        index=False,
    )
    print("\nSTACKED HURDLE SUMMARY\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
