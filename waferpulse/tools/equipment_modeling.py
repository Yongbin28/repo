"""Lot-grouped model validation and evidence-publishing tools."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from waferpulse.core.data_lanes import REAL_OPEN_SOURCE
from waferpulse.core.equipment_contracts import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_SELECTED_FEATURES,
    PIPELINE_VERSION,
    EquipmentDataset,
    EquipmentModelResult,
)
from waferpulse.tools.equipment_data import _json_value, _log


class QualityHurdleRegressor(BaseEstimator, RegressorMixin):
    """Predict continuous response through calibrated wafer-quality risk.

    The public target is strongly bimodal. This estimator first predicts the
    probability of crossing the documented bad-wafer threshold, then returns
    the probability-weighted mean response of the two classes. All response
    means and preprocessing parameters are learned from the active training
    fold only.
    """

    def __init__(
        self,
        response_threshold: float = 0.75,
        c: float = 0.3,
        gamma: float = 0.002,
        probability_scale: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.response_threshold = response_threshold
        self.c = c
        self.gamma = gamma
        self.probability_scale = probability_scale
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QualityHurdleRegressor":
        target = np.asarray(y, dtype=float)
        quality = (target > self.response_threshold).astype(int)
        if np.unique(quality).size != 2:
            raise ValueError("QualityHurdleRegressor requires both good and bad training wafers")

        self.good_response_mean_ = float(target[quality == 0].mean())
        self.bad_response_mean_ = float(target[quality == 1].mean())
        self.feature_columns_ = np.asarray(
            getattr(X, "columns", np.arange(np.shape(X)[1])), dtype=object
        )
        self.risk_model_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("variance", VarianceThreshold()),
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=self.c,
                        gamma=self.gamma,
                        kernel="rbf",
                        probability=True,
                        class_weight=None,
                        random_state=self.random_state,
                    ),
                ),
            ]
        ).fit(X, quality)

        transformed = self.risk_model_.named_steps["imputer"].transform(X)
        transformed = self.risk_model_.named_steps["variance"].transform(transformed)
        scores, _ = f_classif(transformed, quality)
        self.filter_scores_ = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probability = self.risk_model_.predict_proba(X)[:, 1]
        # ``getattr`` keeps bundles created before probability calibration
        # loadable after this estimator gained the new parameter.
        probability_scale = float(getattr(self, "probability_scale", 1.0))
        probability = np.clip(probability * probability_scale, 0.0, 1.0)
        return self.good_response_mean_ + probability * (
            self.bad_response_mean_ - self.good_response_mean_
        )

    def feature_relevance_table(self) -> pd.DataFrame:
        imputer = self.risk_model_.named_steps["imputer"]
        variance = self.risk_model_.named_steps["variance"]
        names = imputer.get_feature_names_out(self.feature_columns_)
        names = np.asarray(names, dtype=object)[variance.get_support()]
        scores = np.asarray(self.filter_scores_, dtype=float)
        total = float(scores.sum())
        importance = scores / total if total > 0 else np.zeros_like(scores)
        return pd.DataFrame(
            {
                "task": "regression",
                "feature": names,
                "importance": importance,
                "importance_method": "quality_class_filter_score",
            }
        ).sort_values("importance", ascending=False)


class QualityHurdleForestBlendRegressor(BaseEstimator, RegressorMixin):
    """Blend quality-risk structure with a shallow nonlinear regressor.

    The weights and component parameters were retained only after repeated
    lot-group validation. Both components are always fitted inside the active
    training fold, so no validation-lot response is used during fitting.
    """

    def __init__(
        self,
        response_threshold: float = 0.75,
        c: float = 0.4,
        gamma: float = 0.0015,
        probability_scale: float = 0.75,
        hurdle_weight: float = 0.7,
        selected_features: int = 120,
        n_estimators: int = 180,
        random_state: int = 42,
    ) -> None:
        self.response_threshold = response_threshold
        self.c = c
        self.gamma = gamma
        self.probability_scale = probability_scale
        self.hurdle_weight = hurdle_weight
        self.selected_features = selected_features
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QualityHurdleForestBlendRegressor":
        self.feature_columns_ = np.asarray(
            getattr(X, "columns", np.arange(np.shape(X)[1])), dtype=object
        )
        self.hurdle_model_ = QualityHurdleRegressor(
            response_threshold=self.response_threshold,
            c=self.c,
            gamma=self.gamma,
            probability_scale=self.probability_scale,
            random_state=self.random_state,
        ).fit(X, y)
        self.forest_model_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("variance", VarianceThreshold()),
                ("selector", SelectKBest(score_func=f_regression, k=self.selected_features)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=8,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=self.random_state,
                    ),
                ),
            ]
        ).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        hurdle_prediction = self.hurdle_model_.predict(X)
        forest_prediction = self.forest_model_.predict(X)
        return (
            self.hurdle_weight * hurdle_prediction + (1.0 - self.hurdle_weight) * forest_prediction
        )

    def feature_relevance_table(self) -> pd.DataFrame:
        hurdle = self.hurdle_model_.feature_relevance_table().copy()
        hurdle["importance"] *= self.hurdle_weight

        imputer = self.forest_model_.named_steps["imputer"]
        variance = self.forest_model_.named_steps["variance"]
        selector = self.forest_model_.named_steps["selector"]
        names = imputer.get_feature_names_out(self.feature_columns_)
        names = np.asarray(names, dtype=object)[variance.get_support()]
        names = names[selector.get_support()]
        forest = pd.DataFrame(
            {
                "task": "regression",
                "feature": names,
                "importance": (
                    (1.0 - self.hurdle_weight)
                    * self.forest_model_.named_steps["model"].feature_importances_
                ),
            }
        )
        combined = pd.concat([hurdle[["task", "feature", "importance"]], forest], ignore_index=True)
        combined = combined.groupby(["task", "feature"], as_index=False)["importance"].sum()
        combined["importance_method"] = "weighted_hurdle_filter_and_tree_impurity"
        return combined.sort_values("importance", ascending=False)


def _make_regression_candidates(
    selected_features: int,
    random_state: int,
    n_estimators: int,
) -> Dict[str, Any]:
    pre = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
        ("selector", SelectKBest(score_func=f_regression, k=selected_features)),
    ]
    return {
        "QualityHurdleForestBlend": QualityHurdleForestBlendRegressor(
            response_threshold=0.75,
            c=0.4,
            gamma=0.0015,
            probability_scale=0.75,
            hurdle_weight=0.7,
            selected_features=selected_features,
            n_estimators=n_estimators,
            random_state=random_state,
        ),
        "QualityHurdleSVM": QualityHurdleRegressor(
            response_threshold=0.75,
            c=0.3,
            gamma=0.002,
            random_state=random_state,
        ),
        "ExtraTrees": Pipeline(
            pre
            + [
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=n_estimators,
                        max_depth=12,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        # A single worker is deliberate: the desktop/PyInstaller runtime
                        # can deny creation of joblib worker handles on Windows.
                        n_jobs=1,
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "RandomForest": Pipeline(
            pre
            + [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=12,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }


def _make_classification_candidates(
    selected_features: int,
    random_state: int,
    n_estimators: int,
) -> Dict[str, Pipeline]:
    pre = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
        ("selector", SelectKBest(score_func=f_classif, k=selected_features)),
    ]
    return {
        "ExtraTrees": Pipeline(
            pre
            + [
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=12,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "RandomForest": Pipeline(
            pre
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=12,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }


def _classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    decision_threshold: float = 0.5,
) -> Dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    predicted = (probability >= decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "recall_bad": float(recall_score(actual, predicted, zero_division=0)),
        "precision_bad": float(precision_score(actual, predicted, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "brier_score": float(brier_score_loss(actual, probability)),
        "decision_threshold": float(decision_threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _choose_recall_guardrail_threshold(
    actual: np.ndarray,
    probability: np.ndarray,
    minimum_recall: float = 0.80,
) -> float:
    """Choose the most precise threshold that still meets a bad-wafer recall guardrail."""

    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.asarray(probability, dtype=float),
            ]
        )
    )
    eligible: List[Tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = (probability >= threshold).astype(int)
        recall = recall_score(actual, predicted, zero_division=0)
        precision = precision_score(actual, predicted, zero_division=0)
        if recall >= minimum_recall:
            eligible.append((float(precision), float(threshold), float(recall)))
    if not eligible:
        return 0.5
    # Precision is primary; the higher threshold wins ties and reduces false holds.
    eligible.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return eligible[0][1]


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _selected_feature_names(pipe: Pipeline, input_columns: Sequence[str]) -> np.ndarray:
    imputer = pipe.named_steps["imputer"]
    variance = pipe.named_steps["variance"]
    selector = pipe.named_steps["selector"]
    names = imputer.get_feature_names_out(np.asarray(input_columns, dtype=object))
    names = np.asarray(names, dtype=object)[variance.get_support()]
    return np.asarray(names, dtype=object)[selector.get_support()]


def _importance_table(
    regression_model: Any,
    classification_model: Pipeline,
    input_columns: Sequence[str],
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for task, pipeline in (
        ("regression", regression_model),
        ("classification", classification_model),
    ):
        if hasattr(pipeline, "feature_relevance_table"):
            tables.append(pipeline.feature_relevance_table())
            continue
        names = _selected_feature_names(pipeline, input_columns)
        importances = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
        tables.append(
            pd.DataFrame(
                {
                    "task": task,
                    "feature": names,
                    "importance": importances,
                    "importance_method": "tree_impurity",
                }
            ).sort_values("importance", ascending=False)
        )
    return pd.concat(tables, ignore_index=True)


def _target_structure_diagnostics(dataset: EquipmentDataset) -> Dict[str, Any]:
    y = dataset.response.to_numpy(dtype=float)
    bad = dataset.bad_label.to_numpy(dtype=int)
    groups = dataset.groups.to_numpy(dtype=str)
    class_mean = np.where(bad == 1, y[bad == 1].mean(), y[bad == 0].mean())
    lot_mean_lookup = pd.Series(y).groupby(pd.Series(groups)).mean()
    lot_mean = pd.Series(groups).map(lot_mean_lookup).to_numpy(dtype=float)
    return {
        "response_min": float(y.min()),
        "response_max": float(y.max()),
        "response_std": float(np.std(y, ddof=1)),
        "class_mean_oracle_r2": float(r2_score(y, class_mean)),
        "lot_mean_oracle_r2": float(r2_score(y, lot_mean)),
        "oracle_warning": "Uses actual labels/lots and is not a deployable model score.",
        "good_response_std": float(np.std(y[bad == 0], ddof=1)),
        "bad_response_std": float(np.std(y[bad == 1], ddof=1)),
        "good_wafers": int((bad == 0).sum()),
        "bad_wafers": int((bad == 1).sum()),
    }


def _oof_regression_diagnostic(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    prediction = np.full(len(y), np.nan, dtype=float)
    for train_index, validation_index in splits:
        fitted = clone(model).fit(X.iloc[train_index], y[train_index])
        prediction[validation_index] = fitted.predict(X.iloc[validation_index])
    return _regression_metrics(y, prediction)


def _conditional_response_diagnostics(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    bad: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> Dict[str, Dict[str, float]]:
    diagnostics: Dict[str, Dict[str, float]] = {}
    for class_value, label in ((0, "good"), (1, "bad")):
        mask = bad == class_value
        subset_X = X.loc[mask].reset_index(drop=True)
        subset_y = y[mask]
        subset_groups = groups[mask]
        folds = min(n_splits, int(pd.Series(subset_groups).nunique()))
        subset_splits = list(GroupKFold(n_splits=folds).split(subset_X, subset_y, subset_groups))
        diagnostics[label] = _oof_regression_diagnostic(model, subset_X, subset_y, subset_splits)
        diagnostics[label]["samples"] = int(mask.sum())
    return diagnostics


def train_equipment_models(
    dataset: EquipmentDataset,
    output_dir: Path,
    n_splits: int = 5,
    selected_features: int = DEFAULT_SELECTED_FEATURES,
    n_estimators: int = 180,
    random_state: int = DEFAULT_RANDOM_STATE,
    log_func: Optional[Callable[[str], None]] = None,
) -> EquipmentModelResult:
    """Train champion models using stratified, lot-grouped out-of-fold validation."""

    X = dataset.features
    y_reg = dataset.response.to_numpy(dtype=float)
    y_cls = dataset.bad_label.to_numpy(dtype=int)
    groups = dataset.groups.to_numpy(dtype=str)
    unique_lots = pd.Series(groups).nunique()
    n_splits = int(min(n_splits, unique_lots))
    if n_splits < 3:
        raise ValueError("At least three unique lots are required for grouped validation")
    selected_features = int(max(1, min(selected_features, X.shape[1])))

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(cv.split(X, y_cls, groups=groups))
    for fold_number, (_, validation_index) in enumerate(splits, start=1):
        if np.unique(y_cls[validation_index]).size < 2:
            raise ValueError(
                f"Fold {fold_number} contains only one class. Reduce folds or revise the lot allocation."
            )

    regression_candidates = _make_regression_candidates(
        selected_features, random_state, n_estimators
    )
    classification_candidates = _make_classification_candidates(
        selected_features, random_state, n_estimators
    )

    regression_runs: Dict[str, Dict[str, Any]] = {}
    for name, base_model in regression_candidates.items():
        _log(log_func, f"Validating real-data regression candidate: {name}...")
        oof = np.full(len(X), np.nan, dtype=float)
        folds: List[Dict[str, Any]] = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            model = clone(base_model)
            model.fit(X.iloc[train_index], y_reg[train_index])
            predicted = model.predict(X.iloc[validation_index])
            oof[validation_index] = predicted
            fold_metric = _regression_metrics(y_reg[validation_index], predicted)
            folds.append({"task": "regression", "model": name, "fold": fold, **fold_metric})
        regression_runs[name] = {
            "oof": oof,
            "metrics": _regression_metrics(y_reg, oof),
            "folds": folds,
            "template": base_model,
        }

    classification_runs: Dict[str, Dict[str, Any]] = {}
    for name, base_model in classification_candidates.items():
        _log(log_func, f"Validating real-data classification candidate: {name}...")
        oof = np.full(len(X), np.nan, dtype=float)
        folds: List[Dict[str, Any]] = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            model = clone(base_model)
            model.fit(X.iloc[train_index], y_cls[train_index])
            probability = model.predict_proba(X.iloc[validation_index])[:, 1]
            oof[validation_index] = probability
            fold_metric = _classification_metrics(y_cls[validation_index], probability)
            folds.append({"task": "classification", "model": name, "fold": fold, **fold_metric})
        classification_runs[name] = {
            "oof": oof,
            "metrics": _classification_metrics(y_cls, oof),
            "folds": folds,
            "template": base_model,
        }

    regression_name = min(
        regression_runs,
        key=lambda name: regression_runs[name]["metrics"]["rmse"],
    )
    classification_name = max(
        classification_runs,
        key=lambda name: classification_runs[name]["metrics"]["pr_auc"],
    )
    selected_regression = regression_runs[regression_name]
    selected_classification = classification_runs[classification_name]
    decision_threshold = _choose_recall_guardrail_threshold(
        y_cls,
        selected_classification["oof"],
        minimum_recall=0.80,
    )
    selected_classification_metrics = _classification_metrics(
        y_cls,
        selected_classification["oof"],
        decision_threshold=decision_threshold,
    )
    selected_classification_folds: List[Dict[str, Any]] = []
    for fold, (_, validation_index) in enumerate(splits, start=1):
        selected_classification_folds.append(
            {
                "task": "classification",
                "model": classification_name,
                "fold": fold,
                **_classification_metrics(
                    y_cls[validation_index],
                    selected_classification["oof"][validation_index],
                    decision_threshold=decision_threshold,
                ),
            }
        )

    _log(log_func, "Running leakage-audit and within-class response diagnostics...")
    random_splits = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X, y_reg)
    )
    diagnostics = {
        "target_structure": _target_structure_diagnostics(dataset),
        "random_wafer_split": {
            **_oof_regression_diagnostic(selected_regression["template"], X, y_reg, random_splits),
            "warning": (
                "Diagnostic only: wafers from the same lot can occur in training and validation."
            ),
        },
        "within_class_lot_grouped": _conditional_response_diagnostics(
            regression_runs["RandomForest"]["template"],
            X,
            y_reg,
            y_cls,
            groups,
            n_splits,
        ),
    }

    _log(log_func, f"Regression champion: {regression_name}; fitting on all real wafers...")
    regression_model = clone(selected_regression["template"]).fit(X, y_reg)
    _log(log_func, f"Classification champion: {classification_name}; fitting on all real wafers...")
    classification_model = clone(selected_classification["template"]).fit(X, y_cls)

    fold_assignment = np.zeros(len(X), dtype=int)
    for fold, (_, validation_index) in enumerate(splits, start=1):
        fold_assignment[validation_index] = fold

    probability = selected_classification["oof"]
    predictions = dataset.identity.copy()
    predictions["fold"] = fold_assignment
    predictions["response_actual"] = y_reg
    predictions["response_predicted_oof"] = selected_regression["oof"]
    predictions["class_actual"] = np.where(y_cls == 1, "bad", "good")
    predictions["bad_probability_oof"] = probability
    predictions["class_predicted_oof"] = np.where(probability >= decision_threshold, "bad", "good")
    predictions["data_lane"] = REAL_OPEN_SOURCE
    predictions["dataset_version"] = dataset.provenance["dataset_version"]

    comparison = {
        "regression": {name: run["metrics"] for name, run in regression_runs.items()},
        "classification": {name: run["metrics"] for name, run in classification_runs.items()},
    }
    metrics = {
        "validation": "StratifiedGroupKFold by manufacturing lot",
        "n_splits": n_splits,
        "selected_features": selected_features,
        "regression_champion": regression_name,
        "classification_champion": classification_name,
        "regression": selected_regression["metrics"],
        "classification": selected_classification_metrics,
        "candidate_comparison": comparison,
        "diagnostics": diagnostics,
    }
    fold_metrics = pd.DataFrame(selected_regression["folds"] + selected_classification_folds)
    importance = _importance_table(regression_model, classification_model, X.columns)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_bundle": output_dir / "equipment_quality_models.joblib",
        "metrics": output_dir / "metrics.json",
        "fold_metrics": output_dir / "fold_metrics.csv",
        "predictions": output_dir / "prediction_ledger_oof.csv",
        "feature_importance": output_dir / "feature_importance.csv",
        "sensor_quality": output_dir / "sensor_quality.csv",
        "provenance": output_dir / "provenance.json",
        "quality_summary": output_dir / "quality_summary.json",
    }
    bundle = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(X.columns),
        "response_threshold": dataset.quality_summary["response_threshold"],
        "bad_probability_threshold": decision_threshold,
        "regression_champion": regression_name,
        "classification_champion": classification_name,
        "regression_model": regression_model,
        "classification_model": classification_model,
        "metrics": metrics,
        "quality_summary": dataset.quality_summary,
        "provenance": dataset.provenance,
    }
    joblib.dump(bundle, paths["model_bundle"])
    paths["metrics"].write_text(
        json.dumps(_json_value(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    importance.to_csv(paths["feature_importance"], index=False)
    dataset.sensor_quality.to_csv(paths["sensor_quality"], index=False)
    paths["provenance"].write_text(
        json.dumps(_json_value(dataset.provenance), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["quality_summary"].write_text(
        json.dumps(_json_value(dataset.quality_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    artifact_paths = {key: str(path.resolve()) for key, path in paths.items()}
    _generate_shap_beeswarm(classification_model, X, bundle["feature_columns"], output_dir)
    _log(log_func, f"Saved real-data model evidence to {output_dir.resolve()}")
    return EquipmentModelResult(
        metrics=metrics,
        fold_metrics=fold_metrics,
        predictions=predictions,
        feature_importance=importance,
        quality_summary=dataset.quality_summary,
        sensor_quality=dataset.sensor_quality,
        provenance=dataset.provenance,
        artifact_paths=artifact_paths,
    )


def _generate_shap_beeswarm(
    classification_model: Any,
    X: pd.DataFrame,
    feature_columns: Sequence[str],
    output_dir: Path,
) -> None:
    try:
        import shap
        import matplotlib.pyplot as plt

        clf = classification_model
        transform_steps = [
            ("imputer", clf.named_steps["imputer"]),
            ("variance", clf.named_steps["variance"]),
            ("selector", clf.named_steps["selector"]),
        ]
        X_curr = X.copy()
        for _, step in transform_steps:
            X_curr = step.transform(X_curr)

        mask = clf.named_steps["selector"].get_support()
        var_mask = clf.named_steps["variance"].get_support()
        raw_cols = np.array(feature_columns)[var_mask][mask]

        def clean_name(name: str) -> str:
            parts = name.split("__")
            if len(parts) == 3:
                eq = "Eq1" if "1" in parts[0] else "Eq2"
                s = parts[1].replace("sensor_", "S")
                stat = parts[2]
                return f"{eq}·{s} ({stat})"
            return name

        cleaned_cols = [clean_name(c) for c in raw_cols]
        model = clf.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_curr)

        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            vals = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            vals = shap_vals[:, :, 1]
        else:
            vals = shap_vals

        plt.rcParams["font.sans-serif"] = "Segoe UI", "DejaVu Sans", "Arial"
        plt.rcParams["font.size"] = 10

        fig = plt.figure(figsize=(14, 6), dpi=200)
        shap.summary_plot(
            vals,
            X_curr,
            feature_names=cleaned_cols,
            max_display=10,
            plot_size=None,
            show=False,
            color_bar=True,
            alpha=0.8,
        )
        ax = plt.gca()
        ax.tick_params(axis="y", labelsize=10, pad=8)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=10.5, labelpad=8)
        plt.title("SHAP Beeswarm Summary Plot — Top 10 Feature Drivers", fontsize=12, fontweight="bold", pad=15)
        plt.subplots_adjust(left=0.18, right=0.92, top=0.92, bottom=0.12)
        fig.savefig(output_dir / "shap_beeswarm_summary.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "shap_beeswarm_summary.svg", format="svg", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass
