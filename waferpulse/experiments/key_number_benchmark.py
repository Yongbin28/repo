"""Strict unseen-lot benchmark for official EquipmentData key numbers.

The companion Zenodo record 10.5281/zenodo.4533818 contains 50 key numbers
derived by process experts from selected windows of the same two equipment
sensor traces.  This experiment verifies the labels against the local raw
EquipmentData response file, excludes identity and targets from predictors,
and evaluates only out-of-fold predictions from manufacturing-lot groups.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
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
from sklearn.svm import SVC, SVR

from waferpulse.tools.equipment_data import load_equipment_dataset


SEED = 42
KEY_NUMBER_DOI = "10.5281/zenodo.4533818"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="cp1252")


def load_key_number_dataset(
    raw_data_dir: Path,
    key_number_dir: Path,
) -> Tuple[Mapping[str, pd.DataFrame], pd.DataFrame, Mapping[str, Any]]:
    process1 = _read(key_number_dir / "process1.csv")
    process2 = _read(key_number_dir / "process2.csv")
    labels = _read(key_number_dir / "response.csv")
    duplicate_rows: Dict[str, int] = {}
    for name, frame in (
        ("process1.csv", process1),
        ("process2.csv", process2),
        ("response.csv", labels),
    ):
        frame["lot"] = frame["lot"].astype(str).str.strip()
        frame["wafer"] = frame["wafer"].astype(str).str.strip()
        conflict = frame.groupby(["lot", "wafer"]).nunique(dropna=False).max(axis=1)
        if (conflict > 1).any():
            raise ValueError("Official key-number files contain conflicting lot/wafer keys")
        duplicate_rows[name] = int(frame.duplicated().sum())
        frame.drop_duplicates(["lot", "wafer"], keep="first", inplace=True)

    official = process1.merge(
        process2, on=["lot", "wafer"], how="inner", validate="one_to_one"
    ).merge(labels, on=["lot", "wafer"], how="inner", validate="one_to_one")
    key_columns = [column for column in official if column.startswith("KN")]
    if len(key_columns) != 50:
        raise ValueError(f"Expected 50 official key numbers, found {len(key_columns)}")
    official[key_columns] = official[key_columns].apply(pd.to_numeric, errors="raise")
    official["response"] = pd.to_numeric(official["response"], errors="raise")
    official["class"] = official["class"].astype(str).str.strip().str.lower()

    raw = load_equipment_dataset(raw_data_dir, stage_mode="both")
    raw_frame = raw.identity.copy()
    raw_frame["response_raw"] = raw.response.to_numpy(float)
    raw_frame["class_raw"] = np.where(raw.bad_label.to_numpy(int) == 1, "bad", "good")
    raw_frame = pd.concat([raw_frame, raw.features.reset_index(drop=True)], axis=1)
    combined = official.merge(
        raw_frame, on=["lot", "wafer"], how="inner", validate="one_to_one"
    )
    if len(combined) != len(official):
        raise ValueError(
            f"Only {len(combined)} of {len(official)} official key-number wafers align "
            "to the local EquipmentData traces"
        )
    if not np.allclose(
        combined["response"], combined["response_raw"], atol=1e-12, rtol=0
    ):
        raise ValueError("Official and local continuous responses are not identical")
    if not np.array_equal(combined["class"].to_numpy(), combined["class_raw"].to_numpy()):
        raise ValueError("Official and local class labels are not identical")

    identity = combined[["lot", "wafer"]].copy()
    features = {
        "official_key_numbers": combined[key_columns].astype(float).reset_index(drop=True),
        "key_numbers_plus_generic_temporal": combined[
            [*key_columns, *raw.features.columns]
        ].astype(float).reset_index(drop=True),
    }
    target = pd.DataFrame(
        {
            "response": combined["response"].astype(float).to_numpy(),
            "bad": combined["class"].eq("bad").astype(int).to_numpy(),
        }
    )
    metadata = {
        "source": "Official EquipmentData domain-expert key-number companion dataset",
        "doi": KEY_NUMBER_DOI,
        "wafers": int(len(combined)),
        "lots": int(combined["lot"].nunique()),
        "bad_wafers": int(target["bad"].sum()),
        "key_numbers": int(len(key_columns)),
        "feature_exclusions": ["lot", "wafer", "response", "class"],
        "label_identity_check": "exact response and class equality with local EquipmentData",
        "exact_duplicate_rows_removed": duplicate_rows,
        "file_sha256": {
            name: _sha256(key_number_dir / name)
            for name in ("process1.csv", "process2.csv", "response.csv")
        },
    }
    return features, pd.concat([identity.reset_index(drop=True), target], axis=1), metadata


def _scaled(estimator: Any, *, k: int | str = "all") -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
    ]
    if k != "all":
        steps.append(("selector", SelectKBest(f_regression, k=k)))
    steps.extend([("scale", StandardScaler()), ("model", estimator)])
    return Pipeline(steps)


def _tree(estimator: Any, *, k: int | str = "all") -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold()),
    ]
    if k != "all":
        steps.append(("selector", SelectKBest(f_regression, k=k)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _regressors(feature_count: int) -> Mapping[str, Any]:
    k = min(200, feature_count)
    models: Dict[str, Any] = {
        "ridge_a100": _scaled(Ridge(alpha=100.0), k=k),
        "pls_10": _scaled(PLSRegression(n_components=min(10, k), max_iter=1000), k=k),
        "rbf_svr": _scaled(SVR(C=3.0, epsilon=0.03, gamma="scale"), k=k),
        "random_forest": _tree(
            RandomForestRegressor(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
        ),
        "extra_trees": _tree(
            ExtraTreesRegressor(
                n_estimators=500,
                max_features=0.7,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
        ),
        "gradient_boosting": _tree(
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=3,
                loss="huber",
                random_state=SEED,
            ),
            k=k,
        ),
        "hist_gradient_boosting": _tree(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=5.0,
                random_state=SEED,
            ),
            k=k,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = _tree(
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=10.0,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = _tree(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=15,
                reg_lambda=5.0,
                verbosity=-1,
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
        )
    except ImportError:
        pass
    return models


def _classifiers(feature_count: int) -> Mapping[str, Any]:
    k = min(200, feature_count)
    return {
        "logistic": _scaled(
            LogisticRegression(
                C=0.3,
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            ),
            k=k,
        ),
        "rbf_svm": _scaled(
            SVC(
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=SEED,
            ),
            k=k,
        ),
        "random_forest": _tree(
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
        ),
        "extra_trees": _tree(
            ExtraTreesClassifier(
                n_estimators=500,
                max_features=0.7,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=SEED,
            ),
            k=k,
        ),
        "hist_gradient_boosting": _tree(
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=5.0,
                random_state=SEED,
            ),
            k=k,
        ),
    }


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> Mapping[str, float]:
    return {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
    }


def run_benchmark(
    raw_data_dir: Path,
    key_number_dir: Path,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    feature_sets, ledger, metadata = load_key_number_dataset(raw_data_dir, key_number_dir)
    response = ledger["response"].to_numpy(float)
    bad = ledger["bad"].to_numpy(int)
    groups = ledger["lot"].to_numpy(str)
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            ledger, bad, groups
        )
    )
    fold_assignment = np.zeros(len(ledger), dtype=int)
    for fold_number, (_, validation) in enumerate(folds, start=1):
        fold_assignment[validation] = fold_number
    ledger["fold"] = fold_assignment

    rows: List[Dict[str, Any]] = []
    prediction_columns: Dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for feature_name, X in feature_sets.items():
        for model_name, template in _regressors(X.shape[1]).items():
            prediction = np.full(len(response), np.nan)
            model_started = time.perf_counter()
            try:
                for train, validation in folds:
                    fitted = clone(template).fit(X.iloc[train], response[train])
                    prediction[validation] = np.asarray(
                        fitted.predict(X.iloc[validation])
                    ).reshape(-1)
                row = {
                    "feature_set": feature_name,
                    "task": "direct_regression",
                    "model": model_name,
                    **_metrics(response, prediction),
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "seconds": time.perf_counter() - model_started,
                    "status": "OK",
                    "error": "",
                }
                prediction_columns[f"{feature_name}__regression__{model_name}"] = prediction
            except Exception as exc:
                row = {
                    "feature_set": feature_name,
                    "task": "direct_regression",
                    "model": model_name,
                    "r2": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "seconds": time.perf_counter() - model_started,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print(json.dumps(row), flush=True)

        for model_name, template in _classifiers(X.shape[1]).items():
            probability = np.full(len(response), np.nan)
            hurdle = np.full(len(response), np.nan)
            model_started = time.perf_counter()
            try:
                for train, validation in folds:
                    fitted = clone(template).fit(X.iloc[train], bad[train])
                    fold_probability = fitted.predict_proba(X.iloc[validation])[:, 1]
                    probability[validation] = fold_probability
                    good_mean = float(response[train][bad[train] == 0].mean())
                    bad_mean = float(response[train][bad[train] == 1].mean())
                    hurdle[validation] = good_mean + fold_probability * (bad_mean - good_mean)
                row = {
                    "feature_set": feature_name,
                    "task": "probability_hurdle",
                    "model": model_name,
                    **_metrics(response, hurdle),
                    "roc_auc": float(roc_auc_score(bad, probability)),
                    "pr_auc": float(average_precision_score(bad, probability)),
                    "seconds": time.perf_counter() - model_started,
                    "status": "OK",
                    "error": "",
                }
                prediction_columns[f"{feature_name}__hurdle__{model_name}"] = hurdle
                prediction_columns[f"{feature_name}__probability__{model_name}"] = probability
            except Exception as exc:
                row = {
                    "feature_set": feature_name,
                    "task": "probability_hurdle",
                    "model": model_name,
                    "r2": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "seconds": time.perf_counter() - model_started,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print(json.dumps(row), flush=True)

    result = pd.DataFrame(rows).sort_values("r2", ascending=False, na_position="last")
    for name, values in prediction_columns.items():
        ledger[f"prediction__{name}"] = values
    metadata = {
        **dict(metadata),
        "validation": "5-fold StratifiedGroupKFold by manufacturing lot",
        "selection_status": "exploratory candidate comparison; no production replacement",
        "best_candidate": result.iloc[0].to_dict(),
        "r2_gate": 0.8,
        "r2_gate_passed": bool(float(result.iloc[0]["r2"]) > 0.8),
        "seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "key_number_benchmark.csv", index=False)
    ledger.to_csv(output_dir / "key_number_oof_predictions.csv", index=False)
    (output_dir / "key_number_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result, metadata


def main(argv: Iterable[str] | None = None) -> None:
    del argv
    root = Path(__file__).resolve().parents[2]
    result, metadata = run_benchmark(
        root / "EquipmentData",
        root / "data" / "equipment_key_numbers",
        root / "output" / "equipment_quality",
    )
    print("\nKEY-NUMBER RESULTS\n" + result.to_string(index=False), flush=True)
    print("\n" + json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
