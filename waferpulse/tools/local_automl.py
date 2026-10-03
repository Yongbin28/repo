"""Guarded in-memory AutoML for user-uploaded numeric regression tables."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from waferpulse.experiments.bosch_plasma_etch_benchmark import model_candidates

MAX_LOCAL_AUTOML_ROWS = 10_000


def benchmark_local_regression(
    frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: Optional[str] = None,
    search_profile: str = "balanced",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare the report's nine regressors using OOF R² without saving the upload."""

    if target_column not in frame:
        raise ValueError(f"Target column not found: {target_column}")
    work = frame.head(MAX_LOCAL_AUTOML_ROWS).copy()
    target = pd.to_numeric(work[target_column], errors="coerce")
    numeric = work.drop(columns=[target_column, group_column], errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.loc[:, numeric.notna().sum().ge(max(5, int(len(work) * 0.2)))]
    valid = target.notna()
    numeric = numeric.loc[valid].reset_index(drop=True)
    target_values = target.loc[valid].to_numpy(dtype=float)
    if len(target_values) < 30:
        raise ValueError("Local AutoML requires at least 30 rows with a numeric target")
    if numeric.shape[1] < 1:
        raise ValueError("No usable numeric predictor columns remain after excluding the target")

    groups = None
    validation = "5-fold shuffled KFold (no group column selected)"
    if group_column:
        groups = work.loc[valid, group_column].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < 3:
            raise ValueError("Grouped validation requires at least three distinct groups")
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        splits = list(splitter.split(numeric, target_values, groups))
        validation = f"{len(splits)}-fold GroupKFold by {group_column}"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(numeric, target_values))

    candidates = {
        name: estimator
        for name, estimator in model_candidates(numeric.shape[1], search_profile).items()
        if name not in {"dummy_mean", "ridge_selected"}
    }
    rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    for name, estimator in candidates.items():
        prediction = np.full(len(target_values), np.nan, dtype=float)
        for train_index, validation_index in splits:
            fitted = clone(estimator).fit(
                numeric.iloc[train_index], target_values[train_index]
            )
            prediction[validation_index] = fitted.predict(numeric.iloc[validation_index])
        rows.append(
            {
                "model": name,
                "r2": float(r2_score(target_values, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(target_values, prediction))),
                "mae": float(mean_absolute_error(target_values, prediction)),
            }
        )
        ledgers.append(
            pd.DataFrame(
                {
                    "row": np.arange(len(target_values)),
                    "model": name,
                    "actual": target_values,
                    "predicted": prediction,
                }
            )
        )
    metrics = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    metadata = {
        "rows": int(len(target_values)),
        "features": int(numeric.shape[1]),
        "validation": validation,
        "search_profile": search_profile,
        "champion": str(metrics.iloc[0]["model"]),
    }
    return metrics, pd.concat(ledgers, ignore_index=True), metadata
