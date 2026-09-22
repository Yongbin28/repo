"""Diagnose within-lot bad-wafer ranking from grouped OOF probabilities.

This module is intentionally diagnostic-only.  For every manufacturing lot it
uses the *true* number of bad wafers and selects that many highest-probability
wafers.  The resulting score is therefore an optimistic ceiling for a proposed
lot-count calibration layer, not a deployable or reportable production model.

All probability inputs must already be out-of-fold predictions generated with
manufacturing lots held out.  Response values and class labels are used only
for evaluation and the explicitly labelled oracle operation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


@dataclass(frozen=True)
class ProbabilitySource:
    name: str
    path: Path
    column: str


def normalize_lot(values: pd.Series) -> pd.Series:
    """Return a stable lower-case lot key."""

    return values.astype(str).str.strip().str.lower()


def normalize_wafer(values: pd.Series) -> pd.Series:
    """Normalize numeric wafer identifiers without changing non-numeric IDs."""

    text = values.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    whole = numeric.notna() & np.isclose(numeric, np.round(numeric))
    result = text.copy()
    result.loc[whole] = np.round(numeric.loc[whole]).astype(np.int64).astype(str)
    return result


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["lot"] = normalize_lot(result["lot"])
    result["wafer"] = normalize_wafer(result["wafer"])
    return result


def oracle_count_selection(
    lots: Sequence[object], actual_bad: Sequence[int], probability: Sequence[float]
) -> np.ndarray:
    """Select the true bad count's top-ranked wafers independently per lot."""

    frame = pd.DataFrame(
        {
            "lot": np.asarray(lots),
            "actual_bad": np.asarray(actual_bad, dtype=np.int8),
            "probability": np.asarray(probability, dtype=np.float64),
            "row": np.arange(len(lots)),
        }
    )
    selected = np.zeros(len(frame), dtype=np.int8)
    for _, group in frame.groupby("lot", sort=False):
        bad_count = int(group["actual_bad"].sum())
        if bad_count == 0:
            continue
        # Stable row order makes exact probability ties reproducible.
        ranked = group.sort_values(
            ["probability", "row"], ascending=[False, True], kind="mergesort"
        )
        selected[ranked.head(bad_count)["row"].to_numpy(dtype=np.int64)] = 1
    return selected


def cross_fitted_class_mean_prediction(
    response: Sequence[float], actual_bad: Sequence[int], fold: Sequence[int],
    predicted_bad: Sequence[int]
) -> np.ndarray:
    """Map predicted class to class means estimated outside each held-out fold."""

    y = np.asarray(response, dtype=np.float64)
    classes = np.asarray(actual_bad, dtype=np.int8)
    folds = np.asarray(fold)
    predictions = np.asarray(predicted_bad, dtype=np.int8)
    result = np.empty(len(y), dtype=np.float64)
    for held_out in np.unique(folds):
        train = folds != held_out
        test = ~train
        means = {}
        for label in (0, 1):
            values = y[train & (classes == label)]
            if not len(values):
                raise ValueError(f"Fold {held_out!r} has no training examples for class {label}")
            means[label] = float(values.mean())
        result[test] = np.where(predictions[test] == 1, means[1], means[0])
    return result


def evaluate_probability_source(reference: pd.DataFrame, probability: np.ndarray) -> dict:
    actual = reference["actual_bad"].to_numpy(dtype=np.int8)
    selected = oracle_count_selection(reference["lot"], actual, probability)
    class_mean_prediction = cross_fitted_class_mean_prediction(
        reference["response"], actual, reference["fold"], selected
    )
    true_positive = int(np.sum((selected == 1) & (actual == 1)))
    selected_count = int(selected.sum())
    bad_count = int(actual.sum())
    return {
        "rows": int(len(reference)),
        "lots": int(reference["lot"].nunique()),
        "bad_wafers": bad_count,
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "oracle_count_true_positives": true_positive,
        "oracle_count_class_mistakes": int(np.sum(selected != actual)),
        "oracle_count_precision": float(true_positive / selected_count),
        "oracle_count_recall": float(true_positive / bad_count),
        "oracle_count_class_mean_r2": float(
            r2_score(reference["response"], class_mean_prediction)
        ),
    }


def within_lot_rank_average(
    reference: pd.DataFrame, probabilities: Sequence[np.ndarray]
) -> np.ndarray:
    """Average scale-free percentile ranks without using labels or responses."""

    if not probabilities:
        raise ValueError("At least one probability vector is required")
    ranked = []
    for probability in probabilities:
        if len(probability) != len(reference):
            raise ValueError("Probability vectors must match the reference length")
        frame = pd.DataFrame({"lot": reference["lot"], "probability": probability})
        ranked.append(
            frame.groupby("lot", sort=False)["probability"]
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float64)
        )
    return np.mean(np.column_stack(ranked), axis=1)


def load_reference(reference_path: Path, ledger_path: Path) -> pd.DataFrame:
    raw = normalize_keys(pd.read_csv(reference_path))
    ledger = normalize_keys(pd.read_csv(ledger_path))
    reference = raw[["lot", "wafer", "response", "actual_bad"]].copy()
    folds = ledger[["lot", "wafer", "fold"]].drop_duplicates(["lot", "wafer"])
    reference = reference.merge(folds, on=["lot", "wafer"], how="left", validate="1:1")
    if reference["fold"].isna().any():
        missing = reference.loc[reference["fold"].isna(), ["lot", "wafer"]].head()
        raise ValueError(f"Missing fold assignments for:\n{missing.to_string(index=False)}")
    if reference.duplicated(["lot", "wafer"]).any():
        raise ValueError("Reference contains duplicate normalized lot/wafer keys")
    return reference.sort_values(["lot", "wafer"], kind="mergesort").reset_index(drop=True)


def align_probability(reference: pd.DataFrame, source: ProbabilitySource) -> np.ndarray:
    frame = normalize_keys(pd.read_csv(source.path, usecols=["lot", "wafer", source.column]))
    if frame.duplicated(["lot", "wafer"]).any():
        raise ValueError(f"{source.name}: duplicate normalized lot/wafer keys")
    aligned = reference[["lot", "wafer"]].merge(
        frame, on=["lot", "wafer"], how="left", validate="1:1"
    )
    probability = aligned[source.column].to_numpy(dtype=np.float64)
    if np.isinf(probability).any():
        raise ValueError(f"{source.name}: probabilities contain infinite values")
    if not np.isfinite(probability).any():
        raise ValueError(f"{source.name}: no probabilities align to the reference")
    return probability


def default_sources(output_dir: Path) -> Iterable[ProbabilitySource]:
    cdae = output_dir / "cdae_trace_oof_predictions_paper56_s3_tuned.csv"
    ledger = output_dir / "prediction_ledger_oof.csv"
    temporal = output_dir / "supervised_temporal_hurdle_oof.csv"
    signal = output_dir / "signal_feature_oof_predictions.csv"
    key_numbers = output_dir / "key_number_oof_predictions.csv"
    return (
        ProbabilitySource("production", ledger, "bad_probability_oof"),
        ProbabilitySource("cdae_tuned", cdae, "cdae_bad_probability"),
        ProbabilitySource("cdae_lot_context", cdae, "lot_context_bad_probability"),
        ProbabilitySource("supervised_temporal", temporal, "bad_probability"),
        ProbabilitySource(
            "signal_random_forest", signal, "prediction__probability__random_forest"
        ),
        ProbabilitySource(
            "signal_extra_trees", signal, "prediction__probability__extra_trees"
        ),
        ProbabilitySource(
            "expert_key_random_forest",
            key_numbers,
            "prediction__official_key_numbers__probability__random_forest",
        ),
        ProbabilitySource(
            "expert_plus_generic_random_forest",
            key_numbers,
            "prediction__key_numbers_plus_generic_temporal__probability__random_forest",
        ),
    )


def run(output_dir: Path) -> pd.DataFrame:
    reference_path = output_dir / "cdae_trace_oof_predictions_paper56_s3_tuned.csv"
    ledger_path = output_dir / "prediction_ledger_oof.csv"
    reference = load_reference(reference_path, ledger_path)

    records = []
    full_probabilities = {}
    for source in default_sources(output_dir):
        if not source.path.exists():
            continue
        probability = align_probability(reference, source)
        available = np.isfinite(probability)
        source_reference = reference.loc[available].reset_index(drop=True)
        record = {
            "source": source.name,
            "missing_rows": int((~available).sum()),
            **evaluate_probability_source(source_reference, probability[available]),
        }
        records.append(record)
        if available.all():
            full_probabilities[source.name] = probability

    ensemble_members = {
        "rank_average_all_full_sources": tuple(full_probabilities),
        "rank_average_production_cdae_context": (
            "production",
            "cdae_lot_context",
        ),
        "rank_average_production_cdae_context_signal_rf": (
            "production",
            "cdae_lot_context",
            "signal_random_forest",
        ),
    }
    for name, members in ensemble_members.items():
        if not all(member in full_probabilities for member in members):
            continue
        probability = within_lot_rank_average(
            reference, [full_probabilities[member] for member in members]
        )
        records.append(
            {
                "source": name,
                "missing_rows": 0,
                **evaluate_probability_source(reference, probability),
            }
        )

    actual = reference["actual_bad"].to_numpy(dtype=np.int8)
    perfect_prediction = cross_fitted_class_mean_prediction(
        reference["response"], actual, reference["fold"], actual
    )
    perfect_r2 = float(r2_score(reference["response"], perfect_prediction))
    result = pd.DataFrame(records).sort_values(
        ["oracle_count_class_mean_r2", "roc_auc"], ascending=False
    )
    result.to_csv(output_dir / "lot_ranking_oracle_count_diagnostic.csv", index=False)

    metadata = {
        "diagnostic_only": True,
        "deployable": False,
        "oracle_information": "true bad-wafer count within each held-out lot",
        "class_means": "estimated outside each held-out fold",
        "perfect_true_class_cross_fitted_r2": perfect_r2,
        "best_oracle_count_r2": float(result.iloc[0]["oracle_count_class_mean_r2"]),
        "best_source": str(result.iloc[0]["source"]),
        "minimum_class_mistakes": int(result["oracle_count_class_mistakes"].min()),
    }
    (output_dir / "lot_ranking_oracle_count_diagnostic.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/equipment_quality")
    )
    args = parser.parse_args()
    result = run(args.output_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
