"""Predict BOSCH wafer-average silicon etch, then derive high-etch class.

The regression models receive upstream process summaries only.  They first
predict the continuous wafer-average silicon etch.  A configurable research
threshold is applied afterward:

    predicted_high_etch = predicted_average_si_etch > threshold

The public BOSCH dataset does not publish a production pass/fail specification,
so the default 44.0 threshold is an experiment rule, not a factory quality
limit.  Complete manufacturing lots remain isolated in five grouped folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from waferpulse.experiments.bosch_plasma_etch_benchmark import (
    _group_oof_predictions,
    load_dataset,
    model_candidates,
)
from waferpulse.experiments.direct_response_then_classify_benchmark import (
    evaluate_predictions,
    response_to_bad,
)


DEFAULT_RESEARCH_THRESHOLD = 44.0


def build_wafer_average_table(data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    dataset = load_dataset(data_root)
    wafer_target = (
        dataset.measurements.groupby(
            ["experiment_key", "lot_number"], as_index=False
        )["si_etch"]
        .mean()
        .rename(columns={"si_etch": "average_si_etch"})
    )
    table = wafer_target.merge(
        dataset.process, on="experiment_key", how="inner", validate="one_to_one"
    )
    process_columns = [
        column for column in dataset.process.columns if column != "experiment_key"
    ]
    return table, process_columns


def benchmark(
    table: pd.DataFrame,
    process_columns: list[str],
    threshold: float = DEFAULT_RESEARCH_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = table[process_columns]
    response = table["average_si_etch"].to_numpy(dtype=float)
    groups = table["lot_number"].to_numpy()
    actual_high_etch = response_to_bad(response, threshold)
    metrics: list[Dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    candidates = {
        name: estimator
        for name, estimator in model_candidates(features.shape[1]).items()
        if name != "dummy_mean"
    }
    for number, (model_name, estimator) in enumerate(candidates.items(), start=1):
        predicted_response = _group_oof_predictions(
            estimator, features, response, groups
        )
        result = {
            "model": model_name,
            "input": "upstream_process_summaries_only",
            "target": "wafer_average_si_etch",
            "wafers": int(len(response)),
            "lots": int(pd.Series(groups).nunique()),
            "research_threshold": float(threshold),
            **evaluate_predictions(response, predicted_response, threshold),
        }
        metrics.append(result)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "experiment_key": table["experiment_key"],
                    "lot_number": groups,
                    "actual_average_si_etch": response,
                    "predicted_average_si_etch": predicted_response,
                    "actual_research_class": np.where(
                        actual_high_etch == 1, "high_etch", "within_threshold"
                    ),
                    "predicted_research_class": np.where(
                        response_to_bad(predicted_response, threshold) == 1,
                        "high_etch",
                        "within_threshold",
                    ),
                }
            )
        )
        print(f"[{number}/{len(candidates)}] {json.dumps(result)}", flush=True)

    metric_frame = pd.DataFrame(metrics).sort_values("r2", ascending=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metric_frame, predictions


def run(
    data_root: Path,
    output: Path,
    threshold: float = DEFAULT_RESEARCH_THRESHOLD,
) -> pd.DataFrame:
    table, process_columns = build_wafer_average_table(data_root)
    metrics, predictions = benchmark(table, process_columns, threshold)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "oof_response_then_class.csv", index=False)
    actual_high_etch = response_to_bad(table["average_si_etch"], threshold)
    summary = {
        "flow": (
            "upstream BOSCH equipment sensors -> predicted wafer-average silicon "
            "etch -> research high-etch category"
        ),
        "class_rule": f"high_etch when predicted_average_si_etch > {threshold}",
        "threshold_status": (
            "Research threshold only; the public dataset supplies no official "
            "production pass/fail specification."
        ),
        "validation": "Five-fold manufacturing-lot-held-out cross-validation",
        "predictor_policy": (
            "Process summaries only; no X/Y, identifiers, dates, or downstream "
            "metrology inputs."
        ),
        "wafers": int(len(table)),
        "lots": int(table["lot_number"].nunique()),
        "within_threshold": int((actual_high_etch == 0).sum()),
        "high_etch": int(actual_high_etch.sum()),
        "best_r2": metrics.iloc[0].to_dict(),
        "best_f1_high_etch": (
            metrics.sort_values("f1_bad", ascending=False).iloc[0].to_dict()
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = (
        Path("dataset/bosch_plasma_etch")
        if Path("dataset/bosch_plasma_etch").is_dir()
        else Path("data/bosch_plasma_etch")
    )
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/bosch_response_then_classify"),
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_RESEARCH_THRESHOLD
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = run(args.data_root, args.output, args.threshold)
    print("\nBOSCH RESPONSE THEN CLASS RESULTS\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
