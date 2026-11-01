"""Compare the original HEAD workflows with the shared coordinator on real data.

Run each route separately with --dataset equipment or --dataset bosch. Outputs
are isolated from dashboard artifacts. Groq choices are fixed for reproducibility;
SHAP plotting alone is skipped because this audit compares numerical results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd

from waferpulse.agents.groq_model_selection_agent import AgenticSearchDecision
from waferpulse.config import WaferPulsePaths
from waferpulse.crews import BoschPredictionCrew, EquipmentPredictionCrew


def baseline_module(root: Path, output: Path, relative: str, baseline_ref: str) -> ModuleType:
    """Keep an inspectable copy of the exact original code used in this audit."""
    source = subprocess.run(
        ["git", "show", f"{baseline_ref}:{relative}"], cwd=root, check=True,
        capture_output=True, encoding="utf-8",
    ).stdout
    destination = output / "baseline_sources" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    name = "_pipeline_verification_" + Path(relative).stem
    module = ModuleType(name)
    module.__file__ = str(destination)
    sys.modules[name] = module
    exec(compile(source, str(destination), "exec"), module.__dict__)
    return module


def compare_frames(before: pd.DataFrame, after: pd.DataFrame, keys: list[str]) -> dict:
    before = before.sort_values(keys).reset_index(drop=True)
    after = after.sort_values(keys).reset_index(drop=True)
    result = {"rows_before": len(before), "rows_after": len(after)}
    try:
        pd.testing.assert_frame_equal(
            before, after, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12,
        )
        result["equal"] = True
    except AssertionError as exc:
        result.update(equal=False, difference=str(exc)[:600])
    if len(before) == len(after) and list(before.columns) == list(after.columns):
        differences = {}
        for column in before.select_dtypes(include="number").columns:
            delta = np.abs(before[column].to_numpy() - after[column].to_numpy())
            differences[column] = float(np.nanmax(delta)) if len(delta) else 0.0
        result["max_absolute_differences"] = differences
    return result


def fixed_decision(profile: str) -> AgenticSearchDecision:
    return AgenticSearchDecision(
        enabled=False, used_agent=False, search_profile=profile,
        model="verification-fixed-profile", max_iterations=8,
        status="verification_fixed_profile", reason="Hold planning constant for comparison.",
    )


def equipment_comparison(root: Path, output: Path, baseline_ref: str) -> dict:
    from waferpulse.tools import equipment_modeling

    paths = WaferPulsePaths.from_root(root)
    old_model = baseline_module(root, output, "waferpulse/agents/equipment_model_agent.py", baseline_ref)
    old_crew = baseline_module(root, output, "waferpulse/crews/equipment_prediction_crew.py", baseline_ref)
    settings = dict(n_splits=3, selected_features=60, n_estimators=60)
    with patch.object(equipment_modeling, "_generate_shap_beeswarm", lambda *args: None):
        print("Equipment BEFORE: original coordinator, all current candidate models", flush=True)
        before = old_crew.EquipmentPredictionCrew(model_agent=old_model.EquipmentModelAgent()).run(
            paths.equipment_data, output / "before", log_func=print, **settings,
        )
        print("Equipment AFTER: shared coordinator, identical settings", flush=True)
        after = EquipmentPredictionCrew().run(
            paths.equipment_data, output / "after", log_func=print, **settings,
        )
    comparisons = {
        "wafer_predictions": compare_frames(before.predictions, after.predictions, ["lot", "wafer"]),
        "fold_metrics": compare_frames(before.fold_metrics, after.fold_metrics, ["task", "fold"]),
        "feature_importance": compare_frames(
            before.feature_importance, after.feature_importance, ["task", "feature"]
        ),
        "candidate_scores_equal": before.metrics["candidate_comparison"] == after.metrics["candidate_comparison"],
        "regression_scores_equal": before.metrics["regression"] == after.metrics["regression"],
        "classification_scores_equal": before.metrics["classification"] == after.metrics["classification"],
    }
    return {
        "settings": {**settings, "search_profile": "balanced", "equipment_stages": "both"},
        "before": before.metrics,
        "after": after.metrics,
        "comparisons": comparisons,
        "note": "Current dashboard defaults on all available matched wafers; historical saved hurdle-model artifacts use an older candidate set.",
    }


def bosch_comparison(root: Path, output: Path, baseline_ref: str) -> dict:
    from waferpulse.agents import prediction_model_agent
    from waferpulse.experiments import bosch_plasma_etch_benchmark as current_bosch
    from waferpulse.experiments import bosch_response_then_classify_benchmark as current_screening

    paths = WaferPulsePaths.from_root(root)
    old_bosch = baseline_module(root, output, "waferpulse/experiments/bosch_plasma_etch_benchmark.py", baseline_ref)
    old_screening = baseline_module(root, output, "waferpulse/experiments/bosch_response_then_classify_benchmark.py", baseline_ref)
    old_screening.load_dataset = old_bosch.load_dataset
    old_screening.model_candidates = old_bosch.model_candidates
    old_screening._group_oof_predictions = old_bosch._group_oof_predictions
    # Plot generation does not enter any model selection or prediction calculation.
    old_bosch._generate_bosch_shap = lambda *args: None
    profile = "conservative"  # the profile recorded in the existing Bosch summary
    threshold = 44.0
    print("Bosch BEFORE: original regression and screening functions", flush=True)
    old_bosch.run(paths.bosch_data, output / "before_regression", search_profile=profile)
    old_screening.run(paths.bosch_data, output / "before_screening", threshold)
    print("Bosch AFTER: shared coordinator, conservative profile for both tasks", flush=True)
    with patch.object(current_bosch, "_generate_bosch_shap", lambda *args: None), patch.object(
        prediction_model_agent, "choose_search_profile_from_summary", return_value=fixed_decision(profile)
    ):
        after = BoschPredictionCrew().run(
            paths.bosch_data, output / "after_regression", output / "after_screening",
            threshold=threshold, log_func=print,
        )
    print("Bosch CONTROL: new screening with the original balanced settings", flush=True)
    current_screening.train_bosch_screening(
        after.dataset, output / "after_screening_balanced_control", threshold, "balanced"
    )
    regression_keys = ["task", "target", "feature_set", "model"]
    comparisons = {}
    for label, filename, keys in [
        ("regression_metrics", "metrics.csv", regression_keys),
        ("regression_predictions", "oof_predictions.csv", regression_keys + ["experiment_key", "lot_number"]),
    ]:
        comparisons[label] = compare_frames(
            pd.read_csv(output / "before_regression" / filename),
            pd.read_csv(output / "after_regression" / filename), keys,
        )
    for destination, label in [
        ("after_screening", "screening_shared_profile"),
        ("after_screening_balanced_control", "screening_same_profile_control"),
    ]:
        comparisons[label] = compare_frames(
            pd.read_csv(output / "before_screening" / "oof_response_then_class.csv"),
            pd.read_csv(output / destination / "oof_response_then_class.csv"),
            ["model", "experiment_key", "lot_number"],
        )
    summaries = {
        name: json.loads((output / name / "summary.json").read_text(encoding="utf-8"))
        for name in ["before_regression", "after_regression", "before_screening", "after_screening", "after_screening_balanced_control"]
    }
    return {
        "settings": {"regression_profile": profile, "original_screening_profile": "balanced", "new_screening_profile": profile, "threshold": threshold},
        "comparisons": comparisons,
        "summaries": summaries,
    }


def main() -> None:
    # Windows redirected consoles may otherwise reject the model log's R² text.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["equipment", "bosch"], required=True)
    parser.add_argument("--output", type=Path, default=Path("output/shared_pipeline_verification"))
    parser.add_argument("--baseline-ref", default="HEAD", help="Original Git commit to compare against.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve() / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    result = (equipment_comparison if args.dataset == "equipment" else bosch_comparison)(root, output, args.baseline_ref)
    result["baseline_commit"] = subprocess.run(
        ["git", "rev-parse", args.baseline_ref], cwd=root, check=True, capture_output=True, encoding="utf-8"
    ).stdout.strip()
    result["source_sha256"] = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((output / "baseline_sources").rglob("*.py"))
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
