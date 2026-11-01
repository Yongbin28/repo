"""Bosch preparation and modelling adapters for the common agent workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from waferpulse.agents.prediction_model_agent import PredictionModelAgent
from waferpulse.core.bosch_contracts import BoschDataset, BoschModelResult


class BoschDataAgent:
    name = "Bosch Data & Feature Agent"

    def run(
        self,
        data_dir: Path,
        refresh_cache: bool = False,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> BoschDataset:
        from waferpulse.experiments.bosch_plasma_etch_benchmark import load_dataset

        return load_dataset(Path(data_dir), refresh_cache=refresh_cache)


class BoschModelAgent(PredictionModelAgent[BoschDataset, BoschModelResult]):
    name = "Bosch Prediction & Evidence Agent"

    def summarize(self, dataset: BoschDataset) -> dict[str, Any]:
        matched = dataset.measurements.loc[
            dataset.measurements["experiment_key"].isin(dataset.process["experiment_key"])
        ]
        return {
            "samples": int(matched["experiment_key"].nunique()),
            "features": int(len(dataset.process.columns.difference(["experiment_key"]))),
            "manufacturing_lots": int(matched["lot_number"].nunique()),
            "task": "wafer-average and spatial etch regression with research screening",
        }

    def train(
        self,
        dataset: BoschDataset,
        *,
        output_dir: Path,
        data_dir: Path,
        classification_output_dir: Path,
        threshold: float,
        search_profile: str,
        agentic_decision: Mapping[str, Any],
        log_func: Optional[Callable[[str], None]] = None,
    ) -> BoschModelResult:
        from waferpulse.experiments.bosch_plasma_etch_benchmark import train_bosch_models
        from waferpulse.experiments.bosch_response_then_classify_benchmark import (
            train_bosch_screening,
        )

        if log_func:
            log_func("[Bosch] Evaluating wafer-average and spatial regression models...")
        regression = train_bosch_models(
            dataset=dataset,
            root=Path(data_dir),
            output=output_dir,
            search_profile=search_profile,
            agentic_decision=agentic_decision,
        )
        if log_func:
            log_func("[Bosch] Evaluating response-based research screening...")
        screening = train_bosch_screening(
            dataset=dataset,
            output=Path(classification_output_dir),
            threshold=threshold,
            search_profile=search_profile,
            agentic_decision=agentic_decision,
        )
        return BoschModelResult(
            dataset=dataset,
            regression_metrics=regression,
            screening_metrics=screening,
        )
