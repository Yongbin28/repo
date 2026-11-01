"""Bosch settings for the shared manufacturing prediction workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from waferpulse.agents.bosch_prediction_agents import BoschDataAgent, BoschModelAgent
from waferpulse.agents.groq_model_selection_agent import DEFAULT_MAX_AGENT_ITERATIONS
from waferpulse.core.bosch_contracts import BoschDataset, BoschModelResult
from waferpulse.crews.prediction_crew import PredictionCrew


class BoschPredictionCrew(PredictionCrew[BoschDataset, BoschModelResult]):
    def __init__(
        self,
        data_agent: Optional[BoschDataAgent] = None,
        model_agent: Optional[BoschModelAgent] = None,
    ) -> None:
        super().__init__(
            data_agent=data_agent or BoschDataAgent(),
            model_agent=model_agent or BoschModelAgent(),
        )

    def validate(
        self,
        data_dir: Path,
        refresh_cache: bool = False,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> BoschDataset:
        return super().validate(
            data_dir=data_dir,
            data_options={"refresh_cache": refresh_cache},
            log_func=log_func,
        )

    def run(
        self,
        data_dir: Path,
        output_dir: Path,
        classification_output_dir: Path,
        threshold: float = 44.0,
        refresh_cache: bool = False,
        use_agentic_controller: bool = False,
        groq_api_key: Optional[str] = None,
        max_agent_iterations: int = DEFAULT_MAX_AGENT_ITERATIONS,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> BoschModelResult:
        return super().run(
            data_dir=data_dir,
            output_dir=output_dir,
            data_options={"refresh_cache": refresh_cache},
            model_options={
                "data_dir": data_dir,
                "classification_output_dir": classification_output_dir,
                "threshold": threshold,
                "use_agentic_controller": use_agentic_controller,
                "groq_api_key": groq_api_key,
                "max_agent_iterations": max_agent_iterations,
            },
            log_func=log_func,
        )
