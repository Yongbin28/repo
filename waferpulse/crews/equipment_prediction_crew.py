"""Equipment settings for the shared manufacturing prediction workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from waferpulse.agents import EquipmentDataAgent, EquipmentModelAgent
from waferpulse.tools.equipment_pipeline import (
    DEFAULT_MAX_INVALID_RATE,
    DEFAULT_SELECTED_FEATURES,
    EquipmentDataset,
    EquipmentModelResult,
)
from waferpulse.agents.groq_model_selection_agent import DEFAULT_MAX_AGENT_ITERATIONS
from waferpulse.crews.prediction_crew import PredictionCrew


class EquipmentPredictionCrew(PredictionCrew[EquipmentDataset, EquipmentModelResult]):
    """Keep the equipment API while using the same coordinator as Bosch."""

    def __init__(
        self,
        data_agent: Optional[EquipmentDataAgent] = None,
        model_agent: Optional[EquipmentModelAgent] = None,
    ) -> None:
        super().__init__(
            data_agent=data_agent or EquipmentDataAgent(),
            model_agent=model_agent or EquipmentModelAgent(),
        )

    def validate(
        self,
        data_dir: Path,
        stage_mode: str = "both",
        max_invalid_rate: float = DEFAULT_MAX_INVALID_RATE,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> EquipmentDataset:
        return super().validate(
            data_dir=data_dir,
            data_options={"stage_mode": stage_mode, "max_invalid_rate": max_invalid_rate},
            log_func=log_func,
        )

    def run(
        self,
        data_dir: Path,
        output_dir: Path,
        stage_mode: str = "both",
        max_invalid_rate: float = DEFAULT_MAX_INVALID_RATE,
        n_splits: int = 5,
        selected_features: int = DEFAULT_SELECTED_FEATURES,
        n_estimators: int = 180,
        use_agentic_controller: bool = False,
        groq_api_key: Optional[str] = None,
        max_agent_iterations: int = DEFAULT_MAX_AGENT_ITERATIONS,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> EquipmentModelResult:
        return super().run(
            data_dir=data_dir,
            output_dir=output_dir,
            data_options={"stage_mode": stage_mode, "max_invalid_rate": max_invalid_rate},
            model_options={
                "n_splits": n_splits,
                "selected_features": selected_features,
                "n_estimators": n_estimators,
                "use_agentic_controller": use_agentic_controller,
                "groq_api_key": groq_api_key,
                "max_agent_iterations": max_agent_iterations,
            },
            log_func=log_func,
        )
