"""Crew 1 — reliable EquipmentData prediction workflow.

The crew mirrors the clear sequential orchestration used by HydroSight while
remaining deterministic and dependency-light: Agent 1 validates/builds data,
then Agent 2 validates models and publishes evidence.
"""

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


class EquipmentPredictionCrew:
    """Coordinate the two real-data agents without mixing demonstration data."""

    def __init__(
        self,
        data_agent: Optional[EquipmentDataAgent] = None,
        model_agent: Optional[EquipmentModelAgent] = None,
    ) -> None:
        self.data_agent = data_agent or EquipmentDataAgent()
        self.model_agent = model_agent or EquipmentModelAgent()

    def validate(
        self,
        data_dir: Path,
        stage_mode: str = "both",
        max_invalid_rate: float = DEFAULT_MAX_INVALID_RATE,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> EquipmentDataset:
        if log_func:
            log_func(f"[Crew / Agent 1] {self.data_agent.name}")
        return self.data_agent.run(
            data_dir=data_dir,
            stage_mode=stage_mode,
            max_invalid_rate=max_invalid_rate,
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
        dataset = self.validate(
            data_dir=data_dir,
            stage_mode=stage_mode,
            max_invalid_rate=max_invalid_rate,
            log_func=log_func,
        )
        if log_func:
            log_func(f"[Crew / Agent 2] {self.model_agent.name}")
        return self.model_agent.run(
            dataset=dataset,
            output_dir=output_dir,
            n_splits=n_splits,
            selected_features=selected_features,
            n_estimators=n_estimators,
            use_agentic_controller=use_agentic_controller,
            groq_api_key=groq_api_key,
            max_agent_iterations=max_agent_iterations,
            log_func=log_func,
        )
