"""Agent 2 — grouped model validation, champion selection, and evidence output."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from waferpulse.tools.equipment_pipeline import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_SELECTED_FEATURES,
    EquipmentDataset,
    EquipmentModelResult,
    train_equipment_models,
)
from waferpulse.agents.groq_model_selection_agent import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_AGENT_ITERATIONS,
    choose_search_profile,
)


class EquipmentModelAgent:
    """Owns honest lot-grouped validation and persisted model evidence."""

    name = "Equipment Prediction & Evidence Agent"

    def run(
        self,
        dataset: EquipmentDataset,
        output_dir: Path,
        n_splits: int = 5,
        selected_features: int = DEFAULT_SELECTED_FEATURES,
        n_estimators: int = 180,
        random_state: int = DEFAULT_RANDOM_STATE,
        use_agentic_controller: bool = False,
        groq_api_key: Optional[str] = None,
        groq_model: str = DEFAULT_GROQ_MODEL,
        max_agent_iterations: int = DEFAULT_MAX_AGENT_ITERATIONS,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> EquipmentModelResult:
        decision = choose_search_profile(
            dataset,
            enabled=use_agentic_controller,
            api_key=groq_api_key,
            model=groq_model,
            max_iterations=max_agent_iterations,
            log_func=log_func,
        )
        return train_equipment_models(
            dataset=dataset,
            output_dir=Path(output_dir),
            n_splits=n_splits,
            selected_features=selected_features,
            n_estimators=n_estimators,
            random_state=random_state,
            search_profile=decision.search_profile,
            agentic_decision=decision.as_dict(),
            log_func=log_func,
        )
