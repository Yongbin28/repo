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
        log_func: Optional[Callable[[str], None]] = None,
    ) -> EquipmentModelResult:
        return train_equipment_models(
            dataset=dataset,
            output_dir=Path(output_dir),
            n_splits=n_splits,
            selected_features=selected_features,
            n_estimators=n_estimators,
            random_state=random_state,
            log_func=log_func,
        )
