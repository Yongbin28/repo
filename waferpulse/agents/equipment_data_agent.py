"""Agent 1 — validate public equipment data and build temporal features."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from waferpulse.tools.equipment_pipeline import (
    DEFAULT_MAX_INVALID_RATE,
    DEFAULT_RESPONSE_THRESHOLD,
    EquipmentDataset,
    load_equipment_dataset,
)


class EquipmentDataAgent:
    """Owns source validation, provenance, matching, and feature engineering."""

    name = "Equipment Data & Feature Agent"

    def run(
        self,
        data_dir: Path,
        stage_mode: str = "both",
        max_invalid_rate: float = DEFAULT_MAX_INVALID_RATE,
        response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> EquipmentDataset:
        return load_equipment_dataset(
            data_dir=Path(data_dir),
            stage_mode=stage_mode,
            max_invalid_rate=max_invalid_rate,
            response_threshold=response_threshold,
            log_func=log_func,
        )
