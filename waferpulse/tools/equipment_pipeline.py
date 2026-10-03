"""Public facade for the reliable open-source EquipmentData workflow.

Implementation is separated into data-quality/feature tools and modelling tools.
The crew API in :mod:`waferpulse.crews` is preferred for application code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from waferpulse.core.equipment_contracts import (
    DEFAULT_MAX_INVALID_RATE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESPONSE_THRESHOLD,
    DEFAULT_SELECTED_FEATURES,
    EXPECTED_SAMPLES_PER_WAFER,
    PIPELINE_VERSION,
    STAGE_LABELS,
    EquipmentDataset,
    EquipmentModelResult,
    EquipmentTraceDataset,
)
from waferpulse.tools.equipment_data import load_equipment_dataset, load_equipment_trace_dataset
from waferpulse.tools.equipment_modeling import (
    _choose_recall_guardrail_threshold,
    train_equipment_models,
)


def run_equipment_pipeline(
    data_dir: Path,
    output_dir: Path,
    stage_mode: str = "both",
    max_invalid_rate: float = DEFAULT_MAX_INVALID_RATE,
    n_splits: int = 5,
    selected_features: int = DEFAULT_SELECTED_FEATURES,
    n_estimators: int = 180,
    search_profile: str = "balanced",
    agentic_decision: Optional[dict] = None,
    log_func: Optional[Callable[[str], None]] = None,
) -> EquipmentModelResult:
    dataset = load_equipment_dataset(
        data_dir=data_dir,
        stage_mode=stage_mode,
        max_invalid_rate=max_invalid_rate,
        log_func=log_func,
    )
    return train_equipment_models(
        dataset=dataset,
        output_dir=output_dir,
        n_splits=n_splits,
        selected_features=selected_features,
        n_estimators=n_estimators,
        search_profile=search_profile,
        agentic_decision=agentic_decision,
        log_func=log_func,
    )


__all__ = [
    "DEFAULT_MAX_INVALID_RATE",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_RESPONSE_THRESHOLD",
    "DEFAULT_SELECTED_FEATURES",
    "EXPECTED_SAMPLES_PER_WAFER",
    "PIPELINE_VERSION",
    "STAGE_LABELS",
    "EquipmentDataset",
    "EquipmentModelResult",
    "EquipmentTraceDataset",
    "load_equipment_dataset",
    "load_equipment_trace_dataset",
    "run_equipment_pipeline",
    "train_equipment_models",
]
