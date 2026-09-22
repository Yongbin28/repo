"""Shared contracts and constants for the EquipmentData workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


PIPELINE_VERSION = "1.3.0"
DEFAULT_RESPONSE_THRESHOLD = 0.75
EXPECTED_SAMPLES_PER_WAFER = 176
DEFAULT_MAX_INVALID_RATE = 0.01
DEFAULT_SELECTED_FEATURES = 120
DEFAULT_RANDOM_STATE = 42

STAGE_LABELS = {
    "both": "Both equipment stages",
    "equipment1": "Equipment 1 only",
    "equipment2": "Equipment 2 only",
}


@dataclass
class EquipmentDataset:
    features: pd.DataFrame
    response: pd.Series
    bad_label: pd.Series
    groups: pd.Series
    identity: pd.DataFrame
    sensor_quality: pd.DataFrame
    quality_summary: Dict[str, Any]
    provenance: Dict[str, Any]


@dataclass
class EquipmentTraceDataset:
    """Validated per-wafer traces before temporal aggregation.

    ``traces`` has shape ``(wafer, sensor, timestamp)``. Sensor names are
    stage-qualified so equipment channels remain unambiguous after joining.
    """

    traces: np.ndarray
    sensor_names: Tuple[str, ...]
    response: pd.Series
    bad_label: pd.Series
    groups: pd.Series
    identity: pd.DataFrame
    sensor_quality: pd.DataFrame
    quality_summary: Dict[str, Any]


@dataclass
class EquipmentModelResult:
    metrics: Dict[str, Any]
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    quality_summary: Dict[str, Any]
    sensor_quality: pd.DataFrame
    provenance: Dict[str, Any]
    artifact_paths: Dict[str, str]
