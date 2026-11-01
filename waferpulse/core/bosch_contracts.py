"""Prepared Bosch data and results passed through the shared coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class BoschDataset:
    process: pd.DataFrame
    measurements: pd.DataFrame
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class BoschModelResult:
    dataset: BoschDataset
    regression_metrics: pd.DataFrame
    screening_metrics: pd.DataFrame
