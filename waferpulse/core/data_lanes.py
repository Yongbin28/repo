"""Data-lane policy and provenance helpers for WaferPulse.

The application deliberately separates claims supported by real open-source
semiconductor data from functions demonstrated with local or generated data.
This module is intentionally dependency-free so every pipeline can enforce the
same policy without importing the Streamlit application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple


REAL_OPEN_SOURCE = "real_open_source"
CMP_VIRTUAL_METROLOGY = "cmp_virtual_metrology"
BOSCH_PLASMA_ETCH = "bosch_plasma_etch"
LOCAL_DEMONSTRATION = "local_demonstration"
REAL_STDF_SAMPLE = "real_stdf_sample"


@dataclass(frozen=True)
class DataLane:
    key: str
    label: str
    evidence_level: str
    description: str
    allowed_functions: Tuple[str, ...]
    forbidden_claims: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


DATA_LANES: Dict[str, DataLane] = {
    REAL_OPEN_SOURCE: DataLane(
        key=REAL_OPEN_SOURCE,
        label="REAL OPEN-SOURCE DATA",
        evidence_level="Model validation",
        description=(
            "Public semiconductor equipment-sensor measurements with wafer response labels. "
            "Used for data validation, temporal feature engineering, prediction, and "
            "model explanation."
        ),
        allowed_functions=(
            "data_quality",
            "feature_engineering",
            "regression",
            "classification",
            "explainability",
            "grouped_validation",
        ),
        forbidden_claims=(
            "Direct wafer-probe to final-test validation",
            "Field reliability or MTTF validation",
        ),
    ),
    CMP_VIRTUAL_METROLOGY: DataLane(
        key=CMP_VIRTUAL_METROLOGY,
        label="REAL OPEN-SOURCE CMP VIRTUAL METROLOGY",
        evidence_level="Prospective campaign validation",
        description=(
            "PHM Society 2016 upstream CMP process traces paired with downstream "
            "wafer material-removal-rate metrology. Used for continuous virtual "
            "metrology and process-performance prediction."
        ),
        allowed_functions=(
            "data_quality",
            "feature_engineering",
            "regression",
            "explainability",
            "grouped_validation",
            "prospective_campaign_validation",
        ),
        forbidden_claims=(
            "Final electrical wafer-test or yield prediction",
            "Field reliability or MTTF validation",
            "Transfer of PHM16 R-squared to the original EquipmentData response",
        ),
    ),
    BOSCH_PLASMA_ETCH: DataLane(
        key=BOSCH_PLASMA_ETCH,
        label="REAL OPEN-SOURCE BOSCH PLASMA ETCH",
        evidence_level="Manufacturing-lot-held-out validation",
        description=(
            "Public upstream Bosch plasma-etch equipment traces paired with "
            "downstream 89-point silicon and oxide etch measurements. Used for "
            "wafer-average silicon-etch prediction and spatial virtual metrology."
        ),
        allowed_functions=(
            "data_quality",
            "feature_engineering",
            "regression",
            "classification",
            "explainability",
            "grouped_validation",
            "spatial_virtual_metrology",
        ),
        forbidden_claims=(
            "Official production pass/fail classification",
            "Final electrical wafer-test or yield prediction",
            "Field reliability or MTTF validation",
            "Pure sensor-only interpretation of coordinate-assisted map scores",
        ),
    ),
    LOCAL_DEMONSTRATION: DataLane(
        key=LOCAL_DEMONSTRATION,
        label="LOCAL / SYNTHETIC DEMONSTRATION",
        evidence_level="Functional verification",
        description=(
            "Locally available or specification-generated STDF-style measurements. "
            "Used to demonstrate monitoring, historical comparison, spatial analysis, "
            "alerts, and dashboard integration."
        ),
        allowed_functions=(
            "stdf_parsing",
            "historical_baseline",
            "spc",
            "psi",
            "cpk",
            "gdbn",
            "alerts",
            "dashboard",
        ),
        forbidden_claims=(
            "Real-world predictive accuracy",
            "Production deployment readiness",
        ),
    ),
    REAL_STDF_SAMPLE: DataLane(
        key=REAL_STDF_SAMPLE,
        label="REAL STDF SAMPLE",
        evidence_level="Parser verification",
        description="A real STDF sample used to verify parsing and die-level feature extraction.",
        allowed_functions=("stdf_parsing", "feature_extraction", "gdbn"),
        forbidden_claims=("Downstream final-test prediction accuracy",),
    ),
}


class DataLaneViolation(ValueError):
    """Raised when a function is requested from a lane that cannot support it."""


def get_data_lane(key: str) -> DataLane:
    try:
        return DATA_LANES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown data lane: {key!r}") from exc


def assert_functions_allowed(lane_key: str, functions: Iterable[str]) -> None:
    lane = get_data_lane(lane_key)
    requested = tuple(dict.fromkeys(functions))
    unsupported = sorted(set(requested).difference(lane.allowed_functions))
    if unsupported:
        raise DataLaneViolation(
            f"{lane.label} cannot support: {', '.join(unsupported)}. "
            f"Allowed functions: {', '.join(lane.allowed_functions)}."
        )


def build_provenance_record(
    lane_key: str,
    dataset_name: str,
    source_files: Iterable[Path],
    functions: Iterable[str],
    dataset_version: str,
) -> Dict[str, object]:
    requested = tuple(dict.fromkeys(functions))
    assert_functions_allowed(lane_key, requested)
    lane = get_data_lane(lane_key)
    return {
        "lane": lane.to_dict(),
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "source_files": [str(Path(path).resolve()) for path in source_files],
        "functions": list(requested),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_synthetic": lane_key == LOCAL_DEMONSTRATION,
    }
