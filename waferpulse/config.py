"""Application paths and defaults shared by crews and dashboard pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WaferPulsePaths:
    root: Path
    equipment_data: Path
    equipment_output: Path
    cmp_data: Path
    cmp_output: Path
    cmp_model: Path
    bosch_data: Path
    bosch_output: Path
    bosch_classification_output: Path

    @classmethod
    def from_root(cls, root: Path) -> "WaferPulsePaths":
        resolved = Path(root).resolve()
        return cls(
            root=resolved,
            equipment_data=resolved / "EquipmentData",
            equipment_output=resolved / "output" / "equipment_quality",
            cmp_data=resolved / "data" / "phm2016_cmp",
            cmp_output=resolved / "output" / "phm2016_cmp",
            cmp_model=(
                resolved
                / "waferpulse"
                / "artifacts"
                / "phm2016_cmp_verified_model.joblib"
            ),
            bosch_data=(
                resolved / "dataset" / "bosch_plasma_etch"
                if (resolved / "dataset" / "bosch_plasma_etch").is_dir()
                else resolved / "data" / "bosch_plasma_etch"
            ),
            bosch_output=resolved / "output" / "bosch_plasma_etch",
            bosch_classification_output=(
                resolved / "output" / "bosch_response_then_classify"
            ),
        )
