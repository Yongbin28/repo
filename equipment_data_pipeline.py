"""Backward-compatible import facade for the real EquipmentData pipeline.

New code should use the agent/crew API under :mod:`waferpulse`.
"""

from waferpulse.tools.equipment_pipeline import *  # noqa: F401,F403

# Private helpers remain available for the existing focused unit tests.
from waferpulse.tools.equipment_pipeline import (  # noqa: F401
    _choose_recall_guardrail_threshold,
)
