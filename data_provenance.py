"""Backward-compatible import facade for the WaferPulse data-lane policy.

New code should import :mod:`waferpulse.core.data_lanes` directly.
"""

from waferpulse.core.data_lanes import *  # noqa: F401,F403
