"""Small reusable Streamlit presentation components."""

from __future__ import annotations

import streamlit as st

from waferpulse.core.data_lanes import (
    BOSCH_PLASMA_ETCH,
    CMP_VIRTUAL_METROLOGY,
    REAL_OPEN_SOURCE,
    get_data_lane,
)


def render_data_lane_notice(lane_key: str) -> None:
    """Render the evidence boundary for the active analytical lane."""
    lane = get_data_lane(lane_key)
    if lane_key in {REAL_OPEN_SOURCE, BOSCH_PLASMA_ETCH, CMP_VIRTUAL_METROLOGY}:
        background, border, text = "#e8f5e9", "#2e7d32", "#1b5e20"
    else:
        background, border, text = "#fff8e1", "#f9a825", "#795548"
    st.markdown(
        f"""
        <div style="background:{background}; border-left:5px solid {border};
                    padding:0.8rem 1rem; border-radius:0.35rem; margin:0.5rem 0 1rem 0;">
          <div style="font-weight:800; color:{text}; letter-spacing:0.04em;">{lane.label}</div>
          <div style="color:{text};"><b>{lane.evidence_level}.</b> {lane.description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
