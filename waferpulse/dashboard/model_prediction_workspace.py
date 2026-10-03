"""Unified selection point for public model evidence and local data analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from waferpulse.core.data_lanes import LOCAL_DEMONSTRATION
from waferpulse.dashboard.bosch_plasma_etch_page import render_bosch_plasma_etch_page
from waferpulse.dashboard.components import render_data_lane_notice
from waferpulse.dashboard.dataset_analysis import (
    ZENODO_RECORD_URL,
    render_bosch_source_analysis,
    render_equipment_source_analysis,
    render_uploaded_dataset_analysis,
)
from waferpulse.dashboard.equipment_prediction_page import render_equipment_prediction_page

BOSCH_SOURCE = "Zenodo BOSCH plasma-etch benchmark"
EQUIPMENT_SOURCE = "Two-Stage Tool Sensor Telemetry (Multi-Chamber Benchmark)"
LOCAL_SOURCE = "Local dataset analysis"


def render_model_prediction_workspace(current_dir: Path, sidebar: Any) -> None:
    """Render a single GUI where the user selects the dataset before the workflow."""

    st.header("Model Prediction & Validation")
    st.caption(
        "Choose the data source first, then run the matching prediction workflow or inspect "
        "the locally available source data."
    )
    sidebar.markdown("---")
    sidebar.subheader("Prediction Dataset")
    source = sidebar.radio(
        "Select data source",
        options=[BOSCH_SOURCE, EQUIPMENT_SOURCE, LOCAL_SOURCE],
        key="prediction_dataset_source",
        help="Each public source uses its own schema-aware feature and model pipeline.",
    )

    if source == BOSCH_SOURCE:
        st.caption(f"Public source: [Zenodo record 17122442]({ZENODO_RECORD_URL})")
        prediction, analysis = st.tabs(["Model prediction", "Source data analysis"])
        with prediction:
            render_bosch_plasma_etch_page(current_dir, sidebar, show_intro=False)
        with analysis:
            st.subheader("Zenodo BOSCH local source explorer")
            render_bosch_source_analysis(current_dir)
        return

    if source == EQUIPMENT_SOURCE:
        prediction, analysis = st.tabs(["Model prediction", "Source data analysis"])
        with prediction:
            render_equipment_prediction_page(current_dir, sidebar, show_intro=False)
        with analysis:
            st.subheader("Two-Stage Multi-Chamber Local Source Explorer")
            render_equipment_source_analysis(current_dir)
        return

    render_data_lane_notice(LOCAL_DEMONSTRATION)
    st.subheader("Local dataset explorer")
    render_uploaded_dataset_analysis(sidebar)
