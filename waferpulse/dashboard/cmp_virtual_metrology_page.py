"""Streamlit page for verified PHM16 CMP virtual metrology."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd
import plotly.express as px
import streamlit as st

from waferpulse.config import WaferPulsePaths
from waferpulse.core.data_lanes import CMP_VIRTUAL_METROLOGY
from waferpulse.dashboard.components import render_data_lane_notice
from waferpulse.tools.cmp_virtual_metrology import (
    CMPPredictionResult,
    load_verified_cmp_model,
    predict_cmp_traces,
)


PREDICTION_STATE_KEY = "cmp_virtual_metrology_prediction"


def _combine_uploaded_files(files: Iterable[Any]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for source_index, uploaded in enumerate(files):
        frame = pd.read_csv(uploaded, low_memory=False)
        frame["_source_index"] = source_index
        frames.append(frame)
    if not frames:
        raise ValueError("Select at least one PHM16-format CMP trace CSV")
    return pd.concat(frames, ignore_index=True)


def _find_local_sample(data_root: Path) -> Path | None:
    matches = sorted(data_root.rglob("CMP-training-000.csv")) if data_root.exists() else []
    return matches[0] if matches else None


def _render_verified_evidence(metadata: dict[str, Any]) -> None:
    final = metadata["final_metrics"]
    development = metadata["selected_development_metrics"]
    st.subheader("1) Verified Upstream → Downstream Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Untouched-campaign R²", f"{final['r2']:.4f}", "PASS > 0.8")
    c2.metric("Untouched RMSE", f"{final['rmse']:.3f}")
    c3.metric("Untouched MAE", f"{final['mae']:.3f}")
    c4.metric("Development grouped R²", f"{development['r2']:.4f}")
    st.success(
        "LightGBM was selected using only campaigns 0–14, then evaluated once on "
        "untouched latest campaigns 15–18. The strict R² gate passed."
    )
    st.caption(
        "Published data-quality rule: four corrupted MRR labels above 4,000 are "
        "excluded by a predeclared ceiling of 300. Retaining them reduces the same "
        "final-campaign sensitivity result to R² 0.0992; the verified 0.9615 score "
        "therefore applies only to the documented clean-label population."
    )
    st.warning(
        "Scope: this model predicts downstream CMP material-removal-rate metrology "
        "from upstream polishing traces. It does not predict final electrical wafer "
        "test, yield, or field reliability, and its R² must not be assigned to the "
        "original EquipmentData response."
    )
    with st.expander("Validation contract and provenance", expanded=False):
        st.json(metadata)
        st.markdown(
            "Sources: [official PHM Society specification]"
            "(https://phmsociety.org/wp-content/uploads/2016/05/PHM16DataChallengeCFP.pdf) · "
            "[archived transport mirror](https://doi.org/10.5281/zenodo.19803296) · "
            "[published outlier evidence](https://doi.org/10.3390/app122211478)"
        )


def _render_benchmark(paths: WaferPulsePaths) -> None:
    benchmark_path = paths.cmp_output / "phm2016_cmp_benchmark.csv"
    if not benchmark_path.exists():
        return
    benchmark = pd.read_csv(benchmark_path)
    development = benchmark.loc[benchmark["lane"].eq("development_group_oof")].copy()
    development = development.sort_values("r2", ascending=False)
    st.subheader("2) Development-Campaign Algorithm Comparison")
    st.dataframe(
        development[["model", "r2", "rmse", "mae"]].style.format(
            {"r2": "{:.4f}", "rmse": "{:.3f}", "mae": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    figure = px.bar(
        development.sort_values("r2"),
        x="r2",
        y="model",
        orientation="h",
        title="Grouped development R² by algorithm",
    )
    figure.add_vline(x=0.8, line_dash="dash", line_color="#c62828")
    st.plotly_chart(figure, use_container_width=True)


def _run_prediction(frame: pd.DataFrame, model_path: Path) -> None:
    with st.spinner("Validating CMP traces and predicting downstream removal rate..."):
        result = predict_cmp_traces(frame, model_path=model_path)
    st.session_state[PREDICTION_STATE_KEY] = result


def _render_prediction_result(result: CMPPredictionResult) -> None:
    st.subheader("4) Predicted Downstream CMP Metrology")
    summary = result.input_summary
    p1, p2, p3 = st.columns(3)
    p1.metric("Trace rows", f"{summary['trace_rows']:,}")
    p2.metric("Wafer-stage predictions", f"{summary['wafer_stage_samples']:,}")
    p3.metric("Physical wafers", f"{summary['physical_wafers']:,}")
    if summary["target_present_in_input"]:
        st.info(
            "AVG_REMOVAL_RATE was present in the uploaded file but was excluded before "
            "feature extraction and prediction."
        )
    st.dataframe(result.predictions, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CMP virtual-metrology predictions",
        data=result.predictions.to_csv(index=False).encode("utf-8"),
        file_name="cmp_virtual_metrology_predictions.csv",
        mime="text/csv",
    )
    missing = summary.get("all_missing_model_features", [])
    if missing:
        st.caption(
            f"This batch did not exercise {len(missing)} chamber/phase feature columns; "
            "the verified pipeline used its training-fitted median imputers for them."
        )


def render_cmp_virtual_metrology_page(current_dir: Path, sidebar: Any) -> None:
    paths = WaferPulsePaths.from_root(current_dir)
    render_data_lane_notice(CMP_VIRTUAL_METROLOGY)
    st.header("🧪 CMP Upstream → Downstream Virtual Metrology")
    st.caption(
        "Upstream pressure, rotation, slurry, consumable and chamber traces → "
        "downstream average wafer material-removal rate."
    )

    try:
        verified = load_verified_cmp_model(paths.cmp_model)
    except Exception as exc:
        st.error(f"Verified model unavailable: {exc}")
        return
    _render_verified_evidence(dict(verified.metadata))
    _render_benchmark(paths)

    st.subheader("3) Run Verified Virtual Metrology")
    uploaded = st.file_uploader(
        "Upload one or more PHM16-format CMP trace CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Each row is a sensor timestamp; predictions are returned per WAFER_ID × STAGE.",
    )
    upload_col, sample_col = st.columns(2)
    if upload_col.button(
        "Predict uploaded CMP traces",
        type="primary",
        use_container_width=True,
        disabled=not uploaded,
    ):
        try:
            _run_prediction(_combine_uploaded_files(uploaded), paths.cmp_model)
            st.success("Verified downstream metrology predictions completed.")
        except Exception as exc:
            st.exception(exc)

    local_sample = _find_local_sample(paths.cmp_data)
    if sample_col.button(
        "Run bundled local sample",
        use_container_width=True,
        disabled=local_sample is None,
        help=(
            "Uses CMP-training-000.csv from the ignored local PHM16 data folder."
            if local_sample is not None
            else "Extract the PHM16 dataset under data/phm2016_cmp first."
        ),
    ):
        try:
            sample = pd.read_csv(local_sample, low_memory=False)
            sample["_source_index"] = 0
            _run_prediction(sample, paths.cmp_model)
            st.success(f"Predicted local sample: {local_sample.name}")
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get(PREDICTION_STATE_KEY)
    if result is not None:
        _render_prediction_result(result)


__all__ = ["render_cmp_virtual_metrology_page"]
