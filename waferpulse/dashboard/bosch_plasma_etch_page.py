"""Streamlit page for the open-source Bosch plasma-etch benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from waferpulse.config import WaferPulsePaths
from waferpulse.core.data_lanes import BOSCH_PLASMA_ETCH
from waferpulse.dashboard.components import render_data_lane_notice

DEFAULT_RESEARCH_THRESHOLD = 44.0
ZENODO_RECORD_URL = "https://zenodo.org/records/17122442"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_metrics(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    for column in ("r2", "rmse", "mae"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def _read_predictions_cached(path_text: str, modified_ns: int, size: int) -> pd.DataFrame:
    del modified_ns, size
    return pd.read_csv(path_text)


def _read_predictions(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        stat = path.stat()
        return _read_predictions_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except (OSError, pd.errors.ParserError):
        return None


def _render_dataset_summary(summary: dict[str, Any]) -> None:
    dataset = summary.get("dataset", {})
    st.subheader("1) Dataset and Validation Contract")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Process traces", f"{dataset.get('process_traces', 0):,}")
    c2.metric("Matched wafers", f"{dataset.get('matched_wafers', 0):,}")
    c3.metric("Manufacturing lots", f"{dataset.get('lots', 0):,}")
    c4.metric("Etch-map points", f"{dataset.get('measurement_rows', 0):,}")
    st.caption(summary.get("validation", "Manufacturing-lot-held-out validation"))
    st.info(summary.get("predictor_policy", "Upstream process summaries only."))


def _render_wafer_average(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame | None,
) -> None:
    st.subheader("2) Wafer-Average Silicon-Etch Prediction")
    wafer = metrics.loc[
        metrics["task"].eq("wafer_mean")
        & metrics["target"].eq("si_etch")
        & metrics["feature_set"].eq("process_only")
    ].sort_values("r2", ascending=False)
    if wafer.empty:
        st.warning("Wafer-average silicon-etch benchmark results are unavailable.")
        return

    best = wafer.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", str(best["model"]).replace("_", " ").title())
    c2.metric("Lot-held-out R²", f"{best['r2']:.4f}")
    c3.metric("RMSE", f"{best['rmse']:.4f}")
    c4.metric("MAE", f"{best['mae']:.4f}")
    st.success(
        "This is the primary sensor-only result: upstream process summaries predict "
        "the downstream wafer-average silicon etch measurement."
    )

    figure = px.bar(
        wafer.sort_values("r2"),
        x="r2",
        y="model",
        orientation="h",
        title="Five-fold lot-held-out R² by regression model",
        labels={"r2": "R²", "model": "Model"},
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color="#616161")
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        wafer[["model", "rows", "groups", "r2", "rmse", "mae"]].style.format(
            {"r2": "{:.4f}", "rmse": "{:.4f}", "mae": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    if predictions is None:
        return
    champion = str(best["model"])
    ledger = predictions.loc[
        predictions["task"].eq("wafer_mean")
        & predictions["target"].eq("si_etch")
        & predictions["feature_set"].eq("process_only")
        & predictions["model"].eq(champion)
    ].copy()
    if ledger.empty:
        return
    figure = px.scatter(
        ledger,
        x="actual",
        y="predicted",
        color="lot_number",
        hover_data=["experiment_key"],
        title=f"Out-of-fold wafer predictions: {champion.replace('_', ' ').title()}",
        labels={
            "actual": "Measured average silicon etch",
            "predicted": "Predicted average silicon etch",
        },
    )
    minimum = float(min(ledger["actual"].min(), ledger["predicted"].min()))
    maximum = float(max(ledger["actual"].max(), ledger["predicted"].max()))
    figure.add_shape(
        type="line",
        x0=minimum,
        y0=minimum,
        x1=maximum,
        y1=maximum,
        line=dict(color="#616161", dash="dash"),
    )
    st.plotly_chart(figure, use_container_width=True)
    with st.expander("Wafer-average prediction ledger", expanded=False):
        st.dataframe(
            ledger[["experiment_key", "lot_number", "actual", "predicted"]],
            use_container_width=True,
            hide_index=True,
        )


def _render_spatial_metrology(metrics: pd.DataFrame) -> None:
    st.subheader("3) Spatial Silicon-Etch Virtual Metrology")
    spatial = metrics.loc[
        metrics["task"].eq("map_point") & metrics["target"].eq("si_etch")
    ].sort_values("r2", ascending=False)
    if spatial.empty:
        st.warning("Spatial silicon-etch benchmark results are unavailable.")
        return

    combined = spatial.loc[spatial["feature_set"].eq("process_plus_coordinate")]
    coordinate = spatial.loc[spatial["feature_set"].eq("coordinate_only")]
    process_only = spatial.loc[spatial["feature_set"].eq("process_only")]
    c1, c2, c3 = st.columns(3)
    if not combined.empty:
        c1.metric("Process + X/Y R²", f"{combined.iloc[0]['r2']:.4f}")
    if not coordinate.empty:
        c2.metric("Coordinate-only R²", f"{coordinate.iloc[0]['r2']:.4f}")
    if not process_only.empty:
        c3.metric("Process-only point R²", f"{process_only.iloc[0]['r2']:.4f}")

    st.warning(
        "The high spatial-map score uses the known X/Y measurement location. Because "
        "the coordinate-only baseline is already strong, this result is spatial virtual "
        "metrology—not pure equipment-sensor prediction."
    )
    comparison = (
        spatial.sort_values("r2", ascending=False)
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("r2", ascending=False)
    )
    st.dataframe(
        comparison[["feature_set", "model", "rows", "groups", "r2", "rmse", "mae"]].style.format(
            {"r2": "{:.4f}", "rmse": "{:.4f}", "mae": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_research_classification(
    output_dir: Path,
    selected_threshold: float,
) -> None:
    st.subheader("4) Research High-Etch Screening")
    summary = _read_json(output_dir / "summary.json")
    metrics = _read_metrics(output_dir / "metrics.csv")
    if summary is None or metrics is None or metrics.empty:
        st.info("Run the research-threshold benchmark to generate classification evidence.")
        return

    stored_threshold = float(metrics.iloc[0].get("research_threshold", float("nan")))
    if abs(stored_threshold - selected_threshold) > 1e-9:
        st.info(
            f"Displayed evidence uses threshold {stored_threshold:.2f}; select "
            "'Rebuild research screening' to evaluate the chosen threshold."
        )
    best_r2 = summary.get("best_r2", {})
    best_f1 = summary.get("best_f1_high_etch", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Research threshold", f"{stored_threshold:.2f}")
    c2.metric("Best response R²", f"{float(best_r2.get('r2', 0.0)):.4f}")
    c3.metric("High-etch F1", f"{float(best_f1.get('f1_bad', 0.0)):.4f}")
    c4.metric("High-etch recall", f"{float(best_f1.get('recall_bad', 0.0)):.1%}")
    predictions = _read_predictions(output_dir / "oof_response_then_class.csv")
    champion = str(best_f1.get("model", ""))
    if predictions is not None and champion:
        ledger = predictions.loc[predictions["model"].astype(str).eq(champion)].copy()
        if not ledger.empty:
            figure = px.scatter(
                ledger,
                x="actual_average_si_etch",
                y="predicted_average_si_etch",
                color="actual_research_class",
                symbol="predicted_research_class",
                hover_data=["experiment_key", "lot_number"],
                title=f"Research screening predictions: {champion.replace('_', ' ').title()}",
                labels={
                    "actual_average_si_etch": "Measured average silicon etch",
                    "predicted_average_si_etch": "Predicted average silicon etch",
                },
            )
            figure.add_hline(y=stored_threshold, line_dash="dash", line_color="#c62828")
            figure.add_vline(x=stored_threshold, line_dash="dash", line_color="#c62828")
            st.plotly_chart(figure, use_container_width=True)


def _render_explainability(output_dir: Path) -> None:
    st.subheader("5) Explainable Process Drivers (SHAP Beeswarm Plot)")
    shap_svg = output_dir / "bosch_shap_beeswarm.svg"
    shap_png = output_dir / "bosch_shap_beeswarm.png"
    if shap_svg.exists() or shap_png.exists():
        rendered = False
        if shap_svg.exists():
            try:
                svg_content = shap_svg.read_text(encoding="utf-8")
                st.image(
                    svg_content,
                    caption="SHAP Summary (Beeswarm) Plot: Distribution and direction of process feature impact on Silicon Etch Rate (red = high feature value, blue = low)",
                    use_container_width=True,
                )
                rendered = True
            except Exception:
                rendered = False
        if not rendered and shap_png.exists():
            st.image(
                str(shap_png),
                caption="SHAP Summary (Beeswarm) Plot: Distribution and direction of process feature impact on Silicon Etch Rate (red = high feature value, blue = low)",
                use_container_width=True,
            )


def _run_regression(paths: WaferPulsePaths, refresh_cache: bool) -> None:
    try:
        from waferpulse.experiments.bosch_plasma_etch_benchmark import run

        with st.spinner("Running lot-held-out Bosch plasma-etch benchmark..."):
            run(paths.bosch_data, paths.bosch_output, refresh_cache=refresh_cache)
        st.success("Bosch regression evidence was rebuilt successfully.")
    except Exception as exc:
        st.error(
            "The Bosch benchmark could not run. Confirm the dataset is present and "
            "install requirements-experiments.txt."
        )
        st.exception(exc)


def _run_classification(paths: WaferPulsePaths, threshold: float) -> None:
    try:
        from waferpulse.experiments.bosch_response_then_classify_benchmark import run

        with st.spinner("Running response-first high-etch screening benchmark..."):
            run(paths.bosch_data, paths.bosch_classification_output, threshold)
        st.success("Bosch research-screening evidence was rebuilt successfully.")
    except Exception as exc:
        st.error(
            "The Bosch screening benchmark could not run. Confirm the dataset is "
            "present and install requirements-experiments.txt."
        )
        st.exception(exc)


def render_bosch_plasma_etch_page(
    current_dir: Path,
    sidebar: Any,
    show_intro: bool = True,
) -> None:
    """Render the Bosch plasma-etch evidence workspace."""

    paths = WaferPulsePaths.from_root(current_dir)
    render_data_lane_notice(BOSCH_PLASMA_ETCH)
    if show_intro:
        st.header("Bosch Plasma Etch Quality Prediction")
    else:
        st.subheader("Zenodo BOSCH plasma-etch prediction")
    st.caption(
        "Upstream 5 Hz process traces → downstream wafer-average silicon etch and "
        f"89-point spatial virtual metrology · [Zenodo record 17122442]({ZENODO_RECORD_URL})"
    )

    sidebar.markdown("---")
    sidebar.subheader("Bosch Benchmark")
    threshold = sidebar.slider(
        "Research high-etch threshold",
        min_value=40.0,
        max_value=48.0,
        value=DEFAULT_RESEARCH_THRESHOLD,
        step=0.25,
        help="Research rule only; the public dataset provides no factory limit.",
    )
    refresh_cache = sidebar.checkbox("Rebuild process-feature cache", value=False)

    run_regression, run_screening = st.columns(2)
    if run_regression.button(
        "Rebuild lot-held-out regression",
        type="primary",
        use_container_width=True,
    ):
        _run_regression(paths, refresh_cache)
    if run_screening.button(
        "Rebuild research screening",
        use_container_width=True,
    ):
        _run_classification(paths, threshold)

    summary = _read_json(paths.bosch_output / "summary.json")
    metrics = _read_metrics(paths.bosch_output / "metrics.csv")
    if summary is None or metrics is None or metrics.empty:
        st.warning(
            "Bosch evidence is not available yet. Place Process_data.nc, "
            "Dictionary_process.nc, and Si_Oxide_etch_89_points.csv under "
            f"{paths.bosch_data}, then rebuild the benchmark."
        )
        return

    _render_dataset_summary(summary)
    predictions = _read_predictions(paths.bosch_output / "oof_predictions.csv")
    _render_wafer_average(metrics, predictions)
    _render_spatial_metrology(metrics)
    _render_research_classification(paths.bosch_classification_output, threshold)
    _render_explainability(paths.bosch_output)

    excel_report_path = paths.bosch_output / "bosch_benchmark_report.xlsx"
    if excel_report_path.is_file():
        st.download_button(
            "📊 Download Multi-Tab Excel Benchmark Report (.xlsx)",
            data=excel_report_path.read_bytes(),
            file_name="bosch_benchmark_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    summary_csv_path = paths.bosch_output / "model_benchmark_summary.csv"
    if summary_csv_path.is_file():
        st.download_button(
            "Download Benchmark Summary CSV",
            data=summary_csv_path.read_text(encoding="utf-8"),
            file_name="model_benchmark_summary.csv",
            mime="text/csv",
        )

    st.download_button(
        "Download Bosch regression metrics",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="bosch_plasma_etch_metrics.csv",
        mime="text/csv",
    )
    if predictions is not None:
        st.download_button(
            "Download Bosch out-of-fold prediction ledger",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="bosch_plasma_etch_oof_predictions.csv",
            mime="text/csv",
        )
