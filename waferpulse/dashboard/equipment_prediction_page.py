"""Streamlit page for the real open-source equipment prediction lane."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from waferpulse.config import WaferPulsePaths
from waferpulse.core.equipment_contracts import EquipmentModelResult
from waferpulse.core.data_lanes import REAL_OPEN_SOURCE
from waferpulse.crews import EquipmentPredictionCrew
from waferpulse.dashboard.components import render_data_lane_notice
from waferpulse.tools.equipment_pipeline import DEFAULT_MAX_INVALID_RATE, STAGE_LABELS

DATASET_STATE_KEY = "equipment_dataset"
MODEL_STATE_KEY = "equipment_model_result"
LOG_STATE_KEY = "equipment_pipeline_logs"


@dataclass(frozen=True)
class PageConfiguration:
    stage_mode: str
    max_invalid_rate: float
    folds: int
    selected_features: int
    estimators: int
    use_agentic_controller: bool
    max_agent_iterations: int
    groq_api_key: str


def _render_configuration(sidebar: Any) -> PageConfiguration:
    sidebar.markdown("---")
    sidebar.subheader("Real-Data Configuration")
    stage_display = sidebar.selectbox(
        "Equipment stages",
        options=list(STAGE_LABELS.values()),
        index=0,
        help="Use both stages for the report's 971-wafer fused dataset.",
    )
    stage_mode = next(key for key, value in STAGE_LABELS.items() if value == stage_display)
    invalid_percent = sidebar.slider(
        "Maximum source-formatted readings per sensor",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_MAX_INVALID_RATE * 100,
        step=0.25,
        format="%.2f%%",
        help=(
            "Month-like values such as 01.Mai are deterministically recovered and audited. "
            "Sensors above this source-format rate remain excluded from the reliable lane."
        ),
    )
    use_agentic_controller = sidebar.checkbox(
        "Groq agentic AutoML planner",
        value=False,
        help=(
            "Uses CrewAI with a bounded Groq planning step. Any missing dependency, "
            "invalid response, timeout, rate limit, or iteration-limit failure falls "
            "back to the deterministic balanced workflow."
        ),
    )
    max_agent_iterations = sidebar.slider(
        "Maximum agent iterations",
        min_value=1,
        max_value=8,
        value=8,
        disabled=not use_agentic_controller,
    )
    groq_api_key = sidebar.text_input(
        "Groq API key",
        type="password",
        value="",
        disabled=not use_agentic_controller,
        help="Leave blank to use the GROQ_API_KEY environment variable.",
    )
    return PageConfiguration(
        stage_mode=stage_mode,
        max_invalid_rate=invalid_percent / 100.0,
        folds=sidebar.slider("Lot-grouped validation folds", 3, 5, 3),
        selected_features=sidebar.slider("Selected temporal features", 40, 200, 60, 10),
        estimators=sidebar.slider("Trees / boosting iterations", 40, 200, 60, 20),
        use_agentic_controller=use_agentic_controller,
        max_agent_iterations=max_agent_iterations,
        groq_api_key=groq_api_key,
    )


def _render_actions(
    paths: WaferPulsePaths,
    crew: EquipmentPredictionCrew,
    config: PageConfiguration,
    has_saved_model: bool = False,
) -> None:
    if LOG_STATE_KEY not in st.session_state:
        st.session_state[LOG_STATE_KEY] = []

    btn_label = "⚡ Re-Run Multi-Agent Pipeline (Live Retraining)" if has_saved_model else "🚀 Run Multi-Agent Equipment Pipeline (Feature & Model Build)"
    run_clicked = st.button(
        btn_label,
        type="primary",
        use_container_width=True,
        help="Executes Agent 1 (EquipmentData validation & temporal features) followed by Agent 2 (GroupKFold model training & evidence generation)."
    )
    log_placeholder = st.empty()

    def ui_log(message: str) -> None:
        st.session_state[LOG_STATE_KEY].append(str(message))
        log_placeholder.code("\n".join(st.session_state[LOG_STATE_KEY][-20:]))

    if run_clicked:
        st.session_state[LOG_STATE_KEY] = []
        if not paths.equipment_data.exists():
            st.error(f"EquipmentData directory not found: {paths.equipment_data}")
            return

        try:
            with st.spinner("Running Multi-Agent Pipeline: Agent 1 (Feature Extraction) → Agent 2 (Model Validation)..."):
                result = crew.run(
                    data_dir=paths.equipment_data,
                    output_dir=paths.equipment_output,
                    stage_mode=config.stage_mode,
                    max_invalid_rate=config.max_invalid_rate,
                    n_splits=config.folds,
                    selected_features=config.selected_features,
                    n_estimators=config.estimators,
                    use_agentic_controller=config.use_agentic_controller,
                    groq_api_key=config.groq_api_key or None,
                    max_agent_iterations=config.max_agent_iterations,
                    log_func=ui_log,
                )
            st.session_state[MODEL_STATE_KEY] = result
            st.session_state.pop(DATASET_STATE_KEY, None)
            st.success("Multi-agent model validation completed and evidence artifacts were saved.")
        except Exception as exc:
            st.exception(exc)

    if st.session_state[LOG_STATE_KEY] and not run_clicked:
        with st.expander("Latest execution log", expanded=False):
            st.code("\n".join(st.session_state[LOG_STATE_KEY][-20:]))


def load_saved_equipment_result(output_dir: Path) -> Optional[EquipmentModelResult]:
    """Load display evidence without unpickling the persisted estimator bundle."""

    output_dir = Path(output_dir)
    paths = {
        "metrics": output_dir / "metrics.json",
        "fold_metrics": output_dir / "fold_metrics.csv",
        "predictions": output_dir / "prediction_ledger_oof.csv",
        "feature_importance": output_dir / "feature_importance.csv",
        "sensor_quality": output_dir / "sensor_quality.csv",
        "provenance": output_dir / "provenance.json",
        "quality_summary": output_dir / "quality_summary.json",
        "model_bundle": output_dir / "equipment_quality_models.joblib",
    }
    required = (
        "metrics",
        "fold_metrics",
        "predictions",
        "feature_importance",
        "sensor_quality",
        "provenance",
    )
    if any(not paths[name].is_file() for name in required):
        return None
    try:
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        quality_summary = (
            json.loads(paths["quality_summary"].read_text(encoding="utf-8"))
            if paths["quality_summary"].is_file()
            else {}
        )
        return EquipmentModelResult(
            metrics=metrics,
            fold_metrics=pd.read_csv(paths["fold_metrics"]),
            predictions=pd.read_csv(paths["predictions"]),
            feature_importance=pd.read_csv(paths["feature_importance"]),
            quality_summary=quality_summary,
            sensor_quality=pd.read_csv(paths["sensor_quality"]),
            provenance=provenance,
            artifact_paths={
                name: str(path.resolve()) for name, path in paths.items() if path.exists()
            },
        )
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return None


def _active_evidence(
    output_dir: Path,
) -> Tuple[
    Optional[Any],
    Optional[Any],
    Optional[Dict[str, Any]],
    Optional[pd.DataFrame],
    Optional[Dict[str, Any]],
]:
    dataset = st.session_state.get(DATASET_STATE_KEY)
    result = st.session_state.get(MODEL_STATE_KEY)
    if result is not None:
        return result, dataset, result.quality_summary, result.sensor_quality, result.provenance
    if dataset is not None:
        return result, dataset, dataset.quality_summary, dataset.sensor_quality, dataset.provenance
    result = load_saved_equipment_result(output_dir)
    if result is not None:
        return (
            result,
            None,
            result.quality_summary or None,
            result.sensor_quality,
            result.provenance,
        )
    return None, None, None, None, None


def _render_quality_gate(
    summary: Dict[str, Any],
    sensor_quality: pd.DataFrame,
    provenance: Dict[str, Any],
) -> None:
    st.subheader("1) Data Quality Gate")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Matched Wafers", f"{summary['matched_wafers']:,}")
    q2.metric("Manufacturing Lots", summary["matched_lots"])
    q3.metric("Good Wafers", summary["class_counts"].get("good", 0))
    q4.metric("Bad Wafers", summary["class_counts"].get("bad", 0))
    q5.metric("Temporal Features", f"{summary['feature_count']:,}")

    st.caption(
        f"Matched {summary['matched_wafers']:,} wafers across {summary['matched_lots']} manufacturing lots · "
        f"{summary['feature_count']:,} engineered temporal features across active sensors."
    )


def _render_feature_preview(dataset: Any) -> None:
    st.subheader("2) Temporal Feature Engineering")
    st.write(
        "Features are calculated independently for each equipment sensor: distribution statistics, "
        "range, first-to-last delta, slope, early/middle/late process means, and missingness."
    )
    preview = pd.concat([dataset.identity, dataset.features.iloc[:, :18]], axis=1)
    st.dataframe(preview.head(25), use_container_width=True, hide_index=True)


def _render_validation_metrics(result: Any) -> None:
    metrics = result.metrics
    regression = metrics["regression"]
    classification = metrics["classification"]
    st.subheader("2) Honest Lot-Grouped Model Validation")
    st.caption(
        f"Regression champion: **{metrics['regression_champion']}** · Classification champion: "
        f"**{metrics['classification_champion']}** · {metrics['validation']} ({metrics['n_splits']} folds)."
    )
    if metrics.get("regression_selection_metric"):
        st.caption(
            "Regression selection: highest nested lot-grouped out-of-fold R² across the "
            "nine report algorithms. Tree growth and pruning parameters are selected only "
            "inside each training fold."
        )
    r1, r2, r3, c1, c2, c3 = st.columns(6)
    r1.metric("MAE", f"{regression['mae']:.4f}")
    r2.metric("RMSE", f"{regression['rmse']:.4f}")
    r3.metric("R²", f"{regression['r2']:.3f}")
    c1.metric("PR-AUC", f"{classification['pr_auc']:.3f}")
    c2.metric("Bad Recall", f"{classification['recall_bad']:.1%}")
    c3.metric("False-Negative Rate", f"{classification['false_negative_rate']:.1%}")
    st.caption(
        f"The bad-risk decision threshold is {classification['decision_threshold']:.3f}; it is "
        "selected from out-of-fold predictions to target at least 80% bad-wafer recall."
    )

    plot_left, plot_right = st.columns(2)
    with plot_left:
        figure = px.scatter(
            result.predictions,
            x="response_actual",
            y="response_predicted_oof",
            color="class_actual",
            hover_data=["lot", "wafer", "fold"],
            title="Out-of-Fold Response: Actual vs Predicted",
            color_discrete_map={"good": "#2e7d32", "bad": "#c62828"},
        )
        minimum = min(
            result.predictions["response_actual"].min(),
            result.predictions["response_predicted_oof"].min(),
        )
        maximum = max(
            result.predictions["response_actual"].max(),
            result.predictions["response_predicted_oof"].max(),
        )
        figure.add_shape(
            type="line",
            x0=minimum,
            y0=minimum,
            x1=maximum,
            y1=maximum,
            line=dict(color="#666", dash="dash"),
        )
        st.plotly_chart(figure, use_container_width=True)
    with plot_right:
        confusion = pd.DataFrame(
            [
                [classification["tn"], classification["fp"]],
                [classification["fn"], classification["tp"]],
            ],
            index=["Actual good", "Actual bad"],
            columns=["Predicted good", "Predicted bad"],
        )
        figure = px.imshow(
            confusion,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Out-of-Fold Quality-Gate Confusion Matrix",
        )
        st.plotly_chart(figure, use_container_width=True)
    st.dataframe(result.fold_metrics, use_container_width=True, hide_index=True)


def _render_nine_algorithm_benchmark(output_dir: Path) -> None:
    summary_path = output_dir / "r2_engineered_nine_algorithm_summary.csv"
    if not summary_path.exists():
        return
    benchmark = pd.read_csv(summary_path)
    view = benchmark[["model", "parameters", "r2", "rmse", "mae"]].copy()
    view.columns = ["Algorithm", "Best parameters", "R²", "RMSE", "MAE"]
    st.markdown("#### Feature Engineering + Nine-Algorithm Benchmark")
    st.caption(
        "Each row is the best parameter setting for one algorithm on the same 686 temporal "
        "features and the same five unseen-lot folds. The production ensemble is evaluated "
        "separately above."
    )
    st.dataframe(
        view.style.format({"R²": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download nine-algorithm comparison",
        data=benchmark.to_csv(index=False).encode("utf-8"),
        file_name="r2_engineered_nine_algorithm_summary.csv",
        mime="text/csv",
    )


def _render_explanations(result: Any) -> None:
    st.subheader("3) Explainable Sensor and Process-Phase Drivers")
    task = st.radio(
        "Explanation target",
        ["classification", "regression"],
        horizontal=True,
        key="equipment_importance_task",
    )
    importance = (
        result.feature_importance[result.feature_importance["task"].eq(task)]
        .nlargest(20, "importance")
        .sort_values("importance")
    )
    figure = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {task.title()} Drivers",
    )
    st.plotly_chart(figure, use_container_width=True)
    methods = sorted(importance.get("importance_method", pd.Series(dtype=str)).dropna().unique())
    if methods == ["quality_class_filter_score"]:
        st.caption(
            "The hurdle regression champion is non-linear, so this view shows normalized training-only "
            "quality-class filter relevance—not causal effect or signed model coefficients."
        )
    else:
        st.caption(
            "Tree impurity importance identifies the equipment, sensor and process phase; it is an "
            "association measure and not a causal claim."
        )

    shap_svg = Path("output/equipment_quality/shap_beeswarm_summary.svg")
    shap_png = Path("output/equipment_quality/shap_beeswarm_summary.png")
    if shap_svg.exists() or shap_png.exists():
        st.markdown("#### SHAP Beeswarm Summary Plot")
        rendered = False
        if shap_svg.exists():
            try:
                svg_content = shap_svg.read_text(encoding="utf-8")
                st.image(
                    svg_content,
                    caption="SHAP Summary (Beeswarm) Plot: Distribution and direction of feature impact on predictions (red = high feature value, blue = low)",
                    use_container_width=True,
                )
                rendered = True
            except Exception:
                rendered = False
        if not rendered and shap_png.exists():
            st.image(
                str(shap_png),
                caption="SHAP Summary (Beeswarm) Plot: Distribution and direction of feature impact on predictions (red = high feature value, blue = low)",
                use_container_width=True,
            )


def _render_evidence(result: Any) -> None:
    st.subheader("4) Prediction Ledger and Evidence Files")
    st.dataframe(result.predictions, use_container_width=True, hide_index=True, height=360)
    st.download_button(
        "Download out-of-fold prediction ledger",
        data=result.predictions.to_csv(index=False).encode("utf-8"),
        file_name="equipment_prediction_ledger_oof.csv",
        mime="text/csv",
    )
    with st.expander("Saved evidence paths", expanded=False):
        st.json(result.artifact_paths)


def render_equipment_prediction_page(
    current_dir: Path,
    sidebar: Any,
    show_intro: bool = True,
) -> None:
    """Render the complete real-data page using the two-agent crew."""
    paths = WaferPulsePaths.from_root(current_dir)
    crew = EquipmentPredictionCrew()

    render_data_lane_notice(REAL_OPEN_SOURCE)
    if show_intro:
        st.header("Legacy EquipmentData Benchmark")
        st.caption(
            "Optional historical comparison: two frontend equipment stages → wafer-test "
            "response. Bosch plasma etch is the current documented project benchmark."
        )
    else:
        st.subheader("Two-Stage Multi-Chamber Sensor Telemetry Prediction")
        st.caption("equipment1.csv + equipment2.csv → continuous response and bad-wafer risk")
    config = _render_configuration(sidebar)
    result, dataset, summary, quality, provenance = _active_evidence(paths.equipment_output)
    has_saved_model = (result is not None)
    _render_actions(paths, crew, config, has_saved_model=has_saved_model)

    if has_saved_model and LOG_STATE_KEY not in st.session_state:
        st.caption("⚡ Showing pre-computed out-of-fold model evidence. Click the button above if you wish to re-train live.")
    elif summary is None and result is None:
        st.info(
            "Click **Run Multi-Agent Equipment Pipeline** above to execute Agent 1 (Feature Extraction) and Agent 2 (Model Validation)."
        )
        return
    if dataset is not None:
        _render_feature_preview(dataset)
    if result is not None:
        _render_validation_metrics(result)
        _render_nine_algorithm_benchmark(paths.equipment_output)
        _render_explanations(result)
        _render_evidence(result)
