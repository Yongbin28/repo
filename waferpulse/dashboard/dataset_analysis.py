"""Reusable, read-only dataset exploration for the Streamlit dashboard."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_numeric_dtype

from waferpulse.agents.groq_model_selection_agent import (
    DEFAULT_MAX_AGENT_ITERATIONS,
    choose_search_profile_from_summary,
)
from waferpulse.tools.local_automl import benchmark_local_regression

SUPPORTED_UPLOAD_TYPES = ("csv", "tsv", "xlsx", "parquet")
MAX_ANALYSIS_ROWS = 50_000
ZENODO_RECORD_URL = "https://zenodo.org/records/17122442"


def _csv_separator(data: bytes, name: str) -> str:
    if Path(name).suffix.lower() == ".tsv":
        return "\t"
    sample = data[:16_384]
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            text = sample.decode(encoding)
        except UnicodeDecodeError:
            continue
        try:
            return csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
        except csv.Error:
            counts = {delimiter: text.count(delimiter) for delimiter in (",", ";", "\t", "|")}
            return max(counts, key=counts.get)
    return ","


def read_table_bytes(name: str, data: bytes) -> pd.DataFrame:
    """Read a supported tabular upload without writing it to disk."""

    suffix = Path(name).suffix.lower()
    buffer = io.BytesIO(data)
    if suffix in {".csv", ".tsv"}:
        separator = _csv_separator(data, name)
        errors: list[Exception] = []
        for encoding in ("utf-8-sig", "cp1252", "latin1"):
            # 1. First attempt: standard fast read_csv
            try:
                buffer.seek(0)
                return pd.read_csv(
                    buffer,
                    sep=separator,
                    encoding=encoding,
                    low_memory=False,
                )
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                errors.append(exc)

            # 2. Second attempt: dynamic header detection for ATE DLog / semiconductor logs with header comments
            try:
                text = data[:65536].decode(encoding, errors="ignore")
                lines = text.splitlines()
                h_idx = None
                for i, line in enumerate(lines[:40]):
                    low = line.lower()
                    if ("device" in low and "bin" in low) or ("wafer" in low and "die" in low):
                        h_idx = i
                        break
                if h_idx is None and lines:
                    counts = [line.count(separator) for line in lines[:30]]
                    max_c = max(counts) if counts else 0
                    if max_c > 2:
                        for i, c in enumerate(counts):
                            if c >= max_c * 0.8:
                                tokens = [t.strip() for t in lines[i].split(separator) if t.strip()]
                                if len(tokens) >= 2:
                                    h_idx = i
                                    break
                buffer.seek(0)
                if h_idx is not None:
                    return pd.read_csv(
                        buffer,
                        sep=separator,
                        skiprows=h_idx,
                        encoding=encoding,
                        engine="python",
                        on_bad_lines="skip",
                    )
                return pd.read_csv(
                    buffer,
                    sep=separator,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as exc:
                errors.append(exc)

        raise ValueError(f"Could not parse {name}: {errors[-1]}")
    if suffix == ".xlsx":
        return pd.read_excel(buffer)
    if suffix == ".parquet":
        return pd.read_parquet(buffer)
    raise ValueError(
        f"Unsupported file type {suffix or '(none)'}. Choose CSV, TSV, XLSX, or Parquet."
    )


def dataset_profile(frame: pd.DataFrame) -> dict[str, int | float]:
    """Return stable high-level quality statistics for a dataframe."""

    rows, columns = frame.shape
    total_cells = rows * columns
    missing_cells = int(frame.isna().sum().sum())
    try:
        duplicate_rows = int(frame.duplicated().sum())
    except TypeError:
        duplicate_rows = 0
    return {
        "rows": int(rows),
        "columns": int(columns),
        "numeric_columns": int(sum(is_numeric_dtype(frame[column]) for column in frame.columns)),
        "missing_cells": missing_cells,
        "missing_rate": float(missing_cells / total_cells) if total_cells else 0.0,
        "duplicate_rows": duplicate_rows,
    }


def analyzable_numeric_frame(
    frame: pd.DataFrame,
    minimum_success_rate: float = 0.8,
) -> pd.DataFrame:
    """Coerce mostly-numeric columns for charts while leaving source data unchanged."""

    sample = frame.head(MAX_ANALYSIS_ROWS)
    numeric: dict[str, pd.Series] = {}
    for column in sample.columns:
        source = sample[column]
        if is_numeric_dtype(source):
            converted = pd.to_numeric(source, errors="coerce")
        else:
            non_missing = int(source.notna().sum())
            if non_missing == 0:
                continue
            cleaned = source.astype("string").str.strip().str.replace(",", ".", regex=False)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if float(converted.notna().sum() / non_missing) < minimum_success_rate:
                continue
        if converted.notna().sum() and converted.nunique(dropna=True) > 1:
            numeric[str(column)] = converted.astype(float)
    return pd.DataFrame(numeric, index=sample.index)


def _column_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        values = frame[column]
        example_values = values.dropna().astype(str).head(3).tolist()
        rows.append(
            {
                "column": str(column),
                "source_type": str(values.dtype),
                "non_null": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()) if len(values) else 0.0,
                "unique": int(values.nunique(dropna=True)),
                "examples": " | ".join(value[:60] for value in example_values),
            }
        )
    return pd.DataFrame(rows)


def render_dataframe_analysis(
    frame: pd.DataFrame,
    *,
    key_prefix: str,
    source_caption: str,
    default_target: str | None = None,
) -> None:
    """Render overview, quality, distribution, and optional target analysis."""

    profile = dataset_profile(frame)
    st.caption(source_caption)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", f"{profile['columns']:,}")
    c3.metric("Numeric", f"{profile['numeric_columns']:,}")
    c4.metric("Missing cells", f"{profile['missing_cells']:,}", f"{profile['missing_rate']:.2%}")
    c5.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")

    st.markdown("#### Data preview")
    st.dataframe(frame.head(200), use_container_width=True, hide_index=True, height=340)
    st.caption("The preview is limited to 200 rows; metrics describe the complete loaded table.")

    columns = _column_profile(frame)
    with st.expander("Column dictionary and missingness", expanded=False):
        st.dataframe(
            columns.style.format({"missing_rate": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        missing = columns.loc[columns["missing"].gt(0)].nlargest(20, "missing_rate")
        if not missing.empty:
            figure = px.bar(
                missing.sort_values("missing_rate"),
                x="missing_rate",
                y="column",
                orientation="h",
                title="Columns with the most missing data",
                labels={"missing_rate": "Missing rate", "column": "Column"},
            )
            figure.update_xaxes(tickformat=".0%")
            st.plotly_chart(figure, use_container_width=True)

    numeric = analyzable_numeric_frame(frame)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Distribution")
        if numeric.empty:
            st.info("No numeric or mostly-numeric columns are available for distribution charts.")
        else:
            selected = st.selectbox(
                "Numeric column",
                options=list(numeric.columns),
                key=f"{key_prefix}_distribution_column",
            )
            figure = px.histogram(
                numeric,
                x=selected,
                nbins=40,
                marginal="box",
                title=f"Distribution of {selected}",
            )
            st.plotly_chart(figure, use_container_width=True)
            selected_values = numeric[selected].dropna()
            st.caption(
                f"Valid values: {len(selected_values):,} · "
                f"median {selected_values.median():.4g} · "
                f"range {selected_values.min():.4g} to {selected_values.max():.4g}."
            )

    with right:
        st.markdown("#### Target analysis")
        target_options = ["No target selected", *[str(column) for column in frame.columns]]
        target_index = (
            target_options.index(default_target) if default_target in target_options else 0
        )
        target = st.selectbox(
            "Target or outcome column",
            options=target_options,
            index=target_index,
            key=f"{key_prefix}_target_column",
            help="Select an outcome to inspect its distribution and strongest numeric associations.",
        )
        if target == "No target selected":
            st.info("Choose a target to inspect class balance or numeric associations.")
        elif target in numeric.columns:
            target_values = numeric[target].dropna()
            st.plotly_chart(
                px.histogram(target_values, x=target, nbins=35, title=f"Target: {target}"),
                use_container_width=True,
            )
            correlations = (
                numeric.corrwith(numeric[target])
                .drop(labels=[target], errors="ignore")
                .dropna()
                .rename("correlation")
                .to_frame()
            )
            if not correlations.empty:
                correlations["absolute_correlation"] = correlations["correlation"].abs()
                correlations = correlations.nlargest(15, "absolute_correlation").reset_index(
                    names="feature"
                )
                st.dataframe(
                    correlations[["feature", "correlation"]].style.format(
                        {"correlation": "{:.3f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Correlation is exploratory association, not causal importance.")
        else:
            counts = (
                frame[target]
                .astype("string")
                .fillna("(missing)")
                .astype(str)
                .value_counts(dropna=False)
                .head(25)
                .rename_axis(target)
                .reset_index(name="rows")
            )
            st.plotly_chart(
                px.bar(counts, x=target, y="rows", title=f"Class/value balance: {target}"),
                use_container_width=True,
            )


@st.cache_data(show_spinner=False)
def _read_local_table(path_text: str, modified_ns: int, size: int) -> pd.DataFrame:
    del modified_ns, size
    path = Path(path_text)
    return read_table_bytes(path.name, path.read_bytes())


def read_local_table(path: Path) -> pd.DataFrame:
    stat = path.stat()
    return _read_local_table(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _render_inventory(rows: list[dict[str, Any]]) -> None:
    inventory = pd.DataFrame(rows)
    st.dataframe(inventory, use_container_width=True, hide_index=True)


def render_equipment_source_analysis(current_dir: Path) -> None:
    data_dir = current_dir / "EquipmentData"
    sources = {
        "Equipment stage 1 traces": (data_dir / "equipment1.csv", None),
        "Equipment stage 2 traces": (data_dir / "equipment2.csv", None),
        "Wafer response labels": (data_dir / "response.csv", "response"),
    }
    _render_inventory(
        [
            {
                "local file": path.name,
                "role": "model input" if "equipment" in path.name else "prediction target",
                "status": "Available" if path.is_file() else "Missing",
                "size (MB)": round(path.stat().st_size / 1_048_576, 2) if path.is_file() else None,
            }
            for path, _ in sources.values()
        ]
    )
    choice = st.selectbox(
        "Table to explore",
        options=list(sources),
        key="equipment_analysis_table",
    )
    path, default_target = sources[choice]
    if not path.is_file():
        st.error(f"Required local file is missing: {path}")
        return
    try:
        with st.spinner(f"Loading {path.name}..."):
            frame = read_local_table(path)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return
    render_dataframe_analysis(
        frame,
        key_prefix=f"equipment_{path.stem}",
        source_caption=f"Read-only local source: {path}",
        default_target=default_target,
    )

    if {"lot", "wafer", "timestamp"}.issubset(frame.columns):
        st.markdown("#### Wafer sensor trace")
        lot = st.selectbox(
            "Lot",
            options=sorted(frame["lot"].astype(str).unique()),
            key=f"equipment_{path.stem}_lot",
        )
        lot_frame = frame.loc[frame["lot"].astype(str).eq(lot)]
        wafer = st.selectbox(
            "Wafer",
            options=sorted(lot_frame["wafer"].astype(str).unique()),
            key=f"equipment_{path.stem}_wafer",
        )
        trace = lot_frame.loc[lot_frame["wafer"].astype(str).eq(wafer)].copy()
        sensor_columns = [column for column in trace.columns if str(column).startswith("sensor_")]
        selected_sensors = st.multiselect(
            "Sensors",
            options=sensor_columns,
            default=sensor_columns[: min(4, len(sensor_columns))],
            max_selections=8,
            key=f"equipment_{path.stem}_sensors",
        )
        if selected_sensors:
            chart = trace[["timestamp", *selected_sensors]].melt(
                id_vars="timestamp", var_name="sensor", value_name="reading"
            )
            chart["sample"] = pd.to_numeric(
                chart["timestamp"].astype(str).str.extract(r"(\d+)$", expand=False),
                errors="coerce",
            )
            chart["reading"] = pd.to_numeric(chart["reading"], errors="coerce")
            st.plotly_chart(
                px.line(
                    chart.sort_values("sample"),
                    x="sample",
                    y="reading",
                    color="sensor",
                    title=f"{path.stem}: {lot} / wafer {wafer}",
                ),
                use_container_width=True,
            )


def render_bosch_source_analysis(current_dir: Path) -> None:
    data_dir = (
        current_dir / "dataset" / "bosch_plasma_etch"
        if (current_dir / "dataset" / "bosch_plasma_etch").is_dir()
        else current_dir / "data" / "bosch_plasma_etch"
    )
    inventory_files = [
        ("Process_data.nc", "5 Hz process parameters", True),
        ("Dictionary_process.nc", "NetCDF decoding dictionary", True),
        ("Si_Oxide_etch_89_points.csv", "89-point wafer metrology", True),
        ("Lot_status.xlsx", "lot/conditioning metadata", False),
        ("bosch_process_wafer_features.parquet", "local derived process features", False),
    ]
    st.markdown(f"[Open the source dataset on Zenodo]({ZENODO_RECORD_URL})")
    _render_inventory(
        [
            {
                "local file": name,
                "role": role,
                "required for model": "Yes" if required else "No",
                "status": "Available" if (data_dir / name).is_file() else "Missing",
                "size (MB)": (
                    round((data_dir / name).stat().st_size / 1_048_576, 2)
                    if (data_dir / name).is_file()
                    else None
                ),
            }
            for name, role, required in inventory_files
        ]
    )
    choices = {
        "89-point wafer measurements": (
            data_dir / "Si_Oxide_etch_89_points.csv",
            "si_etch",
        ),
        "Engineered process features": (
            data_dir / "bosch_process_wafer_features.parquet",
            None,
        ),
        "Lot status and conditioning": (data_dir / "Lot_status.xlsx", None),
    }
    choice = st.selectbox(
        "Table to explore",
        options=list(choices),
        key="bosch_analysis_table",
    )
    path, default_target = choices[choice]
    if not path.is_file():
        st.error(f"This local Zenodo table is not available: {path}")
        return
    try:
        with st.spinner(f"Loading {path.name}..."):
            frame = read_local_table(path)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return
    render_dataframe_analysis(
        frame,
        key_prefix=f"bosch_{path.stem}",
        source_caption=f"Read-only local Zenodo file: {path}",
        default_target=default_target,
    )
    if path.name == "Si_Oxide_etch_89_points.csv" and {"X", "Y", "si_etch"}.issubset(frame):
        st.markdown("#### Wafer spatial map")
        experiment = st.selectbox(
            "Experiment / wafer",
            options=sorted(frame["experiment_key"].astype(str).unique()),
            key="bosch_spatial_experiment",
        )
        wafer = frame.loc[frame["experiment_key"].astype(str).eq(experiment)]
        st.plotly_chart(
            px.scatter(
                wafer,
                x="X",
                y="Y",
                color="si_etch",
                color_continuous_scale="Viridis",
                title=f"Measured silicon etch: {experiment}",
                labels={"si_etch": "Si etch"},
            ),
            use_container_width=True,
        )


def render_uploaded_dataset_analysis(sidebar: Any | None = None) -> None:
    st.info(
        "Upload one local table for exploratory analysis. It is not combined with either "
        "public benchmark and is not used to claim model accuracy."
    )
    uploaded = st.file_uploader(
        "Choose a local dataset",
        type=list(SUPPORTED_UPLOAD_TYPES),
        key="local_analysis_upload",
        help="Supported: CSV, TSV, XLSX, and Parquet.",
    )
    if uploaded is None:
        st.caption(
            "The analyzer will show a preview, data types, missingness, duplicates, "
            "distributions, and optional target associations."
        )
        return

    use_agentic_controller = False
    max_agent_iterations = DEFAULT_MAX_AGENT_ITERATIONS
    groq_api_key = ""
    if sidebar is not None:
        sidebar.markdown("---")
        sidebar.subheader("Local AutoML")
        use_agentic_controller = sidebar.checkbox(
            "Groq agentic AutoML planner",
            value=False,
            key="local_agentic_automl",
            help="Falls back to the balanced nine-model workflow on any agent failure.",
        )
        max_agent_iterations = sidebar.slider(
            "Maximum agent iterations",
            1,
            8,
            DEFAULT_MAX_AGENT_ITERATIONS,
            key="local_agent_iterations",
            disabled=not use_agentic_controller,
        )
        groq_api_key = sidebar.text_input(
            "Groq API key",
            type="password",
            key="local_groq_api_key",
            disabled=not use_agentic_controller,
            help="Leave blank to use GROQ_API_KEY.",
        )
    try:
        with st.spinner(f"Reading {uploaded.name}..."):
            frame = read_table_bytes(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read {uploaded.name}: {exc}")
        return
    st.success(f"Loaded {uploaded.name} successfully.")
    render_dataframe_analysis(
        frame,
        key_prefix="uploaded_local",
        source_caption=(
            f"Local in-memory upload: {uploaded.name}. The dashboard does not save this file."
        ),
    )

    st.markdown("### Optional Local Regression AutoML")
    numeric_targets = [
        str(column)
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 30
    ]
    if not numeric_targets:
        st.info("A local regression benchmark requires a numeric target with at least 30 values.")
        return
    target_column = st.selectbox(
        "Regression target",
        options=numeric_targets,
        key="local_automl_target",
    )
    group_choice = st.selectbox(
        "Validation group column",
        options=["No group column", *[str(column) for column in frame.columns if str(column) != target_column]],
        key="local_automl_group",
        help="Select lot, batch, wafer family, or another leakage boundary when available.",
    )
    if st.button("Run Local Nine-Model AutoML", type="primary", use_container_width=True):
        group_column = None if group_choice == "No group column" else group_choice
        group_count = int(frame[group_column].nunique()) if group_column else 0
        decision = choose_search_profile_from_summary(
            {
                "samples": int(len(frame)),
                "features": int(frame.shape[1] - 1),
                "manufacturing_lots": group_count,
                "task": f"local regression target {target_column}",
            },
            enabled=use_agentic_controller,
            api_key=groq_api_key or None,
            max_iterations=max_agent_iterations,
        )
        try:
            with st.spinner("Running local nine-model benchmark..."):
                metrics, predictions, metadata = benchmark_local_regression(
                    frame,
                    target_column=target_column,
                    group_column=group_column,
                    search_profile=decision.search_profile,
                )
            st.session_state["local_automl_result"] = (
                metrics,
                predictions,
                metadata,
                decision.as_dict(),
            )
        except Exception as exc:
            st.error(f"Local AutoML could not run: {exc}")
    result = st.session_state.get("local_automl_result")
    if result is not None:
        metrics, predictions, metadata, decision = result
        champion = metrics.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Champion", str(champion["model"]).replace("_", " ").title())
        c2.metric("Out-of-fold R²", f"{champion['r2']:.4f}")
        c3.metric("RMSE", f"{champion['rmse']:.4f}")
        st.caption(
            f"{metadata['validation']} · profile {metadata['search_profile']} · "
            f"agent status {decision['status']}. The uploaded file remains in memory only."
        )
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        champion_predictions = predictions.loc[
            predictions["model"].eq(champion["model"])
        ]
        st.plotly_chart(
            px.scatter(
                champion_predictions,
                x="actual",
                y="predicted",
                title="Champion out-of-fold predictions",
            ),
            use_container_width=True,
        )
