"""Verified PHM16 CMP virtual-metrology inference tools.

This production-facing module keeps the model scope explicit: it predicts
downstream average material-removal rate from upstream CMP process traces.  It
does not predict final electrical wafer test, yield, or field reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd


MODEL_FILENAME = "phm2016_cmp_verified_model.joblib"
KEY_COLUMNS = ["WAFER_ID", "STAGE"]
CMP_TRACE_COLUMNS = [
    "MACHINE_ID",
    "MACHINE_DATA",
    "TIMESTAMP",
    "WAFER_ID",
    "STAGE",
    "CHAMBER",
    "USAGE_OF_BACKING_FILM",
    "USAGE_OF_DRESSER",
    "USAGE_OF_POLISHING_TABLE",
    "USAGE_OF_DRESSER_TABLE",
    "PRESSURIZED_CHAMBER_PRESSURE",
    "MAIN_OUTER_AIR_BAG_PRESSURE",
    "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE",
    "RIPPLE_AIR_BAG_PRESSURE",
    "USAGE_OF_MEMBRANE",
    "USAGE_OF_PRESSURIZED_SHEET",
    "SLURRY_FLOW_LINE_A",
    "SLURRY_FLOW_LINE_B",
    "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION",
    "STAGE_ROTATION",
    "HEAD_ROTATION",
    "DRESSING_WATER_STATUS",
    "EDGE_AIR_BAG_PRESSURE",
]


@dataclass(frozen=True)
class VerifiedCMPModel:
    model: Any
    feature_columns: Tuple[str, ...]
    metadata: Mapping[str, Any]
    artifact_path: Path


@dataclass(frozen=True)
class CMPPredictionResult:
    predictions: pd.DataFrame
    model_metadata: Mapping[str, Any]
    input_summary: Mapping[str, Any]


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / MODEL_FILENAME


def load_verified_cmp_model(path: Path | None = None) -> VerifiedCMPModel:
    artifact_path = Path(path or default_model_path()).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Verified PHM16 CMP model is missing: {artifact_path}. "
            "Run waferpulse.experiments.phm2016_cmp_benchmark first."
        )
    bundle = joblib.load(artifact_path)
    required = {"model", "feature_columns", "metadata"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("CMP model artifact does not satisfy the verified bundle contract")
    metadata = bundle["metadata"]
    final_metrics = metadata.get("final_metrics", {})
    if not metadata.get("r2_gate_passed") or float(final_metrics.get("r2", -np.inf)) <= 0.8:
        raise ValueError("CMP model artifact has not passed the strict R² > 0.8 gate")
    feature_columns = tuple(str(column) for column in bundle["feature_columns"])
    if not feature_columns:
        raise ValueError("CMP model artifact has no feature schema")
    return VerifiedCMPModel(
        model=bundle["model"],
        feature_columns=feature_columns,
        metadata=metadata,
        artifact_path=artifact_path,
    )


def validate_cmp_trace_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CMP_TRACE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"CMP trace file is missing required columns: {missing}")
    result = frame[CMP_TRACE_COLUMNS + (["_source_index"] if "_source_index" in frame else [])].copy()
    result["STAGE"] = result["STAGE"].astype(str).str.strip().str.upper()
    invalid_stages = sorted(set(result["STAGE"].dropna()).difference({"A", "B"}))
    if invalid_stages:
        raise ValueError(f"CMP trace contains unsupported stages: {invalid_stages}")
    for column in result.columns.difference(["STAGE"]):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[KEY_COLUMNS + ["TIMESTAMP"]].isna().any().any():
        raise ValueError("CMP trace contains missing or non-numeric wafer/time identifiers")
    if "_source_index" not in result:
        result["_source_index"] = 0
    return result


def _aggregate_for_inference(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # The benchmark owns the exact frozen feature contract. Importing lazily
    # avoids loading experimental model dependencies at application startup.
    from waferpulse.experiments.phm2016_cmp_benchmark import _aggregate_features

    return _aggregate_features(frame)


def predict_cmp_traces(
    traces: pd.DataFrame,
    *,
    verified_model: VerifiedCMPModel | None = None,
    model_path: Path | None = None,
) -> CMPPredictionResult:
    verified = verified_model or load_verified_cmp_model(model_path)
    clean = validate_cmp_trace_frame(traces)
    features, identity = _aggregate_for_inference(clean)
    aligned = features.reindex(columns=verified.feature_columns)
    if aligned.isna().all(axis=0).any():
        missing_features = aligned.columns[aligned.isna().all(axis=0)].tolist()
        # Chamber/phase columns can be absent for one uploaded mini-batch and
        # are handled by the fold-trained median imputer. A completely absent
        # feature family is still recorded for operator review.
    else:
        missing_features = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        prediction = np.asarray(verified.model.predict(aligned)).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("Verified CMP model produced non-finite predictions")

    identity_aligned = (
        identity.set_index(KEY_COLUMNS).reindex(features.index).reset_index()
    )
    output = identity_aligned[KEY_COLUMNS].copy()
    output["predicted_avg_removal_rate"] = prediction
    output["prediction_scope"] = "CMP downstream material-removal-rate metrology"
    output["model_final_campaign_r2"] = float(verified.metadata["final_metrics"]["r2"])
    return CMPPredictionResult(
        predictions=output,
        model_metadata=verified.metadata,
        input_summary={
            "trace_rows": int(len(clean)),
            "wafer_stage_samples": int(len(output)),
            "physical_wafers": int(output["WAFER_ID"].nunique()),
            "stages": sorted(output["STAGE"].unique().tolist()),
            "all_missing_model_features": missing_features,
            "target_present_in_input": "AVG_REMOVAL_RATE" in traces.columns,
        },
    )


__all__ = [
    "CMPPredictionResult",
    "CMP_TRACE_COLUMNS",
    "MODEL_FILENAME",
    "VerifiedCMPModel",
    "default_model_path",
    "load_verified_cmp_model",
    "predict_cmp_traces",
    "validate_cmp_trace_frame",
]
