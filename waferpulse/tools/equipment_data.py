"""Data-quality and temporal feature tools for public EquipmentData."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from waferpulse.core.data_lanes import REAL_OPEN_SOURCE, build_provenance_record
from waferpulse.core.equipment_contracts import (
    DEFAULT_MAX_INVALID_RATE,
    DEFAULT_RESPONSE_THRESHOLD,
    EXPECTED_SAMPLES_PER_WAFER,
    PIPELINE_VERSION,
    STAGE_LABELS,
    EquipmentDataset,
    EquipmentTraceDataset,
)


_GERMAN_MONTH_NUMBER = {
    "jän": 1,
    "jan": 1,
    "feb": 2,
    "mär": 3,
    "mar": 3,
    "apr": 4,
    "mai": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "okt": 10,
    "nov": 11,
    "dez": 12,
}
_MONTH_TOKEN = "|".join(re.escape(month) for month in _GERMAN_MONTH_NUMBER)
_DAY_MONTH_PATTERN = re.compile(
    rf"^(?P<whole>\d{{1,2}})\.(?P<month>{_MONTH_TOKEN})$",
    flags=re.IGNORECASE,
)
_MONTH_FRACTION_PATTERN = re.compile(
    rf"^(?P<month>{_MONTH_TOKEN})\.(?P<fraction>\d{{1,2}})$",
    flags=re.IGNORECASE,
)

def _log(log_func: Optional[Callable[[str], None]], message: str) -> None:
    if log_func:
        log_func(message)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_semicolon_csv(path: Path) -> pd.DataFrame:
    # The source files contain German month abbreviations encoded as CP1252.
    return pd.read_csv(path, sep=";", dtype=str, encoding="cp1252", low_memory=False)


def _recover_spreadsheet_decimal(value: Any) -> float:
    """Recover decimals that spreadsheet software rendered as German dates.

    Examples from the published source are ``01.Sep`` -> ``1.09`` and
    ``Mär.92`` -> ``3.92``. Scientific notation with a decimal comma is also
    normalised. Unrecognised values remain NaN and are handled by the usual
    sensor-quality rules.
    """

    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan

    decimal_comma = text.replace(",", ".")
    try:
        return float(decimal_comma)
    except ValueError:
        pass

    day_month = _DAY_MONTH_PATTERN.fullmatch(text)
    if day_month:
        whole = int(day_month.group("whole"))
        month = _GERMAN_MONTH_NUMBER[day_month.group("month").lower()]
        return float(f"{whole}.{month:02d}")

    month_fraction = _MONTH_FRACTION_PATTERN.fullmatch(text)
    if month_fraction:
        whole = _GERMAN_MONTH_NUMBER[month_fraction.group("month").lower()]
        fraction = int(month_fraction.group("fraction"))
        return float(f"{whole}.{fraction:02d}")
    return np.nan


def _coerce_sensor_numeric(raw: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return numeric readings, recovered-format mask, and unresolved mask."""

    numeric = pd.to_numeric(raw, errors="coerce")
    source_formatted = raw.notna() & numeric.isna()
    if source_formatted.any():
        recovered_values = raw.loc[source_formatted].map(_recover_spreadsheet_decimal)
        numeric.loc[source_formatted] = recovered_values
    recovered = source_formatted & numeric.notna()
    unresolved = raw.notna() & numeric.isna()
    return numeric.astype(float), recovered, unresolved


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["lot"] = result["lot"].astype(str).str.strip()
    result["wafer"] = result["wafer"].astype(str).str.strip()
    return result


def _deduplicate_response(
    response: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = {"lot", "wafer", "response", "class"}
    missing = required.difference(response.columns)
    if missing:
        raise ValueError(f"response.csv is missing columns: {sorted(missing)}")

    response = _normalise_keys(response)
    response["response"] = pd.to_numeric(response["response"], errors="coerce")
    response["class"] = response["class"].astype(str).str.strip().str.lower()
    if response["response"].isna().any():
        raise ValueError("response.csv contains missing or non-numeric response values")

    conflict_counts = response.groupby(["lot", "wafer"])[["response", "class"]].nunique(dropna=False)
    conflicts = conflict_counts[(conflict_counts["response"] > 1) | (conflict_counts["class"] > 1)]
    if not conflicts.empty:
        example = conflicts.index[0]
        raise ValueError(f"Conflicting duplicate response labels found for lot/wafer {example}")

    exact_duplicate_rows = int(response.duplicated().sum())
    duplicate_key_rows = int(response.duplicated(["lot", "wafer"], keep=False).sum())
    response = response.drop_duplicates(["lot", "wafer"], keep="first").reset_index(drop=True)

    expected_class = np.where(response["response"] > threshold, "bad", "good")
    mismatches = int((response["class"].to_numpy() != expected_class).sum())
    if mismatches:
        raise ValueError(
            f"{mismatches} response labels disagree with the configured threshold {threshold:.3f}"
        )

    return response, {
        "raw_response_rows": int(len(response) + exact_duplicate_rows),
        "unique_response_wafers": int(len(response)),
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "duplicate_key_rows_before_cleanup": duplicate_key_rows,
        "conflicting_duplicate_keys": 0,
        "threshold_label_mismatches": mismatches,
    }


def _timestamp_index(timestamp: pd.Series) -> pd.Series:
    values = timestamp.astype(str).str.extract(r"(\d+)$", expand=False)
    return pd.to_numeric(values, errors="coerce")


def _inspect_equipment(
    frame: pd.DataFrame,
    equipment: str,
    max_invalid_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    required = {"lot", "wafer", "timestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{equipment}.csv is missing columns: {sorted(missing)}")

    frame = _normalise_keys(frame)
    sensor_cols = [column for column in frame.columns if str(column).startswith("sensor_")]
    if not sensor_cols:
        raise ValueError(f"{equipment}.csv contains no sensor columns")

    timestamp_number = _timestamp_index(frame["timestamp"])
    if timestamp_number.isna().any():
        raise ValueError(f"{equipment}.csv contains unparseable timestamp values")
    frame = frame.assign(_timestamp_number=timestamp_number.astype(int))

    duplicate_timestamps = int(frame.duplicated(["lot", "wafer", "_timestamp_number"]).sum())
    if duplicate_timestamps:
        raise ValueError(f"{equipment}.csv contains {duplicate_timestamps} duplicate wafer timestamps")

    sequence_counts = frame.groupby(["lot", "wafer"], sort=False).size()
    incomplete = sequence_counts[sequence_counts != EXPECTED_SAMPLES_PER_WAFER]
    if not incomplete.empty:
        raise ValueError(
            f"{equipment}.csv contains {len(incomplete)} incomplete wafer sequences; "
            f"expected {EXPECTED_SAMPLES_PER_WAFER} observations per wafer"
        )

    numeric = frame[["lot", "wafer", "_timestamp_number"]].copy()
    quality_rows: List[Dict[str, Any]] = []
    retained: List[str] = []
    for sensor in sensor_cols:
        raw = frame[sensor]
        converted, recovered, unresolved = _coerce_sensor_numeric(raw)
        source_formatted = recovered | unresolved
        source_formatted_rate = float(source_formatted.mean())
        invalid_rate = float(unresolved.mean())
        examples = raw[unresolved].value_counts().head(5).index.astype(str).tolist()
        source_format_examples = (
            raw[source_formatted].value_counts().head(5).index.astype(str).tolist()
        )
        # A deterministic conversion recovers individual readings, but a sensor
        # dominated by spreadsheet-formatted cells remains outside the reliable
        # feature lane. This avoids making a high-confidence model claim from a
        # heavily transformed source column while retaining low-rate recoveries.
        dropped = source_formatted_rate > max_invalid_rate or converted.notna().sum() == 0
        quality_rows.append(
            {
                "equipment": equipment,
                "sensor": sensor,
                "rows": int(len(frame)),
                "source_formatted_count": int((recovered | unresolved).sum()),
                "source_formatted_rate": source_formatted_rate,
                "recovered_decimal_count": int(recovered.sum()),
                "non_numeric_count": int(unresolved.sum()),
                "non_numeric_rate": invalid_rate,
                "missing_count": int(raw.isna().sum()),
                "numeric_unique": int(converted.nunique(dropna=True)),
                "status": "DROPPED" if dropped else (
                    "IMPUTE_IN_MODEL" if unresolved.any() else (
                        "RECOVERED_SOURCE_FORMAT" if recovered.any() else "VALID"
                    )
                ),
                "examples": " | ".join(examples),
                "source_format_examples": " | ".join(source_format_examples),
                "exclusion_reason": (
                    "HIGH_SOURCE_FORMAT_RATE" if source_formatted_rate > max_invalid_rate else ""
                ),
            }
        )
        if not dropped:
            retained.append(sensor)
            numeric[sensor] = converted.astype(float)

    if not retained:
        raise ValueError(f"No reliable sensors remain in {equipment}.csv")

    quality = pd.DataFrame(quality_rows)
    summary = {
        "rows": int(len(frame)),
        "lots": int(frame["lot"].nunique()),
        "wafers": int(frame[["lot", "wafer"]].drop_duplicates().shape[0]),
        "samples_per_wafer": EXPECTED_SAMPLES_PER_WAFER,
        "sensor_count": int(len(sensor_cols)),
        "retained_sensor_count": int(len(retained)),
        "dropped_sensor_count": int(len(sensor_cols) - len(retained)),
        "dropped_sensors": quality.loc[quality["status"].eq("DROPPED"), "sensor"].tolist(),
        "duplicate_timestamps": duplicate_timestamps,
        "incomplete_sequences": int(len(incomplete)),
    }
    return numeric, quality, summary


def _aggregate_temporal_features(frame: pd.DataFrame, equipment: str) -> pd.DataFrame:
    sensor_cols = [column for column in frame.columns if str(column).startswith("sensor_")]
    ordered = frame.sort_values(["lot", "wafer", "_timestamp_number"]).copy()
    groups = ordered.groupby(["lot", "wafer"], sort=False)

    feature_frames: List[pd.DataFrame] = []
    for statistic in ("mean", "std", "min", "max", "median"):
        values = getattr(groups[sensor_cols], statistic)()
        values.columns = [f"{equipment}__{sensor}__{statistic}" for sensor in sensor_cols]
        feature_frames.append(values)

    for statistic, quantile in (("q25", 0.25), ("q75", 0.75)):
        values = groups[sensor_cols].quantile(quantile)
        values.columns = [f"{equipment}__{sensor}__{statistic}" for sensor in sensor_cols]
        feature_frames.append(values)

    minimum = groups[sensor_cols].min()
    maximum = groups[sensor_cols].max()
    ranges = maximum - minimum
    ranges.columns = [f"{equipment}__{sensor}__range" for sensor in sensor_cols]
    feature_frames.append(ranges)

    first = groups[sensor_cols].first()
    last = groups[sensor_cols].last()
    delta = last - first
    delta.columns = [f"{equipment}__{sensor}__delta" for sensor in sensor_cols]
    feature_frames.append(delta)

    slope = (last - first) / float(EXPECTED_SAMPLES_PER_WAFER - 1)
    slope.columns = [f"{equipment}__{sensor}__slope" for sensor in sensor_cols]
    feature_frames.append(slope)

    phase = pd.cut(
        ordered["_timestamp_number"],
        bins=[-np.inf, 58, 116, np.inf],
        labels=["early", "middle", "late"],
    )
    phased = (
        ordered.assign(_phase=phase)
        .groupby(["lot", "wafer", "_phase"], observed=True)[sensor_cols]
        .mean()
        .unstack("_phase")
    )
    phased.columns = [f"{equipment}__{sensor}__{phase_name}_mean" for sensor, phase_name in phased.columns]
    feature_frames.append(phased)

    missing_rate = groups[sensor_cols].apply(lambda part: part.isna().mean())
    missing_rate.columns = [f"{equipment}__{sensor}__missing_rate" for sensor in sensor_cols]
    feature_frames.append(missing_rate)

    features = pd.concat(feature_frames, axis=1)
    return features.replace([np.inf, -np.inf], np.nan).reset_index()


def _trace_tensor(
    frame: pd.DataFrame,
    identity: pd.DataFrame,
    equipment: str,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Align one equipment stage to a validated wafer-by-sensor trace tensor."""

    sensor_cols = [column for column in frame.columns if str(column).startswith("sensor_")]
    wafer_keys = np.empty(len(identity), dtype=object)
    wafer_keys[:] = list(identity[["lot", "wafer"]].itertuples(index=False, name=None))
    expected_index = pd.MultiIndex.from_arrays(
        [
            np.repeat(wafer_keys, EXPECTED_SAMPLES_PER_WAFER),
            np.tile(np.arange(EXPECTED_SAMPLES_PER_WAFER), len(identity)),
        ],
        names=["wafer_key", "_timestamp_number"],
    )
    indexed = frame.copy()
    indexed["wafer_key"] = list(zip(indexed["lot"], indexed["wafer"]))
    aligned = indexed.set_index(["wafer_key", "_timestamp_number"])[sensor_cols].reindex(
        expected_index
    )
    values = aligned.to_numpy(dtype=np.float32).reshape(
        len(identity), EXPECTED_SAMPLES_PER_WAFER, len(sensor_cols)
    )
    names = tuple(f"{equipment}__{sensor}" for sensor in sensor_cols)
    return values.transpose(0, 2, 1), names


def load_equipment_trace_dataset(
    data_dir: Path,
    stage_mode: str = "both",
    max_invalid_rate: float = DEFAULT_MAX_INVALID_RATE,
    response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
    log_func: Optional[Callable[[str], None]] = None,
) -> EquipmentTraceDataset:
    """Load reliable raw traces for fold-trained representation learning.

    This shares exactly the same decoding, sensor exclusion, duplicate-label
    checks, and wafer matching rules as :func:`load_equipment_dataset`. Missing
    readings are intentionally retained as NaN so each model fold can learn
    imputation and standardisation parameters from training lots only.
    """

    if stage_mode not in STAGE_LABELS:
        raise ValueError(f"Unknown stage mode {stage_mode!r}; choose from {sorted(STAGE_LABELS)}")
    if not 0 <= max_invalid_rate <= 1:
        raise ValueError("max_invalid_rate must be between 0 and 1")

    data_dir = Path(data_dir)
    paths = {
        "equipment1": data_dir / "equipment1.csv",
        "equipment2": data_dir / "equipment2.csv",
        "response": data_dir / "response.csv",
    }
    required_equipment = ("equipment1", "equipment2") if stage_mode == "both" else (stage_mode,)
    required_paths = [paths["response"], *(paths[name] for name in required_equipment)]
    missing_files = [str(path) for path in required_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"EquipmentData files are missing: {missing_files}")

    response, response_summary = _deduplicate_response(
        _read_semicolon_csv(paths["response"]), response_threshold
    )
    numeric_frames: Dict[str, pd.DataFrame] = {}
    quality_tables: List[pd.DataFrame] = []
    equipment_summaries: Dict[str, Any] = {}
    matched_keys = response[["lot", "wafer"]].copy()

    for equipment in required_equipment:
        _log(log_func, f"Validating raw traces from {equipment}.csv...")
        numeric, quality, summary = _inspect_equipment(
            _read_semicolon_csv(paths[equipment]), equipment, max_invalid_rate
        )
        numeric_frames[equipment] = numeric
        quality_tables.append(quality)
        equipment_summaries[equipment] = summary
        keys = numeric[["lot", "wafer"]].drop_duplicates()
        matched_keys = matched_keys.merge(keys, on=["lot", "wafer"], how="inner")

    combined = matched_keys.merge(
        response, on=["lot", "wafer"], how="inner", validate="one_to_one"
    )
    if combined.empty:
        raise ValueError("No matched equipment/response wafers were found")

    identity = combined[["lot", "wafer"]].reset_index(drop=True)
    tensors: List[np.ndarray] = []
    sensor_names: List[str] = []
    for equipment in required_equipment:
        tensor, names = _trace_tensor(numeric_frames[equipment], identity, equipment)
        tensors.append(tensor)
        sensor_names.extend(names)

    traces = np.concatenate(tensors, axis=1)
    if traces.shape[2] != EXPECTED_SAMPLES_PER_WAFER:
        raise ValueError(f"Trace tensor has unexpected shape {traces.shape}")

    bad = combined["class"].eq("bad").astype(int).reset_index(drop=True)
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "stage_mode": stage_mode,
        "stage_label": STAGE_LABELS[stage_mode],
        "max_invalid_rate": float(max_invalid_rate),
        "response_threshold": float(response_threshold),
        "equipment": equipment_summaries,
        "response": response_summary,
        "matched_wafers": int(len(combined)),
        "matched_lots": int(combined["lot"].nunique()),
        "retained_sensor_count": int(traces.shape[1]),
        "samples_per_wafer": int(traces.shape[2]),
        "class_counts": {
            str(key): int(value) for key, value in combined["class"].value_counts().items()
        },
        "bad_rate": float(bad.mean()),
    }
    _log(
        log_func,
        f"Prepared raw trace tensor {traces.shape} from {summary['matched_lots']} lots.",
    )
    return EquipmentTraceDataset(
        traces=traces,
        sensor_names=tuple(sensor_names),
        response=combined["response"].astype(float).reset_index(drop=True),
        bad_label=bad,
        groups=combined["lot"].astype(str).reset_index(drop=True),
        identity=identity,
        sensor_quality=pd.concat(quality_tables, ignore_index=True),
        quality_summary=summary,
    )


def load_equipment_dataset(
    data_dir: Path,
    stage_mode: str = "both",
    max_invalid_rate: float = DEFAULT_MAX_INVALID_RATE,
    response_threshold: float = DEFAULT_RESPONSE_THRESHOLD,
    log_func: Optional[Callable[[str], None]] = None,
) -> EquipmentDataset:
    """Load, validate, and feature-engineer the public EquipmentData files."""

    if stage_mode not in STAGE_LABELS:
        raise ValueError(f"Unknown stage mode {stage_mode!r}; choose from {sorted(STAGE_LABELS)}")
    if not 0 <= max_invalid_rate <= 1:
        raise ValueError("max_invalid_rate must be between 0 and 1")

    data_dir = Path(data_dir)
    paths = {
        "equipment1": data_dir / "equipment1.csv",
        "equipment2": data_dir / "equipment2.csv",
        "response": data_dir / "response.csv",
    }
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"EquipmentData files are missing: {missing_files}")

    _log(log_func, "Reading public EquipmentData files with CP1252-safe parsing...")
    response_raw = _read_semicolon_csv(paths["response"])
    response, response_summary = _deduplicate_response(response_raw, response_threshold)

    equipment_frames: Dict[str, pd.DataFrame] = {}
    quality_tables: List[pd.DataFrame] = []
    equipment_summaries: Dict[str, Any] = {}
    required_equipment = ("equipment1", "equipment2") if stage_mode == "both" else (stage_mode,)

    for equipment in required_equipment:
        _log(log_func, f"Validating {equipment}.csv and identifying unreliable sensors...")
        raw = _read_semicolon_csv(paths[equipment])
        numeric, quality, summary = _inspect_equipment(raw, equipment, max_invalid_rate)
        equipment_frames[equipment] = _aggregate_temporal_features(numeric, equipment)
        quality_tables.append(quality)
        equipment_summaries[equipment] = summary

    if stage_mode == "both":
        features = equipment_frames["equipment1"].merge(
            equipment_frames["equipment2"],
            on=["lot", "wafer"],
            how="inner",
            validate="one_to_one",
        )
    else:
        features = equipment_frames[stage_mode]

    combined = features.merge(response, on=["lot", "wafer"], how="inner", validate="one_to_one")
    if combined.empty:
        raise ValueError("No matched equipment/response wafers were found")

    identity = combined[["lot", "wafer"]].reset_index(drop=True)
    feature_cols = [
        column
        for column in features.columns
        if column not in {"lot", "wafer"}
    ]
    X = combined[feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y = combined["response"].astype(float).reset_index(drop=True)
    bad = combined["class"].eq("bad").astype(int).reset_index(drop=True)
    groups = combined["lot"].astype(str).reset_index(drop=True)

    class_counts = combined["class"].value_counts().to_dict()
    lot_class = pd.crosstab(combined["lot"], combined["class"])
    single_class_lots = int((lot_class.gt(0).sum(axis=1) == 1).sum())
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "stage_mode": stage_mode,
        "stage_label": STAGE_LABELS[stage_mode],
        "max_invalid_rate": float(max_invalid_rate),
        "response_threshold": float(response_threshold),
        "equipment": equipment_summaries,
        "response": response_summary,
        "matched_wafers": int(len(combined)),
        "matched_lots": int(combined["lot"].nunique()),
        "feature_count": int(X.shape[1]),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "single_class_lots": single_class_lots,
        "bad_rate": float(bad.mean()),
        "data_quality_status": "PASS_WITH_EXCLUSIONS" if any(
            summary["dropped_sensor_count"] for summary in equipment_summaries.values()
        ) else "PASS",
    }

    source_files = [paths["response"]] + [paths[key] for key in required_equipment]
    hashes = {path.name: _sha256(path) for path in source_files}
    dataset_version = hashlib.sha256(
        "|".join(f"{name}:{digest}" for name, digest in sorted(hashes.items())).encode("utf-8")
    ).hexdigest()[:16]
    provenance = build_provenance_record(
        lane_key=REAL_OPEN_SOURCE,
        dataset_name="Semiconductor Frontend Equipment Sensor Data",
        source_files=source_files,
        functions=(
            "data_quality",
            "feature_engineering",
            "regression",
            "classification",
            "explainability",
            "grouped_validation",
        ),
        dataset_version=dataset_version,
    )
    provenance["source_sha256"] = hashes
    provenance["response_semantics"] = {
        "continuous_target": "wafer response",
        "bad_when": f"response > {response_threshold}",
        "claim_limit": "Upstream equipment sensors to wafer-test response; not direct TP-to-FT validation.",
    }

    _log(
        log_func,
        f"Prepared {len(combined)} wafers from {combined['lot'].nunique()} lots with "
        f"{X.shape[1]} reliable temporal features.",
    )
    return EquipmentDataset(
        features=X,
        response=y,
        bad_label=bad,
        groups=groups,
        identity=identity,
        sensor_quality=pd.concat(quality_tables, ignore_index=True),
        quality_summary=summary,
        provenance=provenance,
    )
