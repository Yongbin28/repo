from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from waferpulse.tools.cmp_virtual_metrology import (
    CMP_TRACE_COLUMNS,
    load_verified_cmp_model,
    predict_cmp_traces,
    validate_cmp_trace_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_cmp_model_passed_strict_campaign_r2_gate() -> None:
    verified = load_verified_cmp_model()

    assert verified.metadata["r2_gate_passed"] is True
    assert verified.metadata["final_metrics"]["r2"] > 0.8
    assert verified.metadata["selected_model"] == "lightgbm"
    assert "WAFER_ID" not in verified.feature_columns
    assert "TIMESTAMP" not in verified.feature_columns
    assert all("source" not in column.lower() for column in verified.feature_columns)


def test_cmp_ledger_reproduces_untouched_score_without_wafer_overlap() -> None:
    verified = load_verified_cmp_model()
    ledger_path = ROOT / "output" / "phm2016_cmp" / "phm2016_cmp_prediction_ledger.csv"
    if not ledger_path.exists():
        pytest.skip("PHM16 benchmark evidence ledger is not present")
    ledger = pd.read_csv(ledger_path)
    development = ledger.loc[ledger["split"].eq("development_group_oof")]
    final = ledger.loc[ledger["split"].eq("untouched_latest_campaigns")]

    assert set(development["campaign"].unique()) == set(range(15))
    assert set(final["campaign"].unique()) == {15, 16, 17, 18}
    assert set(development["WAFER_ID"]).isdisjoint(set(final["WAFER_ID"]))
    assert len(final) == verified.metadata["final_samples"]

    expected = verified.metadata["final_metrics"]
    assert r2_score(final["actual"], final["prediction"]) == pytest.approx(
        expected["r2"], abs=1e-12
    )
    assert mean_squared_error(
        final["actual"], final["prediction"]
    ) ** 0.5 == pytest.approx(expected["rmse"], abs=1e-12)
    assert mean_absolute_error(
        final["actual"], final["prediction"]
    ) == pytest.approx(expected["mae"], abs=1e-12)


def test_cmp_prediction_uses_real_trace_rows_and_returns_finite_wafer_outputs() -> None:
    samples = sorted((ROOT / "data" / "phm2016_cmp").rglob("CMP-training-000.csv"))
    if not samples:
        pytest.skip("Local ignored PHM16 trace archive is not present")
    traces = pd.read_csv(samples[0], low_memory=False)
    traces["_source_index"] = 0

    result = predict_cmp_traces(traces)

    expected = traces[["WAFER_ID", "STAGE"]].drop_duplicates().shape[0]
    assert len(result.predictions) == expected
    assert np.isfinite(result.predictions["predicted_avg_removal_rate"]).all()
    assert result.input_summary["target_present_in_input"] is False
    assert result.model_metadata["final_metrics"]["r2"] > 0.8


def test_cmp_trace_contract_rejects_missing_upstream_signal() -> None:
    frame = pd.DataFrame(columns=CMP_TRACE_COLUMNS).drop(
        columns=["PRESSURIZED_CHAMBER_PRESSURE"]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        validate_cmp_trace_frame(frame)
