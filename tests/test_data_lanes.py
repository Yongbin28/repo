from pathlib import Path

import pytest

from data_provenance import (
    BOSCH_PLASMA_ETCH,
    CMP_VIRTUAL_METROLOGY,
    LOCAL_DEMONSTRATION,
    REAL_OPEN_SOURCE,
    DataLaneViolation,
    assert_functions_allowed,
    build_provenance_record,
)


def test_real_lane_allows_prediction_but_rejects_spc() -> None:
    assert_functions_allowed(REAL_OPEN_SOURCE, ["feature_engineering", "regression"])
    with pytest.raises(DataLaneViolation):
        assert_functions_allowed(REAL_OPEN_SOURCE, ["spc"])


def test_local_lane_allows_monitoring_but_rejects_model_validation() -> None:
    assert_functions_allowed(LOCAL_DEMONSTRATION, ["spc", "psi", "gdbn"])
    with pytest.raises(DataLaneViolation):
        assert_functions_allowed(LOCAL_DEMONSTRATION, ["regression"])


def test_cmp_lane_allows_virtual_metrology_but_rejects_spc() -> None:
    assert_functions_allowed(
        CMP_VIRTUAL_METROLOGY,
        ["regression", "prospective_campaign_validation"],
    )
    with pytest.raises(DataLaneViolation):
        assert_functions_allowed(CMP_VIRTUAL_METROLOGY, ["spc"])


def test_bosch_lane_allows_grouped_spatial_metrology_but_rejects_spc() -> None:
    assert_functions_allowed(
        BOSCH_PLASMA_ETCH,
        ["regression", "grouped_validation", "spatial_virtual_metrology"],
    )
    with pytest.raises(DataLaneViolation):
        assert_functions_allowed(BOSCH_PLASMA_ETCH, ["spc"])


def test_provenance_records_lane_and_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    record = build_provenance_record(
        REAL_OPEN_SOURCE,
        "test",
        [source],
        ["data_quality"],
        "v1",
    )
    assert record["lane"]["key"] == REAL_OPEN_SOURCE
    assert record["is_synthetic"] is False
    assert record["source_files"] == [str(source.resolve())]
