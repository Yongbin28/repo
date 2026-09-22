import json
from pathlib import Path

import pandas as pd

from waferpulse.dashboard.dataset_analysis import (
    analyzable_numeric_frame,
    dataset_profile,
    read_table_bytes,
)
from waferpulse.dashboard.equipment_prediction_page import load_saved_equipment_result


def test_read_table_bytes_detects_semicolon_and_cp1252() -> None:
    source = "lot;sensor;label\nlot1;1,25;März\n".encode("cp1252")

    frame = read_table_bytes("equipment.csv", source)

    assert frame.to_dict("records") == [{"lot": "lot1", "sensor": "1,25", "label": "März"}]


def test_profile_and_numeric_coercion_keep_source_unchanged() -> None:
    frame = pd.DataFrame(
        {
            "numeric_text": ["1.0", "2.0", None, "4.0"],
            "category": ["a", "b", "b", "b"],
            "native": [1, 2, 3, 4],
        }
    )

    profile = dataset_profile(frame)
    numeric = analyzable_numeric_frame(frame)

    assert profile["rows"] == 4
    assert profile["columns"] == 3
    assert profile["missing_cells"] == 1
    assert set(numeric.columns) == {"numeric_text", "native"}
    assert frame["numeric_text"].dtype == object


def test_load_saved_equipment_result_uses_display_files_only(tmp_path: Path) -> None:
    metrics = {
        "regression": {"mae": 0.1, "rmse": 0.2, "r2": 0.3},
        "classification": {
            "pr_auc": 0.4,
            "recall_bad": 0.8,
            "false_negative_rate": 0.2,
        },
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "provenance.json").write_text(
        json.dumps({"dataset_name": "fixture"}), encoding="utf-8"
    )
    (tmp_path / "quality_summary.json").write_text(
        json.dumps({"matched_wafers": 2}), encoding="utf-8"
    )
    pd.DataFrame({"fold": [1]}).to_csv(tmp_path / "fold_metrics.csv", index=False)
    pd.DataFrame({"lot": ["lot1"]}).to_csv(tmp_path / "prediction_ledger_oof.csv", index=False)
    pd.DataFrame({"task": ["regression"], "feature": ["x"], "importance": [1.0]}).to_csv(
        tmp_path / "feature_importance.csv", index=False
    )
    pd.DataFrame({"sensor": ["sensor_1"], "status": ["VALID"]}).to_csv(
        tmp_path / "sensor_quality.csv", index=False
    )

    result = load_saved_equipment_result(tmp_path)

    assert result is not None
    assert result.metrics["regression"]["r2"] == 0.3
    assert result.quality_summary["matched_wafers"] == 2
    assert "model_bundle" not in result.artifact_paths


def test_target_analysis_handles_nullable_and_categorical_missing_values(monkeypatch):
    from unittest.mock import MagicMock
    from waferpulse.dashboard import dataset_analysis as analysis

    for values in (
        pd.Series([1, None, 1], dtype="Int64"),
        pd.Series(pd.Categorical(["pass", None, "pass"])),
    ):
        ui = MagicMock()
        ui.columns.side_effect = lambda count: [MagicMock() for _ in range(count)]
        ui.selectbox.return_value = "target"
        monkeypatch.setattr(analysis, "st", ui)
        frame = pd.DataFrame({"target": values})
        original = frame.copy(deep=True)

        analysis.render_dataframe_analysis(
            frame, key_prefix="test", source_caption="fixture", default_target="target"
        )

        chart = ui.plotly_chart.call_args.args[0]
        assert list(chart.data[0].y) == [2, 1]
        assert "(missing)" in chart.data[0].x
        pd.testing.assert_frame_equal(frame, original)


def test_read_table_bytes_handles_ate_dlog_metadata_header() -> None:
    source = (
        "Created from DLOG: sample.std_1.csv\n"
        "Test Program: PROG_V1\n"
        "Date: 01/01/2025\n"
        ",,,,,T1.0,T2.0\n"
        ",,,,,V,uA\n"
        "Device #,Bin,Site,X,Y,VDD,IDD\n"
        "1,1,1,10,20,1.2,0.05\n"
        "2,1,1,10,21,1.19,0.04\n"
    ).encode("utf-8")

    frame = read_table_bytes("sample.std_1.csv", source)

    assert frame.shape == (2, 7)
    assert list(frame.columns) == ["Device #", "Bin", "Site", "X", "Y", "VDD", "IDD"]
    assert frame["Device #"].tolist() == [1, 2]

