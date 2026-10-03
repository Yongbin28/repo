from pathlib import Path

import numpy as np
import pandas as pd

from equipment_data_pipeline import (
    EXPECTED_SAMPLES_PER_WAFER,
    _choose_recall_guardrail_threshold,
    load_equipment_dataset,
)
from waferpulse.tools.equipment_modeling import (
    QualityHurdleForestBlendRegressor,
    _make_regression_candidates,
    _regression_parameter_grids,
)
from waferpulse.tools.equipment_data import load_equipment_trace_dataset


def _write_fixture(data_dir: Path) -> None:
    rows_1 = []
    rows_2 = []
    responses = []
    for lot_number in range(1, 4):
        lot = f"lot{lot_number}"
        for wafer in ("1", "2"):
            response = 0.4 if wafer == "1" else 0.9
            class_name = "good" if response <= 0.75 else "bad"
            # The response source intentionally contains exact duplicates.
            responses.extend(
                [
                    {"lot": lot, "wafer": wafer, "response": response, "class": class_name},
                    {"lot": lot, "wafer": wafer, "response": response, "class": class_name},
                ]
            )
            for timestamp in range(EXPECTED_SAMPLES_PER_WAFER):
                corrupted = "01.Mai" if timestamp == 0 else str(timestamp / 100)
                rows_1.append(
                    {
                        "lot": lot,
                        "wafer": wafer,
                        "timestamp": f"timestamp_{timestamp}",
                        "sensor_1": str(response + timestamp / 1000),
                        "sensor_2": corrupted,
                    }
                )
                rows_2.append(
                    {
                        "lot": lot,
                        "wafer": wafer,
                        "timestamp": f"timestamp_{timestamp}",
                        "sensor_25": str(response - timestamp / 2000),
                    }
                )

    pd.DataFrame(rows_1).to_csv(
        data_dir / "equipment1.csv", sep=";", index=False, encoding="cp1252"
    )
    pd.DataFrame(rows_2).to_csv(
        data_dir / "equipment2.csv", sep=";", index=False, encoding="cp1252"
    )
    pd.DataFrame(responses).to_csv(
        data_dir / "response.csv", sep=";", index=False, encoding="cp1252"
    )


def test_load_equipment_dataset_deduplicates_and_recovers_spreadsheet_decimal(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    dataset = load_equipment_dataset(tmp_path, stage_mode="both", max_invalid_rate=0.01)

    assert len(dataset.features) == 6
    assert dataset.quality_summary["response"]["exact_duplicate_rows_removed"] == 6
    assert dataset.quality_summary["class_counts"] == {"good": 3, "bad": 3}
    assert "sensor_2" not in dataset.quality_summary["equipment"]["equipment1"]["dropped_sensors"]
    assert dataset.features.columns.str.startswith(("equipment1__", "equipment2__")).all()
    assert any("__sensor_2__" in column for column in dataset.features.columns)
    sensor_2_quality = dataset.sensor_quality.loc[dataset.sensor_quality["sensor"].eq("sensor_2")].iloc[0]
    assert sensor_2_quality["status"] == "RECOVERED_SOURCE_FORMAT"
    assert sensor_2_quality["recovered_decimal_count"] == 6


def test_load_equipment_trace_dataset_preserves_wafer_sensor_time_shape(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    dataset = load_equipment_trace_dataset(
        tmp_path,
        stage_mode="both",
        max_invalid_rate=0.01,
    )

    assert dataset.traces.shape == (6, 3, EXPECTED_SAMPLES_PER_WAFER)
    assert dataset.sensor_names == (
        "equipment1__sensor_1",
        "equipment1__sensor_2",
        "equipment2__sensor_25",
    )
    assert dataset.identity.to_dict("records")[0] == {"lot": "lot1", "wafer": "1"}
    assert np.isclose(dataset.traces[0, 0, 0], 0.4)
    assert np.isclose(dataset.traces[0, 1, 0], 1.05)
    assert dataset.quality_summary["retained_sensor_count"] == 3


def test_recall_guardrail_threshold_meets_requested_recall() -> None:
    actual = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    probability = np.asarray([0.05, 0.1, 0.2, 0.4, 0.25, 0.45, 0.7, 0.9])
    threshold = _choose_recall_guardrail_threshold(actual, probability, minimum_recall=0.75)
    predicted = probability >= threshold
    recall = predicted[actual == 1].mean()
    assert recall >= 0.75


def test_quality_hurdle_forest_blend_fits_and_predicts() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(40, 8)), columns=[f"feature_{i}" for i in range(8)])
    y = np.r_[np.full(20, 0.35), np.full(20, 0.95)] + rng.normal(0, 0.02, 40)
    model = QualityHurdleForestBlendRegressor(
        selected_features=5,
        n_estimators=20,
        random_state=42,
    ).fit(X, y)

    prediction = model.predict(X)
    relevance = model.feature_relevance_table()

    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()
    assert set(relevance["task"]) == {"regression"}
    assert np.isclose(relevance["importance"].sum(), 1.0)


def test_production_regression_candidates_match_report_nine_algorithms() -> None:
    candidates = _make_regression_candidates(
        selected_features=5,
        random_state=42,
        n_estimators=20,
    )
    assert list(candidates) == [
        "Ridge",
        "Lasso",
        "ElasticNet",
        "KNN",
        "ExtraTrees",
        "RandomForest",
        "HistGBR",
        "XGBoost",
        "LightGBM",
    ]


def test_forest_search_tunes_growth_and_cost_complexity_pruning() -> None:
    grids = _regression_parameter_grids()
    for model_name in ("ExtraTrees", "RandomForest"):
        assert {row["model__max_depth"] for row in grids[model_name]} == {8, 12, None}
        assert {row["model__min_samples_leaf"] for row in grids[model_name]} == {2, 5}
        assert {row["model__ccp_alpha"] for row in grids[model_name]} == {0.0, 0.0001}


def test_search_profiles_are_bounded_and_balanced_remains_default() -> None:
    balanced = _regression_parameter_grids()
    conservative = _regression_parameter_grids("conservative")
    extensive = _regression_parameter_grids("extensive")

    assert len(conservative["RandomForest"]) < len(balanced["RandomForest"])
    assert len(extensive["RandomForest"]) > len(balanced["RandomForest"])
