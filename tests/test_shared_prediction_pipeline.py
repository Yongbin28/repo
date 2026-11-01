"""Verify both routes use the same handover and bounded planning behavior."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from waferpulse.agents import equipment_model_agent
from waferpulse.core.bosch_contracts import BoschDataset
from waferpulse.crews import BoschPredictionCrew, EquipmentPredictionCrew
from waferpulse.experiments import bosch_plasma_etch_benchmark as bosch
from waferpulse.experiments import bosch_response_then_classify_benchmark as screening


class DataAgent:
    name = "fixture preparation"

    def __init__(self, dataset=None, failure=None):
        self.dataset = dataset
        self.failure = failure
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return self.dataset


class ModelAgent:
    name = "fixture modelling"

    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["dataset"]


def run_crew(crew, tmp_path, **kwargs):
    if isinstance(crew, BoschPredictionCrew):
        kwargs["classification_output_dir"] = tmp_path / "screening"
    return crew.run(tmp_path / "data", tmp_path / "models", **kwargs)


@pytest.mark.parametrize("crew_class", [EquipmentPredictionCrew, BoschPredictionCrew])
def test_preparation_failure_stops_before_modelling(crew_class, tmp_path):
    data_agent = DataAgent(failure=ValueError("invalid source"))
    model_agent = ModelAgent()
    crew = crew_class(data_agent=data_agent, model_agent=model_agent)

    with pytest.raises(ValueError, match="invalid source"):
        run_crew(crew, tmp_path)

    assert len(data_agent.calls) == 1
    assert model_agent.calls == []


@pytest.mark.parametrize("crew_class", [EquipmentPredictionCrew, BoschPredictionCrew])
def test_validate_only_prepares_and_run_passes_same_object(crew_class, tmp_path):
    dataset = object()
    data_agent = DataAgent(dataset)
    model_agent = ModelAgent()
    crew = crew_class(data_agent=data_agent, model_agent=model_agent)

    assert crew.validate(tmp_path / "data") is dataset
    assert model_agent.calls == []

    messages = []
    assert run_crew(crew, tmp_path, log_func=messages.append) is dataset
    assert len(data_agent.calls) == 2  # once for validate, once for the full run
    assert len(model_agent.calls) == 1
    assert model_agent.calls[0]["dataset"] is dataset
    assert "Agent 1" in messages[0]
    assert "Agent 2" in messages[1]


@pytest.mark.parametrize("route", ["equipment", "bosch"])
@pytest.mark.parametrize("status_code, expected_profile", [(200, "conservative"), (429, "balanced")])
def test_both_routes_plan_from_prepared_data_and_share_fallback(
    route, status_code, expected_profile, monkeypatch, tmp_path
):
    requests_seen = []
    training_calls = []
    if route == "equipment":
        dataset = SimpleNamespace(
            features=pd.DataFrame(np.zeros((4, 2))),
            groups=pd.Series([1, 1, 2, 2]),
            bad_label=pd.Series([0, 1, 0, 1]),
        )
        crew = EquipmentPredictionCrew(data_agent=DataAgent(dataset))

        def fit_equipment(**kwargs):
            training_calls.append(kwargs)
            return "equipment-result"

        monkeypatch.setattr(equipment_model_agent, "train_equipment_models", fit_equipment)
    else:
        dataset = BoschDataset(
            process=pd.DataFrame({"experiment_key": ["a", "b"], "sensor": [1.0, 2.0]}),
            measurements=pd.DataFrame({
                "experiment_key": ["a", "b", "unmatched"], "lot_number": [1, 2, 3]
            }),
            metadata={"matched_wafers": 999, "lots": 999},
        )
        crew = BoschPredictionCrew(data_agent=DataAgent(dataset))

        def fit_bosch(**kwargs):
            training_calls.append(kwargs)
            return pd.DataFrame({"r2": [0.5]})

        monkeypatch.setattr(bosch, "train_bosch_models", fit_bosch)
        monkeypatch.setattr(screening, "train_bosch_screening", fit_bosch)

    def respond(*args, **kwargs):
        requests_seen.append(kwargs["json"])
        return SimpleNamespace(
            status_code=status_code,
            json=lambda: {"choices": [{"message": {"content": json.dumps({
                "search_profile": "conservative", "reason": "small dataset"
            })}}]},
        )

    monkeypatch.setattr(requests, "post", respond)
    run_crew(crew, tmp_path, use_agentic_controller=True, groq_api_key="fixture-key")

    assert len(requests_seen) == 1
    summary = requests_seen[0]["messages"][1]["content"]
    assert '"manufacturing_lots": 2' in summary
    assert f'"samples": {4 if route == "equipment" else 2}' in summary
    assert len(training_calls) == (1 if route == "equipment" else 2)
    for call in training_calls:
        assert call["dataset"] is dataset
        assert call["search_profile"] == expected_profile
        assert call["agentic_decision"]["used_agent"] is (status_code == 200)


def test_bosch_pipeline_loads_once_and_exports_both_evidence_sets(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # The real loader checks source presence, then uses the feature cache.
    (data_dir / "Process_data.nc").touch()
    (data_dir / "Dictionary_process.nc").touch()
    process_rows = []
    measurement_rows = []
    for wafer in range(24):
        key = f"wafer-{wafer}"
        process_rows.append({"experiment_key": key, "sensor": float(wafer)})
        for point in range(2):
            measurement_rows.append({
                "experiment_key": key,
                "lot_number": wafer // 4,
                "wafer_number": wafer,
                "X": point * 10_000.0,
                "Y": point * 20_000.0,
                "si_etch": 40.0 + wafer * 0.4 + point * 0.1,
                "oxide_etch": 20.0 + wafer * 0.2 + point * 0.05,
            })
    pd.DataFrame(process_rows).to_csv(data_dir / "bosch_process_wafer_features.csv", index=False)
    pd.DataFrame(measurement_rows).to_csv(data_dir / "Si_Oxide_etch_89_points.csv", index=False)

    loads = []
    load_dataset = bosch.load_dataset
    candidates = bosch.model_candidates
    profiles = []

    def load_once(*args, **kwargs):
        dataset = load_dataset(*args, **kwargs)
        loads.append(dataset)
        return dataset

    def small_candidates(count, search_profile="balanced"):
        profiles.append(search_profile)
        return {name: model for name, model in candidates(count, search_profile).items()
                if name in {"dummy_mean", "ridge"}}

    monkeypatch.setattr(bosch, "load_dataset", load_once)
    monkeypatch.setattr(bosch, "model_candidates", small_candidates)
    monkeypatch.setattr(screening, "model_candidates", small_candidates)
    monkeypatch.setattr(bosch, "_generate_bosch_shap", lambda *args: None)

    # Exercise the real planner's deterministic disabled/default path.
    result = BoschPredictionCrew().run(
        data_dir=data_dir,
        output_dir=tmp_path / "regression",
        classification_output_dir=tmp_path / "screening",
        threshold=44.5,
    )

    assert len(loads) == 1
    assert result.dataset is loads[0]
    assert result.regression_metrics["r2"].notna().all()
    assert result.screening_metrics["research_threshold"].eq(44.5).all()
    assert set(profiles) == {"balanced"}
    for name in ["metrics.csv", "oof_predictions.csv", "bosch_benchmark_report.xlsx"]:
        assert (tmp_path / "regression" / name).is_file()
    assert (tmp_path / "screening" / "oof_response_then_class.csv").is_file()
    for directory in ["regression", "screening"]:
        summary = json.loads((tmp_path / directory / "summary.json").read_text())
        assert summary["regression_search_profile"] == "balanced"
        assert summary["agentic_controller"]["status"] == "disabled"


def test_bosch_dashboard_runs_the_shared_crew(monkeypatch, tmp_path):
    from waferpulse.config import WaferPulsePaths
    from waferpulse.dashboard import bosch_plasma_etch_page as page

    calls = []
    successes = []
    errors = []
    monkeypatch.setattr(page, "BoschPredictionCrew", lambda: SimpleNamespace(
        run=lambda **kwargs: calls.append(kwargs)
    ))
    monkeypatch.setattr(page, "st", SimpleNamespace(
        empty=lambda: SimpleNamespace(code=lambda text: None),
        spinner=lambda text: nullcontext(),
        success=successes.append,
        info=errors.append,
        error=errors.append,
        exception=errors.append,
    ))
    paths = WaferPulsePaths.from_root(tmp_path)
    page._run_full_benchmark(
        paths,
        refresh_cache=True,
        threshold=45.0,
        use_agentic_controller=False,
        groq_api_key="",
        max_agent_iterations=3,
    )

    assert errors == []
    assert len(successes) == 1
    assert len(calls) == 1
    assert calls[0]["data_dir"] == paths.bosch_data
    assert calls[0]["output_dir"] == paths.bosch_output
    assert calls[0]["classification_output_dir"] == paths.bosch_classification_output
    assert calls[0]["threshold"] == 45.0
    assert calls[0]["refresh_cache"] is True
    assert calls[0]["max_agent_iterations"] == 3
