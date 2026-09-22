from pathlib import Path

from waferpulse.crews import EquipmentPredictionCrew


class FakeDataAgent:
    name = "fake-data"

    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return "validated-dataset"


class FakeModelAgent:
    name = "fake-model"

    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return "model-result"


def test_crew_runs_data_agent_before_model_agent(tmp_path: Path) -> None:
    data_agent = FakeDataAgent()
    model_agent = FakeModelAgent()
    messages = []
    crew = EquipmentPredictionCrew(data_agent=data_agent, model_agent=model_agent)

    result = crew.run(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        log_func=messages.append,
    )

    assert result == "model-result"
    assert data_agent.calls[0]["data_dir"] == tmp_path / "data"
    assert model_agent.calls[0]["dataset"] == "validated-dataset"
    assert model_agent.calls[0]["output_dir"] == tmp_path / "output"
    assert "Agent 1" in messages[0]
    assert "Agent 2" in messages[1]
