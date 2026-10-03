import numpy as np
import pandas as pd
import requests

from waferpulse.agents.groq_model_selection_agent import (
    MAX_AGENT_ITERATIONS,
    choose_search_profile,
)


class MinimalDataset:
    features = pd.DataFrame(np.zeros((6, 2)))
    groups = pd.Series(["a", "a", "b", "b", "c", "c"])
    bad_label = pd.Series([0, 1, 0, 1, 0, 1])


def test_disabled_agent_uses_deterministic_balanced_workflow() -> None:
    decision = choose_search_profile(MinimalDataset(), enabled=False)

    assert decision.status == "disabled"
    assert decision.used_agent is False
    assert decision.search_profile == "balanced"


def test_missing_key_falls_back_and_caps_max_iterations(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    decision = choose_search_profile(
        MinimalDataset(),
        enabled=True,
        api_key="",
        max_iterations=999,
    )

    assert decision.status == "fallback"
    assert decision.used_agent is False
    assert decision.search_profile == "balanced"
    assert decision.max_iterations == MAX_AGENT_ITERATIONS


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_direct_groq_json_selects_allowed_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"search_profile":"conservative","reason":"small data"}'
                        }
                    }
                ]
            },
        ),
    )
    decision = choose_search_profile(
        MinimalDataset(), enabled=True, api_key="test-key", max_iterations=8
    )

    assert decision.status == "success"
    assert decision.used_agent is True
    assert decision.search_profile == "conservative"


def test_groq_rate_limit_falls_back_without_aborting_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(429),
    )
    decision = choose_search_profile(
        MinimalDataset(), enabled=True, api_key="test-key", max_iterations=8
    )

    assert decision.status == "fallback"
    assert decision.search_profile == "balanced"
    assert "429" in decision.reason
