import numpy as np
import pandas as pd

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
