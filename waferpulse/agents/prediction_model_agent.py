"""Common bounded planning and deterministic training stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from waferpulse.agents.groq_model_selection_agent import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_AGENT_ITERATIONS,
    choose_search_profile_from_summary,
)


DatasetT = TypeVar("DatasetT")
ResultT = TypeVar("ResultT")


class PredictionModelAgent(Generic[DatasetT, ResultT]):
    """Choose a profile from current prepared data, then evaluate and save."""

    def summarize(self, dataset: DatasetT) -> Dict[str, Any]:
        raise NotImplementedError

    def train(self, dataset: DatasetT, **kwargs: Any) -> ResultT:
        raise NotImplementedError

    def run(
        self,
        dataset: DatasetT,
        output_dir: Path,
        *,
        use_agentic_controller: bool = False,
        groq_api_key: Optional[str] = None,
        groq_model: str = DEFAULT_GROQ_MODEL,
        max_agent_iterations: int = DEFAULT_MAX_AGENT_ITERATIONS,
        log_func: Optional[Callable[[str], None]] = None,
        **model_options: Any,
    ) -> ResultT:
        decision = choose_search_profile_from_summary(
            self.summarize(dataset),
            enabled=use_agentic_controller,
            api_key=groq_api_key,
            model=groq_model,
            max_iterations=max_agent_iterations,
            log_func=log_func,
        )
        return self.train(
            dataset=dataset,
            output_dir=Path(output_dir),
            search_profile=decision.search_profile,
            agentic_decision=decision.as_dict(),
            log_func=log_func,
            **model_options,
        )
