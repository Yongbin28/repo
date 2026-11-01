"""Shared preparation-to-modelling coordinator for manufacturing datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Optional, TypeVar


DatasetT = TypeVar("DatasetT")
ResultT = TypeVar("ResultT")


class PredictionCrew(Generic[DatasetT, ResultT]):
    """Prepare once, then pass that exact dataset to the model agent.

    Dataset-specific crews supply agents and options. Execution order, progress
    messages and stopping on preparation failure are shared by both routes.
    """

    def __init__(self, data_agent: Any, model_agent: Any) -> None:
        self.data_agent = data_agent
        self.model_agent = model_agent

    def _prepare(
        self,
        data_dir: Path,
        data_options: Optional[Mapping[str, Any]],
        log_func: Optional[Callable[[str], None]],
    ) -> DatasetT:
        if log_func:
            log_func(f"[Crew / Agent 1] {self.data_agent.name}")
        return self.data_agent.run(
            data_dir=data_dir, log_func=log_func, **dict(data_options or {})
        )

    def validate(
        self,
        data_dir: Path,
        *,
        data_options: Optional[Mapping[str, Any]] = None,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> DatasetT:
        """Run preparation without planning or fitting models."""
        return self._prepare(data_dir, data_options, log_func)

    def run(
        self,
        data_dir: Path,
        output_dir: Path,
        *,
        data_options: Optional[Mapping[str, Any]] = None,
        model_options: Optional[Mapping[str, Any]] = None,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> ResultT:
        dataset = self._prepare(data_dir, data_options, log_func)
        if log_func:
            log_func(f"[Crew / Agent 2] {self.model_agent.name}")
        return self.model_agent.run(
            dataset=dataset,
            output_dir=output_dir,
            log_func=log_func,
            **dict(model_options or {}),
        )
