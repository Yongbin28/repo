"""Optional CrewAI/Groq controller for choosing an AutoML search profile.

The LLM may choose only among bounded, reviewed search profiles. It never
calculates metrics or approves the champion; those remain deterministic.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_AGENT_ITERATIONS = 8
MAX_AGENT_ITERATIONS = 8
DEFAULT_AGENT_TIMEOUT_SECONDS = 45
ALLOWED_SEARCH_PROFILES = {"conservative", "balanced", "extensive"}


@dataclass(frozen=True)
class AgenticSearchDecision:
    enabled: bool
    used_agent: bool
    search_profile: str
    model: str
    max_iterations: int
    status: str
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def choose_search_profile(
    dataset: Any,
    *,
    enabled: bool = False,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GROQ_MODEL,
    max_iterations: int = DEFAULT_MAX_AGENT_ITERATIONS,
    timeout_seconds: int = DEFAULT_AGENT_TIMEOUT_SECONDS,
    log_func: Optional[Callable[[str], None]] = None,
) -> AgenticSearchDecision:
    """Ask a bounded CrewAI agent for a profile; fall back on any failure."""

    summary = {
        "samples": int(len(dataset.features)),
        "features": int(dataset.features.shape[1]),
        "manufacturing_lots": int(dataset.groups.nunique()),
        "bad_wafers": int(dataset.bad_label.sum()),
        "good_wafers": int((dataset.bad_label == 0).sum()),
    }
    return choose_search_profile_from_summary(
        summary,
        enabled=enabled,
        api_key=api_key,
        model=model,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
        log_func=log_func,
    )


def choose_search_profile_from_summary(
    summary: Dict[str, Any],
    *,
    enabled: bool = False,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GROQ_MODEL,
    max_iterations: int = DEFAULT_MAX_AGENT_ITERATIONS,
    timeout_seconds: int = DEFAULT_AGENT_TIMEOUT_SECONDS,
    log_func: Optional[Callable[[str], None]] = None,
) -> AgenticSearchDecision:
    """Dataset-agnostic bounded CrewAI planner used by all prediction lanes."""

    iterations = max(1, min(int(max_iterations), MAX_AGENT_ITERATIONS))

    def fallback(reason: str, status: str = "fallback") -> AgenticSearchDecision:
        if log_func:
            log_func(f"[Agentic AutoML] {reason} Using deterministic balanced workflow.")
        return AgenticSearchDecision(
            enabled=enabled,
            used_agent=False,
            search_profile="balanced",
            model=model,
            max_iterations=iterations,
            status=status,
            reason=reason,
        )

    if not enabled:
        return fallback("Groq controller disabled.", status="disabled")

    effective_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not effective_key:
        return fallback("GROQ_API_KEY is unavailable.")

    try:
        from crewai import Agent, Crew, LLM, Process, Task
        from crewai.tools import tool
        from pydantic import BaseModel, Field
    except ImportError as exc:
        return fallback(f"CrewAI dependency unavailable: {exc}")

    class SearchPlan(BaseModel):
        search_profile: str = Field(
            description="Exactly one of conservative, balanced, or extensive"
        )
        reason: str = Field(description="Short technical justification")

    @tool("inspect_allowed_search_profiles")
    def inspect_allowed_search_profiles() -> str:
        """Return the only permitted AutoML search profiles and their tradeoffs."""

        return (
            "conservative: smallest grids, fastest, best for limited samples/resources; "
            "balanced: default reviewed grids and recommended general choice; "
            "extensive: wider grids, slowest, use only with enough lots and samples."
        )

    try:
        llm = LLM(
            model=f"groq/{model}",
            api_key=effective_key,
            temperature=0,
            timeout=timeout_seconds,
        )
        agent = Agent(
            role="Semiconductor AutoML Search Planner",
            goal=(
                "Choose a bounded hyperparameter-search profile for honest unseen-lot "
                "validation. Never select the final model; deterministic R² does that."
            ),
            backstory=(
                "You specialize in semiconductor virtual metrology, grouped validation, "
                "overfitting control, and computationally efficient model search."
            ),
            llm=llm,
            tools=[inspect_allowed_search_profiles],
            allow_delegation=False,
            max_iter=iterations,
            max_execution_time=timeout_seconds,
            verbose=False,
        )
        task = Task(
            description=(
                f"Dataset summary: {summary}. First inspect the allowed profiles using the "
                "tool. Then choose exactly one profile. Prefer balanced unless the evidence "
                "clearly supports conservative or extensive. Return only the structured plan."
            ),
            expected_output="A structured search profile and short technical reason.",
            output_pydantic=SearchPlan,
            agent=agent,
        )
        output = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()
        plan = output.pydantic
        if plan is None:
            return fallback("Groq agent did not return a valid structured plan.")
        profile = str(plan.search_profile).strip().lower()
        if profile not in ALLOWED_SEARCH_PROFILES:
            return fallback(f"Groq agent returned disallowed profile: {profile!r}.")
        if log_func:
            log_func(
                f"[Agentic AutoML] Groq selected '{profile}' within the "
                f"{iterations}-iteration cap: {plan.reason}"
            )
        return AgenticSearchDecision(
            enabled=True,
            used_agent=True,
            search_profile=profile,
            model=model,
            max_iterations=iterations,
            status="success",
            reason=str(plan.reason),
        )
    except Exception as exc:
        return fallback(f"Groq/CrewAI controller failed: {type(exc).__name__}: {exc}")
