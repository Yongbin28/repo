"""Optional lightweight Groq controller for choosing an AutoML search profile.

The LLM may choose only among bounded, reviewed search profiles. It never
calculates metrics or approves the champion; those remain deterministic.
"""

from __future__ import annotations

import os
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_AGENT_ITERATIONS = 8
MAX_AGENT_ITERATIONS = 8
DEFAULT_AGENT_TIMEOUT_SECONDS = 45
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
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
    """Ask a bounded Groq agent for a profile; fall back on any failure."""

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
    """Dataset-agnostic bounded Groq planner used by all prediction lanes."""

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
        try:
            import streamlit as st

            effective_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            effective_key = ""
    if not effective_key:
        return fallback("GROQ_API_KEY is unavailable.")

    try:
        import requests

        system_message = (
            "You are a semiconductor AutoML search planner. Choose a bounded search "
            "profile, never a winning model. Valid profiles: conservative (fastest, "
            "lower overfitting risk), balanced (default reviewed search), extensive "
            "(slowest, wider/deeper tree search for sufficient data). Return JSON only "
            "with keys search_profile and reason. Prefer balanced unless the dataset "
            "clearly justifies another profile."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Dataset summary: {json.dumps(summary)}"},
        ]
        deadline = time.monotonic() + max(1, int(timeout_seconds))
        last_problem = "no valid response"
        for attempt in range(1, iterations + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return fallback("Groq planner exceeded its execution-time limit.")
            response = requests.post(
                GROQ_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {effective_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 180,
                    "response_format": {"type": "json_object"},
                },
                timeout=min(12.0, max(1.0, remaining)),
            )
            if response.status_code == 429:
                return fallback("Groq free-plan rate limit reached (HTTP 429).")
            if response.status_code != 200:
                return fallback(f"Groq API returned HTTP {response.status_code}.")
            content = response.json()["choices"][0]["message"]["content"].strip()
            try:
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                plan = json.loads(match.group(0) if match else content)
                profile = str(plan.get("search_profile", "")).strip().lower()
                reason = str(plan.get("reason", "")).strip()
                if profile in ALLOWED_SEARCH_PROFILES and reason:
                    if log_func:
                        log_func(
                            f"[Agentic AutoML] Groq selected '{profile}' on decision "
                            f"{attempt}/{iterations}: {reason}"
                        )
                    return AgenticSearchDecision(
                        enabled=True,
                        used_agent=True,
                        search_profile=profile,
                        model=model,
                        max_iterations=iterations,
                        status="success",
                        reason=reason,
                    )
                last_problem = f"disallowed or incomplete plan: {plan}"
            except (ValueError, TypeError, AttributeError) as exc:
                last_problem = f"invalid JSON: {exc}"
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Invalid plan. Return JSON only with a valid search_profile "
                            "and a non-empty reason."
                        ),
                    },
                ]
            )
        return fallback(
            f"Groq planner reached the {iterations}-decision limit ({last_problem})."
        )
    except Exception as exc:
        return fallback(f"Groq controller failed: {type(exc).__name__}: {exc}")
