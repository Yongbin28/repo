"""
waferpulse/tools/ai_report.py — AI Fab Engineering Shift Handover Note Generator

Generates a shift handover summary from supplied wafer analytics. External
generation is opt-in; otherwise a deterministic local template is used.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def sanitize_api_key(raw_key: Optional[str]) -> str:
    """Sanitize API key by stripping whitespace, quotes, and common copy-paste artifacts."""
    if not raw_key:
        return ""
    key = str(raw_key).strip().strip("'\"")
    # Common typo when copy-pasting e.g. "agsk_..." from Chinese punctuation or prefix
    if key.startswith("agsk_"):
        key = key[1:]
    return key


def generate_shift_handover_note(
    wafer_id: str,
    product_generic: str,
    part_number: str,
    predicted_yield: float,
    historical_avg_yield: float,
    reliability_grade: str,
    defect_pattern: str,
    num_fail_dies: int,
    total_dies: int,
    top_outliers: Optional[list] = None,
    api_key: Optional[str] = None,
    use_ai: bool = False,
) -> Dict[str, Any]:
    """
    Generate an AI Fab Engineering Shift Handover Report.
    
    Sends supplied details to Groq only when ``use_ai`` is explicitly enabled
    and an API key is supplied through the argument, environment, or Streamlit
    secrets. Otherwise, generates a local template.
    """
    effective_key = ""
    if use_ai and api_key != "disabled":
        if api_key:
            effective_key = sanitize_api_key(api_key)
        if not effective_key:
            effective_key = sanitize_api_key(os.environ.get("GROQ_API_KEY", ""))
        if not effective_key:
            try:
                import streamlit as st
                effective_key = sanitize_api_key(str(st.secrets.get("GROQ_API_KEY", "")))
            except Exception:
                pass
    
    # Calculate yield metrics
    yield_pct = predicted_yield * 100.0 if predicted_yield <= 1.0 else predicted_yield
    hist_pct = historical_avg_yield * 100.0 if historical_avg_yield <= 1.0 else historical_avg_yield
    yield_delta = yield_pct - hist_pct
    fail_rate = (num_fail_dies / max(total_dies, 1)) * 100.0

    prompt = _build_fab_prompt(
        wafer_id=wafer_id,
        product_generic=product_generic,
        part_number=part_number,
        yield_pct=yield_pct,
        hist_pct=hist_pct,
        yield_delta=yield_delta,
        reliability_grade=reliability_grade,
        defect_pattern=defect_pattern,
        num_fail_dies=num_fail_dies,
        total_dies=total_dies,
        fail_rate=fail_rate,
        top_outliers=top_outliers or [],
    )

    if effective_key and use_ai:
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {effective_key}",
                "Content-Type": "application/json",
            }
            system_msg = (
                "Write a concise engineering handover from supplied data. Clearly label model outputs as predictions "
                "and distinguish them from measurements. Treat pattern names and outlier labels as unverified inputs. "
                "Do not infer or assert physical causes, mechanisms, equipment conditions, or confirmed defects. "
                "Offer only suggestions to verify source measurements and review approved process records. "
                "Do not recommend equipment setting changes or production disposition. Keep under 200 words."
            )
            for model_name in DEFAULT_MODELS:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 350,
                    "temperature": 0.25,
                }
                try:
                    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=12.0)
                    if response.status_code == 200:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        return {
                            "source": f"Groq Cloud ({model_name})",
                            "note": content,
                            "status": "success",
                        }
                    else:
                        logger.warning(f"Groq API model {model_name} returned status {response.status_code}: {response.text}")
                except Exception as req_err:
                    logger.warning(f"Groq model {model_name} request failed: {req_err}")
        except Exception as exc:
            logger.warning(f"Groq API call failed ({exc}); using expert template.")

    # Graceful fallback to deterministic expert template
    fallback_note = _generate_template_note(
        wafer_id=wafer_id,
        product_generic=product_generic,
        yield_pct=yield_pct,
        hist_pct=hist_pct,
        yield_delta=yield_delta,
        reliability_grade=reliability_grade,
        defect_pattern=defect_pattern,
        num_fail_dies=num_fail_dies,
        total_dies=total_dies,
        fail_rate=fail_rate,
        top_outliers=top_outliers or [],
    )
    return {
        "source": "Deterministic Fab Rule Engine (Template Fallback)",
        "note": fallback_note,
        "status": "fallback",
    }


def _build_fab_prompt(
    wafer_id: str,
    product_generic: str,
    part_number: str,
    yield_pct: float,
    hist_pct: float,
    yield_delta: float,
    reliability_grade: str,
    defect_pattern: str,
    num_fail_dies: int,
    total_dies: int,
    fail_rate: float,
    top_outliers: list,
) -> str:
    outlier_text = ", ".join(top_outliers) if top_outliers else "None supplied"
    return f"""Prepare a concise engineering handover summary from these supplied values. They may contain confidential production information. The caller must explicitly authorize external processing before this prompt is sent to Groq. Do not describe predictions as measurements. Do not infer or assert a physical root cause, mechanism, equipment condition, or confirmed defect from the supplied pattern label. Treat all supplied labels and counts as unverified and suggest verification only. Do not make production disposition or equipment-setting recommendations.

- Wafer ID: {wafer_id}
- Product Family: {product_generic} ({part_number})
- Predicted Yield: {yield_pct:.2f}% (Historical Average: {hist_pct:.2f}%, Delta: {yield_delta:+.2f} percentage points)
- GDBN Reliability Grade: {reliability_grade}
- Spatial Defect Topology: {defect_pattern}
- Failed Dies: {num_fail_dies} out of {total_dies} tested ({fail_rate:.2f}% failure rate)
- Top Electrical Outliers: {outlier_text}

Include a short summary, clearly distinguish prediction from measurement, identify the supplied pattern and outlier labels as items for review, and suggest checking the source measurements and approved process records. Do not claim a diagnosis or direct a production action."""


def _generate_template_note(
    wafer_id: str,
    product_generic: str,
    yield_pct: float,
    hist_pct: float,
    yield_delta: float,
    reliability_grade: str,
    defect_pattern: str,
    num_fail_dies: int,
    total_dies: int,
    fail_rate: float,
    top_outliers: list,
) -> str:
    outliers_str = (
        f"Supplied outlier labels: {', '.join(top_outliers)}."
        if top_outliers
        else "No outlier labels were supplied to this handover function."
    )

    return (
        f"**[FAB SHIFT HANDOVER REPORT — {wafer_id}]**\n\n"
        f"- **Model summary:** {reliability_grade} was supplied for {product_generic}; this note does not determine disposition.\n"
        f"- **Predicted yield:** {yield_pct:.2f}% ({yield_delta:+.2f} percentage points versus the supplied historical average of {hist_pct:.1f}%).\n"
        f"- **Reported failed-die count:** {num_fail_dies}/{total_dies} ({fail_rate:.1f}%), as supplied.\n"
        f"- **Pattern for review:** {defect_pattern}. This is a supplied label, not a confirmed cause or mechanism.\n"
        f"- **Outlier information:** {outliers_str}\n"
        "- **Suggested follow-up:** Verify these labels and counts against source measurements and approved records; investigate independently before making production or equipment decisions."
    )
