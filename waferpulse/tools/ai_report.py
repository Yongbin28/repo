"""
waferpulse/tools/ai_report.py — AI Fab Engineering Shift Handover Note Generator

Generates automated, professional semiconductor fab engineering shift reports
and root-cause action items from wafer analytics results.
Powered by Groq's ultra-fast free inference API (llama-3.1-8b-instant) with
automatic graceful fallback to a deterministic rule-based template if no API
key is present.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"


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
) -> Dict[str, Any]:
    """
    Generate an AI Fab Engineering Shift Handover Report.
    
    Tries Groq (llama-3.1-8b-instant) first. If no API key or network error,
    seamlessly falls back to an expert deterministic fab engineering template.
    """
    effective_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
    
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

    if effective_key:
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {effective_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a Senior Fab Process Integration Engineer and Quality Lead at Micron Technology. "
                            "You write concise, high-impact Fab Engineering Shift Handover notes. "
                            "Focus on physical root-cause mechanisms (thermal, chuck clamping, etch bias, CMP slurry) "
                            "and actionable fab containment instructions. Keep responses under 200 words in professional engineering prose."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 350,
                "temperature": 0.25,
            }
            response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=6.0)
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                return {
                    "source": f"Groq Cloud ({DEFAULT_MODEL})",
                    "note": content,
                    "status": "success",
                }
            else:
                logger.warning(f"Groq API returned status {response.status_code}: {response.text}")
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
    outlier_text = ", ".join(top_outliers) if top_outliers else "None identified beyond 3-sigma"
    return f"""Analyze this semiconductor wafer test outcome and produce an official Fab Engineering Shift Handover Report:

- Wafer ID: {wafer_id}
- Product Family: {product_generic} ({part_number})
- Final Measured Yield: {yield_pct:.2f}% (Historical Baseline: {hist_pct:.2f}%, Delta: {yield_delta:+.2f}%)
- GDBN Reliability Grade: {reliability_grade}
- Spatial Defect Topology: {defect_pattern}
- Failed Dies: {num_fail_dies} out of {total_dies} tested ({fail_rate:.2f}% failure rate)
- Top Electrical Outliers: {outlier_text}

Include:
1. Shift Executive Summary & Yield Disposition
2. Probable Physical Process Root-Cause (e.g. plasma etch drift, CMP planarization pressure, thermal chuck non-uniformity)
3. Actionable Containment & Preventive Maintenance Recommendation for incoming fab shifts."""


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
    status_phrase = "Nominal production run" if yield_pct >= 90.0 else "Quality Excursion Alert"
    
    # Root cause heuristic based on pattern
    pattern_lower = defect_pattern.lower()
    if "edge" in pattern_lower:
        mechanism = "Edge-ring plasma sheath distortion or spin-coater resist bead non-uniformity along the perimeter."
        action = "Inspect electrostatic chuck clamping voltage and verify gas-flow distribution on Chamber 2."
    elif "ring" in pattern_lower or "donut" in pattern_lower:
        mechanism = "Radial thermal gradient across wafer center-to-edge during chemical mechanical planarization (CMP)."
        action = "Check pad conditioner downforce profile and slurry injection flow rate."
    elif "scratch" in pattern_lower:
        mechanism = "Mechanical handling abrasion or robotic end-effector particulate contamination."
        action = "Quarantine cassette robot FOUP-04 and inspect transfer arm vacuum suction cups."
    else:
        mechanism = "Random substrate point defects or localized gate oxide breakdown."
        action = "Proceed with standard inline defect review and verify baseline particle count."

    outliers_str = f"Critical parameter drift flagged on: {', '.join(top_outliers)}." if top_outliers else "All electrical limits remained within 3-sigma boundaries."

    return (
        f"**[FAB SHIFT HANDOVER REPORT — {wafer_id}]**\n\n"
        f"- **Disposition:** {status_phrase} ({reliability_grade}) on {product_generic}.\n"
        f"- **Yield Performance:** Yield evaluated at {yield_pct:.2f}% ({yield_delta:+.2f}% vs {hist_pct:.1f}% baseline), "
        f"with {num_fail_dies}/{total_dies} dies failing ({fail_rate:.1f}%).\n"
        f"- **Spatial Root Cause:** {defect_pattern} identified. Primary physical mechanism indicates {mechanism}\n"
        f"- **Electrical Diagnostics:** {outliers_str}\n"
        f"- **Containment Protocol:** {action}"
    )
