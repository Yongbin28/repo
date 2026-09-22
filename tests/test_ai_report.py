"""
tests/test_ai_report.py — Unit tests for AI Fab Engineering Shift Report Generator
"""

import pytest
from waferpulse.tools.ai_report import generate_shift_handover_note


def test_ai_report_fallback_generation():
    result = generate_shift_handover_note(
        wafer_id="SYN_0044-01",
        product_generic="DDR4SDRAM",
        part_number="MT40A1G8",
        predicted_yield=0.7154,
        historical_avg_yield=0.884,
        reliability_grade="Grade C (Derated)",
        defect_pattern="Perimeter Edge-Ring Cluster (23 edge dies)",
        num_fail_dies=182,
        total_dies=638,
        top_outliers=["T44.0 (Leakage)", "T12.0 (Access Time)"],
        api_key=None,
    )
    assert result["status"] == "fallback"
    assert "Deterministic Fab Rule Engine" in result["source"]
    assert "SYN_0044-01" in result["note"]
    assert "DDR4SDRAM" in result["note"]
    assert "71.54%" in result["note"]
    assert "Containment Protocol" in result["note"]


def test_ai_report_high_yield_disposition():
    result = generate_shift_handover_note(
        wafer_id="SYN_0053-01",
        product_generic="DDR4SDRAM",
        part_number="MT40A1G8",
        predicted_yield=0.945,
        historical_avg_yield=0.910,
        reliability_grade="Grade A (Automotive)",
        defect_pattern="Nominal Random Defect Distribution",
        num_fail_dies=35,
        total_dies=638,
        top_outliers=[],
        api_key=None,
    )
    assert result["status"] == "fallback"
    assert "Nominal production run" in result["note"]
    assert "Grade A" in result["note"]
