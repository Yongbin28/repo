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
        use_ai=False,
    )
    assert result["status"] == "fallback"
    assert "Deterministic Fab Rule Engine" in result["source"]
    assert "SYN_0044-01" in result["note"]
    assert "DDR4SDRAM" in result["note"]
    assert "71.54%" in result["note"]
    assert "Suggested follow-up" in result["note"]
    assert "Predicted yield" in result["note"]
    assert "not a confirmed cause or mechanism" in result["note"]
    assert "chuck" not in result["note"].lower()
    assert "quarantine" not in result["note"].lower()


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
        use_ai=False,
    )
    assert result["status"] == "fallback"
    assert "Predicted yield:** 94.50%" in result["note"]
    assert "Grade A" in result["note"]


def test_ai_report_does_not_send_data_without_explicit_opt_in(monkeypatch):
    import requests

    def unexpected_request(*args, **kwargs):
        pytest.fail("No external request should be made without explicit opt-in")

    monkeypatch.setattr(requests, "post", unexpected_request)
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
        top_outliers=["T44.0 (Leakage)"],
    )

    assert result["status"] == "fallback"


def test_ai_report_external_generation_requires_opt_in_and_uses_predicted_wording(monkeypatch):
    import requests

    observed = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "Review the predicted yield."}}]}

    def mocked_post(url, *, headers, json, timeout):
        observed["url"] = url
        observed["headers"] = headers
        observed["payload"] = json
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr(requests, "post", mocked_post)
    result = generate_shift_handover_note(
        wafer_id="LOT-01",
        product_generic="TEST",
        part_number="P1",
        predicted_yield=0.8,
        historical_avg_yield=0.9,
        reliability_grade="Grade B",
        defect_pattern="Edge pattern flag",
        num_fail_dies=20,
        total_dies=100,
        api_key="test-key",
        use_ai=True,
    )

    assert result["status"] == "success"
    assert "Groq Cloud" in result["source"]
    assert "Predicted Yield: 80.00%" in observed["payload"]["messages"][1]["content"]
    assert "Final Measured Yield" not in observed["payload"]["messages"][1]["content"]
    assert "Do not infer or assert physical causes" in observed["payload"]["messages"][0]["content"]
    assert observed["headers"]["Authorization"] == "Bearer test-key"
