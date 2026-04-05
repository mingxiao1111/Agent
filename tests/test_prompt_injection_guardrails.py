from __future__ import annotations

from pathlib import Path

import pytest

from app.async_pipeline import run_agent_async_pipeline_sync
from app.guardrails import detect_high_risk, normalize_text
from tests.utils import load_jsonl


CASE_FILE = Path(__file__).resolve().parent / "datasets" / "prompt_injection_cases.jsonl"
INJECTION_CASES = load_jsonl(CASE_FILE)
BLOCK_CASES = [case for case in INJECTION_CASES if bool(case.get("expect_block", False))]


@pytest.mark.parametrize("case", INJECTION_CASES, ids=[str(x.get("id", "case")) for x in INJECTION_CASES])
def test_prompt_injection_and_risk_dataset(case: dict) -> None:
    query = str(case.get("query", ""))
    expected = bool(case.get("expect_block", False))

    flagged, reason = detect_high_risk(normalize_text(query))
    assert flagged is expected
    if expected:
        assert reason != "none"


@pytest.mark.parametrize("case", BLOCK_CASES, ids=[str(x.get("id", "case")) for x in BLOCK_CASES])
def test_blocked_cases_short_circuit_pipeline(case: dict) -> None:
    result = run_agent_async_pipeline_sync(
        str(case.get("query", "")),
        disable_llm_response=True,
        disable_llm_followups=True,
        use_cache=False,
    )

    assert result.get("risk_level") == "high"
    assert result.get("intent") == "emergency"
    assert bool(result.get("handoff")) is True
    assert bool(result.get("answer"))
