from __future__ import annotations

from pathlib import Path

import pytest

from app import workflow
from app.guardrails import classify_intent, classify_intent_candidates, normalize_text
from tests.utils import load_jsonl


CASE_FILE = Path(__file__).resolve().parent / "datasets" / "intent_complex_cases.jsonl"
INTENT_CASES = load_jsonl(CASE_FILE)


@pytest.mark.parametrize("case", INTENT_CASES, ids=[str(x.get("id", "case")) for x in INTENT_CASES])
def test_complex_intent_dataset(case: dict) -> None:
    query = str(case.get("query", ""))
    expected_intent = str(case.get("expected_intent", "other"))
    normalized = normalize_text(query)
    predicted = classify_intent(normalized)
    assert predicted == expected_intent

    expected_secondary = [str(item).strip() for item in case.get("expected_secondary_intents", []) if str(item).strip()]
    if expected_secondary:
        ranked = classify_intent_candidates(normalized, max_items=1 + len(expected_secondary))
        assert ranked[0] == expected_intent
        assert ranked[1 : 1 + len(expected_secondary)] == expected_secondary


def test_intent_node_accepts_llm_refinement(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_refine(*, query: str, fallback_intent: str, llm_runtime=None):  # type: ignore[no-untyped-def]
        return {
            "intent": "medical_knowledge",
            "confidence": 0.92,
            "needs_handoff": False,
            "source": "llm",
        }

    monkeypatch.setattr(workflow, "classify_intent_with_llm", fake_refine)
    patch = workflow.intent_node(
        {
            "normalized_input": "我想了解糖尿病并发症和预后",
            "llm_provider": "default",
            "llm_model": "",
            "llm_thinking": False,
        }
    )

    assert patch["intent"] == "medical_knowledge"
    assert patch["intent_source"] == "llm"
    assert float(patch["intent_confidence"]) >= 0.9
