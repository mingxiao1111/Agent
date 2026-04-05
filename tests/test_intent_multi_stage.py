from __future__ import annotations

import pytest

from app import workflow


def test_intent_node_merges_secondary_intents(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_refine(*, query: str, fallback_intent: str, llm_runtime=None):  # type: ignore[no-untyped-def]
        return {
            "intent": "report_interpretation",
            "confidence": 0.91,
            "needs_handoff": False,
            "secondary_intents": ["appointment_process"],
            "source": "llm",
        }

    monkeypatch.setattr(workflow, "classify_intent_with_llm", fake_refine)
    patch = workflow.intent_node(
        {
            "normalized_input": "我发烧三天了，血常规报告怎么看，另外挂哪个科",
            "llm_provider": "default",
            "llm_model": "",
            "llm_thinking": False,
        }
    )

    assert patch["intent"] == "report_interpretation"
    assert patch["secondary_intents"] == ["appointment_process", "symptom_consult"]
    assert patch["intent_candidates"] == ["report_interpretation", "appointment_process", "symptom_consult"]


def test_response_node_handles_multi_intent_rule_answer() -> None:
    patch = workflow.response_node(
        {
            "user_input": "我最近咳嗽，还想顺便问挂哪个科",
            "intent": "appointment_process",
            "secondary_intents": ["symptom_consult"],
            "intent_source": "rule",
            "intent_confidence": 0.88,
            "risk_level": "low",
            "tool_results": {
                "department": "呼吸科",
                "schedule": "周一到周五",
                "booking_steps": "先实名认证，再挂号。",
            },
            "context_docs": [],
            "citations": [],
            "disable_llm_response": True,
            "disable_llm_followups": True,
        }
    )

    assert "复合诉求" in patch["answer"]
    assert "呼吸科" in patch["answer"]
    assert "持续加重" in patch["answer"]
    assert any("挂号后错过时间" in item for item in patch["follow_ups"])
    assert any("基础检查" in item for item in patch["follow_ups"])
