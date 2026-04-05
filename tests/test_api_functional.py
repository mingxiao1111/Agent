from __future__ import annotations

import json


def _fake_pipeline_state(query: str) -> dict:
    return {
        "answer": f"已收到：{query}",
        "intent": "symptom_consult",
        "intent_source": "rule",
        "intent_confidence": 0.87,
        "risk_level": "low",
        "confidence": 0.85,
        "handoff": False,
        "citations": ["kb:respiratory_01"],
        "handoff_summary": "",
        "follow_ups": ["咳嗽持续多久了？", "是否伴随发热？", "是否有基础病？"],
        "tool_results": {"department": "呼吸科"},
        "context_docs": [{"id": "kb:respiratory_01", "title": "呼吸科就诊", "content": "示例"}],
        "llm_provider": "default",
        "llm_model": "",
        "llm_thinking": False,
    }


def test_chat_api_smoke(client, web_module, monkeypatch) -> None:
    def fake_run_pipeline(query: str, **kwargs):  # type: ignore[no-untyped-def]
        trace = kwargs.get("trace")
        if isinstance(trace, list):
            trace.extend(
                [
                    {"stage": "risk", "status": "done", "risk_level": "low"},
                    {"stage": "intent", "status": "done", "intent": "symptom_consult"},
                ]
            )
        return _fake_pipeline_state(query)

    monkeypatch.setattr(web_module, "run_agent_async_pipeline_sync", fake_run_pipeline)

    resp = client.post(
        "/api/chat",
        json={
            "query": "我咳嗽两天了",
            "enable_cache": False,
            "llm_provider": "default",
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body, dict)
    assert body.get("answer", "").startswith("已收到：")
    assert body.get("intent") == "symptom_consult"
    assert isinstance(body.get("session_id"), str) and body.get("session_id")
    assert isinstance(body.get("pipeline_trace"), list) and body["pipeline_trace"]


def test_chat_stream_api_smoke(client, web_module, monkeypatch) -> None:
    def fake_run_pipeline(query: str, **kwargs):  # type: ignore[no-untyped-def]
        trace = kwargs.get("trace")
        if isinstance(trace, list):
            trace.append({"stage": "intent", "status": "done", "intent": "symptom_consult"})
        state = _fake_pipeline_state(query)
        state["answer"] = "规则兜底回答"
        return state

    def fake_stream_response(**kwargs):  # type: ignore[no-untyped-def]
        yield "这是"
        yield "流式"
        yield "回答"

    def fake_followups(**kwargs):  # type: ignore[no-untyped-def]
        return ["下一步做什么？", "需要就医吗？", "还有哪些注意事项？"]

    monkeypatch.setattr(web_module, "run_agent_async_pipeline_sync", fake_run_pipeline)
    monkeypatch.setattr(web_module, "stream_response_with_llm", fake_stream_response)
    monkeypatch.setattr(web_module, "generate_followups_with_llm", fake_followups)

    resp = client.post(
        "/api/chat/stream",
        json={
            "query": "流式接口测试",
            "enable_cache": False,
        },
    )
    assert resp.status_code == 200
    payload = resp.get_data(as_text=True)
    assert "event: token" in payload
    assert '"text": "这是"' in payload
    assert '"text": "流式"' in payload
    assert '"text": "回答"' in payload
    assert "event: meta" in payload
    assert "event: done" in payload


def test_feedback_api_persists_record(client, web_module) -> None:
    resp = client.post(
        "/api/chat/feedback",
        json={
            "reaction": "up",
            "session_id": "sess-demo",
            "message_id": "msg-demo",
            "user_query": "这条回答有帮助吗？",
            "assistant_answer": "有帮助，建议继续观察。",
            "meta": {"intent": "symptom_consult", "confidence": 0.9},
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body, dict) and body.get("ok") is True

    rel = str(body.get("saved_to", ""))
    target = web_module.BASE_DIR / rel
    assert target.exists()

    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved.get("reaction") == "up"
    assert saved.get("assistant_answer") == "有帮助，建议继续观察。"
