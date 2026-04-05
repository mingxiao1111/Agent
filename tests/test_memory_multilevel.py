from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.utils import load_jsonl


CASE_FILE = Path(__file__).resolve().parent / "datasets" / "memory_multilevel_cases.jsonl"
ALL_MEMORY_CASES = load_jsonl(CASE_FILE)
MEMORY_CASES = {str(item.get("id", "")): item for item in ALL_MEMORY_CASES}
M3_CASES = [item for item in ALL_MEMORY_CASES if str(item.get("kind", "")).strip() == "m3"]


def _build_session_from_case(web_module: Any, case: dict[str, Any]) -> dict[str, Any]:
    _, session = web_module._new_general_session()
    for turn in case.get("turns", []):
        web_module._append_general_turn(
            session,
            str(turn.get("query", "")),
            str(turn.get("answer", "")),
            intent=str(turn.get("intent", "other")),
            intent_confidence=float(turn.get("intent_confidence", 0.8) or 0.8),
            llm_runtime={"provider": "default", "model": "", "thinking": False},
        )
    return session


def _expected_items(case: dict[str, Any]) -> list[str]:
    raw_items = case.get("expected_items")
    if isinstance(raw_items, list):
        return [str(item).strip() for item in raw_items if str(item).strip()]

    single = str(case.get("expected_item", "")).strip()
    return [single] if single else []


def test_m0_window_and_m1_summary(web_module) -> None:
    case = MEMORY_CASES["memory_001"]
    session = _build_session_from_case(web_module, case)
    history = web_module._build_general_history_text(session, query=str(case.get("probe_query", "")))

    assert "最近对话(M0):" in history
    assert "段内摘要(M1):" in history

    # M0 只保留最近 3 轮原文。
    assert "用户: 第6轮我还是咳嗽" in history
    assert "用户: 第5轮我还是咳嗽" in history
    assert "用户: 第4轮我还是咳嗽" in history
    assert "用户: 第1轮我还是咳嗽" not in history


def test_m2_and_m3_blocks_present(web_module) -> None:
    case = MEMORY_CASES["memory_002"]
    session = _build_session_from_case(web_module, case)
    probe_query = str(case.get("probe_query", ""))
    history = web_module._build_general_history_text(session, query=probe_query)

    assert "长期记忆(M2):" in history
    assert "医疗核心事实(M3):" in history
    assert "过敏史" in history
    assert "慢病/基础病" in history
    assert "当前用药" in history

    rows = web_module._retrieve_general_m2_records(
        session,
        probe_query,
        top_k=3,
        intent_hint="medication_question",
    )
    assert rows
    merged = " ".join(str(row.get("text", "")) for row in rows)
    assert any(key in merged for key in ("过敏", "高血压", "阿司匹林"))


@pytest.mark.parametrize("case", M3_CASES, ids=[str(item.get("id", "m3")) for item in M3_CASES])
def test_m3_core_entity_write_and_recall(web_module, case: dict[str, Any]) -> None:
    session = _build_session_from_case(web_module, case)
    expected_label = str(case.get("expected_label", "")).strip()
    expected_status = str(case.get("expected_status", "confirmed")).strip().lower()
    expected_items = _expected_items(case)
    m3_text = web_module._build_general_m3_text(session)
    history_text = web_module._build_general_history_text(session, query=str(case.get("probe_query", "")))

    assert expected_label
    assert expected_label in m3_text
    assert expected_label in history_text

    if expected_status == "negated":
        assert "用户否认" in m3_text
        assert "用户否认" in history_text
    else:
        assert expected_items
        for expected_item in expected_items:
            assert expected_item in m3_text
            assert expected_item in history_text

    if expected_label == "妊娠/哺乳":
        assert any(word in m3_text for word in ("怀孕", "孕", "哺乳"))
