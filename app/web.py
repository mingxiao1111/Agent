from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from .async_pipeline import run_agent_async_pipeline_sync
from .llm_chains import generate_followups_with_llm, stream_response_with_llm
from .tcm_graph import run_tcm_collect, run_tcm_round, stream_tcm_collect

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"

web_app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

TCM_SESSIONS: dict[str, dict] = {}
GENERAL_SESSIONS: dict[str, dict[str, Any]] = {}


def _to_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        value = int(default)
    return max(minimum, min(maximum, value))


GENERAL_MEMORY_TURNS = _to_int("CHAT_SHORT_MEMORY_TURNS", 6, 2, 20)
GENERAL_MEMORY_MAX_CHARS = _to_int("CHAT_SHORT_MEMORY_MAX_CHARS", 1200, 240, 6000)
VOLCENGINE_ALLOWED_MODELS = {
    "deepseek-v3-2-251201",
    "doubao-seed-2-0-pro-260215",
}


@web_app.get("/")
def index():
    return render_template("index.html")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _serialize_result(result: dict) -> dict:
    return {
        "answer": result.get("answer", ""),
        "intent": result.get("intent", "other"),
        "intent_source": result.get("intent_source", "rule"),
        "risk_level": result.get("risk_level", "low"),
        "confidence": result.get("confidence", 0.0),
        "handoff": bool(result.get("handoff", False)),
        "citations": result.get("citations", []),
        "handoff_summary": result.get("handoff_summary", ""),
        "follow_ups": result.get("follow_ups", []),
        "pipeline_trace": result.get("pipeline_trace", []),
        "llm_runtime": {
            "provider": result.get("llm_provider", "default"),
            "model": result.get("llm_model", ""),
            "thinking": bool(result.get("llm_thinking", False)),
        },
    }


def _parse_llm_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("llm_provider", "default")).strip().lower()
    if provider not in {"default", "volcengine"}:
        provider = "default"

    model = str(payload.get("llm_model", "")).strip()
    if provider == "volcengine":
        if model not in VOLCENGINE_ALLOWED_MODELS:
            model = "deepseek-v3-2-251201"
    else:
        model = ""

    thinking_raw = payload.get("llm_thinking", False)
    if isinstance(thinking_raw, bool):
        thinking = thinking_raw
    elif isinstance(thinking_raw, str):
        thinking = thinking_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        thinking = bool(thinking_raw)

    return {
        "provider": provider,
        "model": model,
        "thinking": bool(thinking) if provider == "volcengine" else False,
    }


def _new_general_session() -> tuple[str, dict[str, Any]]:
    sid = str(uuid.uuid4())
    session = {"turns": []}
    GENERAL_SESSIONS[sid] = session
    return sid, session


def _get_or_create_general_session(session_id: str) -> tuple[str, dict[str, Any]]:
    sid = str(session_id or "").strip()
    if sid and sid in GENERAL_SESSIONS:
        return sid, GENERAL_SESSIONS[sid]
    return _new_general_session()


def _build_general_history_text(session: dict[str, Any]) -> str:
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        return ""
    lines: list[str] = []
    for turn in turns[-GENERAL_MEMORY_TURNS:]:
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("user", "")).strip()
        a = str(turn.get("assistant", "")).strip()
        if q:
            lines.append(f"用户: {q[:160]}")
        if a:
            lines.append(f"助手: {a[:220]}")
    text = "\n".join(lines).strip()
    if len(text) <= GENERAL_MEMORY_MAX_CHARS:
        return text
    return text[-GENERAL_MEMORY_MAX_CHARS:]


def _append_general_turn(session: dict[str, Any], query: str, answer: str) -> None:
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        turns = []
    turns.append(
        {
            "user": str(query or "").strip()[:2000],
            "assistant": str(answer or "").strip()[:4000],
        }
    )
    session["turns"] = turns[-GENERAL_MEMORY_TURNS:]


def _case_refs_preview(items: list[dict[str, Any]] | Any, limit: int = 6) -> list[dict[str, Any]]:
    """给前端展示的医案引用预览：尽量覆盖不同来源。"""

    if not isinstance(items, list) or not items:
        return []

    limit = max(1, min(int(limit or 6), 12))
    refs = [x for x in items if isinstance(x, dict)]
    if not refs:
        return []

    def _norm_source(row: dict[str, Any]) -> str:
        src = str(row.get("source", "")).strip().lower()
        return src if src in {"txt", "md", "docx"} else "other"

    def _score(row: dict[str, Any]) -> float:
        try:
            return float(row.get("score", row.get("keyword_score", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    ranked = sorted(refs, key=_score, reverse=True)
    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str, int]] = set()

    for src in ("txt", "md", "docx", "other"):
        for row in ranked:
            if _norm_source(row) != src:
                continue
            key = (_norm_source(row), str(row.get("file", "")), int(row.get("line_no", 0) or 0))
            if key in used_keys:
                continue
            selected.append(row)
            used_keys.add(key)
            break
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for row in ranked:
            if len(selected) >= limit:
                break
            key = (_norm_source(row), str(row.get("file", "")), int(row.get("line_no", 0) or 0))
            if key in used_keys:
                continue
            selected.append(row)
            used_keys.add(key)

    return selected[:limit]


@web_app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    incoming_session_id = str(payload.get("session_id", "")).strip()
    enable_online_search = _to_bool(payload.get("enable_online_search", False))
    llm_runtime = _parse_llm_runtime(payload)

    if not query:
        return jsonify({"error": "query is required"}), 400

    session_id, session = _get_or_create_general_session(incoming_session_id)
    history_text = _build_general_history_text(session)

    try:
        pipeline_trace: list[dict[str, Any]] = []
        result = run_agent_async_pipeline_sync(
            query,
            llm_provider=llm_runtime["provider"],
            llm_model=llm_runtime["model"],
            llm_thinking=bool(llm_runtime["thinking"]),
            conversation_history_text=history_text,
            enable_online_search=enable_online_search,
            use_cache=_to_bool(payload.get("enable_cache", True)),
            trace=pipeline_trace,
        )
        result = dict(result)
        result["pipeline_trace"] = pipeline_trace
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"agent_failed: {exc}"}), 500

    _append_general_turn(session, query, str(result.get("answer", "")))
    response_payload = _serialize_result(result)
    response_payload["session_id"] = session_id
    return jsonify(response_payload)


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
    )


def _iter_token_like_chunks(text: str) -> Iterator[str]:
    """将上游 chunk 细分为更小片段，提升前端 token-streaming 观感。"""

    raw = str(text or "")
    if not raw:
        return

    cjk_step = _to_int("CHAT_STREAM_CJK_STEP", 2, 1, 8)
    latin_step = _to_int("CHAT_STREAM_LATIN_STEP", 5, 1, 16)
    step = cjk_step if any(_is_cjk_char(ch) for ch in raw) else latin_step

    buf: list[str] = []
    for ch in raw:
        buf.append(ch)
        if ch == "\n":
            piece = "".join(buf)
            if piece:
                yield piece
            buf = []
            continue
        if len(buf) >= step:
            piece = "".join(buf)
            if piece:
                yield piece
            buf = []
    if buf:
        piece = "".join(buf)
        if piece:
            yield piece


@web_app.post("/api/chat/stream")
def chat_stream():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    incoming_session_id = str(payload.get("session_id", "")).strip()
    enable_online_search = _to_bool(payload.get("enable_online_search", False))
    enable_cache = _to_bool(payload.get("enable_cache", True))
    llm_runtime = _parse_llm_runtime(payload)

    if not query:
        return jsonify({"error": "query is required"}), 400

    session_id, session = _get_or_create_general_session(incoming_session_id)
    history_text = _build_general_history_text(session)

    def generate():
        followups_executor: ThreadPoolExecutor | None = None
        followups_future = None
        try:
            pipeline_trace: list[dict[str, Any]] = []
            state = run_agent_async_pipeline_sync(
                query,
                llm_provider=llm_runtime["provider"],
                llm_model=llm_runtime["model"],
                llm_thinking=bool(llm_runtime["thinking"]),
                conversation_history_text=history_text,
                disable_llm_response=True,
                disable_llm_followups=True,
                enable_online_search=enable_online_search,
                use_cache=enable_cache,
                trace=pipeline_trace,
            )
            state = dict(state)
            state["pipeline_trace"] = pipeline_trace
            result = _serialize_result(state)
            rule_answer = result.pop("answer", "")

            for stage_item in pipeline_trace:
                yield _sse("stage", stage_item)

            # 猜你想问与主回复流式生成并行，避免在最后额外等待。
            followups_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream_followups")
            followups_future = followups_executor.submit(
                generate_followups_with_llm,
                query=query,
                intent=state.get("intent", "other"),
                risk_level=state.get("risk_level", "low"),
                conversation_history_text=history_text,
                llm_runtime=llm_runtime,
            )

            stream_iter = stream_response_with_llm(
                query=query,
                intent=state.get("intent", "other"),
                risk_level=state.get("risk_level", "low"),
                tool_results=state.get("tool_results", {}),
                context_docs=state.get("context_docs", []),
                citations=state.get("citations", []),
                handoff=bool(state.get("handoff", False)),
                enable_online_search=enable_online_search,
                conversation_history_text=history_text,
                llm_runtime=llm_runtime,
            )

            sent_any = False
            streamed_parts: list[str] = []
            if stream_iter is not None:
                try:
                    for text in stream_iter:
                        if not text:
                            continue
                        for piece in _iter_token_like_chunks(text):
                            sent_any = True
                            streamed_parts.append(piece)
                            yield _sse("token", {"text": piece})
                except Exception:
                    pass

            if not sent_any and rule_answer:
                streamed_parts = [rule_answer]
                yield _sse("token", {"text": rule_answer})

            final_answer = "".join(streamed_parts).strip() or rule_answer

            if followups_future is not None:
                try:
                    llm_followups = followups_future.result(timeout=0.9)
                except Exception:
                    llm_followups = None
                if llm_followups:
                    result["follow_ups"] = llm_followups

            _append_general_turn(session, query, final_answer)
            result["session_id"] = session_id
            result["llm_provider"] = llm_runtime["provider"]
            result["llm_model"] = llm_runtime["model"]
            result["llm_thinking"] = bool(llm_runtime["thinking"])

            yield _sse("meta", result)
            yield _sse("done", {"ok": True})
        except Exception as exc:  # pragma: no cover
            yield _sse("error", {"error": f"agent_failed: {exc}"})
        finally:
            if followups_executor is not None:
                followups_executor.shutdown(wait=False)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _new_tcm_session(seed_query: str = "") -> tuple[str, dict]:
    sid = str(uuid.uuid4())
    session = {
        "history": [seed_query] if seed_query else [],
        "round": 0,
        "confidence": 0.0,
        "done": False,
        "symptom_profile": {},
        "case_refs": [],
        "candidates": [],
        "questionnaire": [],
        "asked_question_keys": [],
        "answers_history": [],
        "red_flags": [],
        "result": {},
    }
    TCM_SESSIONS[sid] = session
    return sid, session


@web_app.post("/api/tcm/init")
def tcm_init():
    payload = request.get_json(silent=True) or {}
    seed_query = str(payload.get("seed_query", "")).strip()
    sid, _ = _new_tcm_session(seed_query)

    msg = (
        "已进入中医辨证问诊模式。为了提供更准确的诊断，请您尽量补充更详细的症状信息："
    )
    return jsonify({"session_id": sid, "message": msg})


@web_app.post("/api/tcm/collect")
def tcm_collect():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id", "")).strip()
    user_input = str(payload.get("user_input", "")).strip()

    if not session_id or session_id not in TCM_SESSIONS:
        return jsonify({"error": "invalid_session"}), 400
    if not user_input:
        return jsonify({"error": "user_input is required"}), 400

    session = TCM_SESSIONS[session_id]

    collect_state = run_tcm_collect(
        history=session.get("history", []),
        user_input=user_input,
        asked_question_keys=session.get("asked_question_keys", []),
        round_no=int(session.get("round", 0)),
    )

    session["history"] = collect_state.get("history", session.get("history", []))
    session["done"] = bool(collect_state.get("done", False))
    session["confidence"] = float(collect_state.get("confidence", 0.0))
    session["symptom_profile"] = collect_state.get("symptom_profile", {})
    session["case_refs"] = collect_state.get("case_refs", [])
    session["candidates"] = collect_state.get("candidates", [])
    session["questionnaire"] = collect_state.get("questionnaire", [])
    session["asked_question_keys"] = collect_state.get("asked_question_keys", session.get("asked_question_keys", []))
    session["red_flags"] = collect_state.get("red_flags", [])

    return jsonify(
        {
            "done": bool(session.get("done", False)),
            "need_more": bool(collect_state.get("need_more", True)),
            "message": collect_state.get("message", "请继续补充症状。"),
            "round": int(session.get("round", 0)),
            "confidence": float(session.get("confidence", 0.0)),
            "symptom_profile": session.get("symptom_profile", {}),
            "extraction_ok": bool(session.get("symptom_profile", {}).get("extraction_ok", True)),
            "extraction_error": str(session.get("symptom_profile", {}).get("extraction_error", "")),
            "llm_invoked": bool(session.get("symptom_profile", {}).get("llm_invoked", False)),
            "candidates": session.get("candidates", []),
            "questionnaire": session.get("questionnaire", []),
            "case_refs": _case_refs_preview(session.get("case_refs", []), limit=6),
            "red_flags": session.get("red_flags", []),
        }
    )


@web_app.post("/api/tcm/collect/stream")
def tcm_collect_stream():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id", "")).strip()
    user_input = str(payload.get("user_input", "")).strip()

    if not session_id or session_id not in TCM_SESSIONS:
        return jsonify({"error": "invalid_session"}), 400
    if not user_input:
        return jsonify({"error": "user_input is required"}), 400

    session = TCM_SESSIONS[session_id]

    def generate():
        try:
            final_state: dict[str, Any] = {}
            for item in stream_tcm_collect(
                history=session.get("history", []),
                user_input=user_input,
                asked_question_keys=session.get("asked_question_keys", []),
                round_no=int(session.get("round", 0)),
            ):
                event = str(item.get("event", "stage"))
                data = item.get("data", {})
                if event == "stage":
                    yield _sse("stage", data if isinstance(data, dict) else {"text": str(data)})
                    continue
                if event == "analysis":
                    yield _sse("analysis", data if isinstance(data, dict) else {"message": str(data)})
                    continue
                if event == "result":
                    final_state = data if isinstance(data, dict) else {}
                    break

            if not final_state:
                raise ValueError("tcm_collect_stream_empty_result")

            session["history"] = final_state.get("history", session.get("history", []))
            session["done"] = bool(final_state.get("done", False))
            session["confidence"] = float(final_state.get("confidence", 0.0))
            session["symptom_profile"] = final_state.get("symptom_profile", {})
            session["case_refs"] = final_state.get("case_refs", [])
            session["candidates"] = final_state.get("candidates", [])
            session["questionnaire"] = final_state.get("questionnaire", [])
            session["asked_question_keys"] = final_state.get("asked_question_keys", session.get("asked_question_keys", []))
            session["red_flags"] = final_state.get("red_flags", [])

            result_payload = {
                "done": bool(session.get("done", False)),
                "need_more": bool(final_state.get("need_more", True)),
                "message": final_state.get("message", "请继续补充症状。"),
                "round": int(session.get("round", 0)),
                "confidence": float(session.get("confidence", 0.0)),
                "symptom_profile": session.get("symptom_profile", {}),
                "extraction_ok": bool(session.get("symptom_profile", {}).get("extraction_ok", True)),
                "extraction_error": str(session.get("symptom_profile", {}).get("extraction_error", "")),
                "llm_invoked": bool(session.get("symptom_profile", {}).get("llm_invoked", False)),
                "candidates": session.get("candidates", []),
                "questionnaire": session.get("questionnaire", []),
                "case_refs": _case_refs_preview(session.get("case_refs", []), limit=6),
                "red_flags": session.get("red_flags", []),
            }
            yield _sse("result", result_payload)
            yield _sse("done", {"ok": True})
        except Exception as exc:  # pragma: no cover
            yield _sse("error", {"error": f"tcm_collect_stream_failed: {exc}"})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@web_app.post("/api/tcm/questionnaire")
def tcm_questionnaire_submit():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id", "")).strip()
    answers = payload.get("answers", {})

    if not session_id or session_id not in TCM_SESSIONS:
        return jsonify({"error": "invalid_session"}), 400
    if not isinstance(answers, dict) or not answers:
        return jsonify({"error": "answers are required"}), 400

    session = TCM_SESSIONS[session_id]

    normalized_answers: dict[str, Any] = {}
    for key, value in answers.items():
        qid = str(key).strip()
        if not qid:
            continue
        normalized_answers[qid] = value

    if not normalized_answers:
        return jsonify({"error": "valid answers are required"}), 400

    round_state = run_tcm_round(
        history=session.get("history", []),
        round_no=int(session.get("round", 0)),
        asked_question_keys=session.get("asked_question_keys", []),
        answers_history=session.get("answers_history", []),
        symptom_profile=session.get("symptom_profile", {}),
        candidates=session.get("candidates", []),
        questionnaire=session.get("questionnaire", []),
        answers=normalized_answers,
        case_refs=session.get("case_refs", []),
    )

    session["history"] = round_state.get("history", session.get("history", []))
    session["round"] = int(round_state.get("round", session.get("round", 0)))
    session["confidence"] = float(round_state.get("confidence", session.get("confidence", 0.0)))
    session["done"] = bool(round_state.get("done", False))
    session["symptom_profile"] = round_state.get("symptom_profile", session.get("symptom_profile", {}))
    session["case_refs"] = round_state.get("case_refs", session.get("case_refs", []))
    session["candidates"] = round_state.get("candidates", session.get("candidates", []))
    session["questionnaire"] = round_state.get("questionnaire", [])
    session["answers_history"] = round_state.get("answers_history", session.get("answers_history", []))
    session["asked_question_keys"] = round_state.get("asked_question_keys", session.get("asked_question_keys", []))
    session["red_flags"] = round_state.get("red_flags", session.get("red_flags", []))
    session["result"] = round_state.get("result", session.get("result", {}))

    follow_ups = []
    if session.get("done", False):
        follow_ups = [
            "退出中医辨证模式，回到普通咨询",
            "继续补充症状，重新开始辨证",
        ]
    else:
        follow_ups = [
            "继续完成下一轮问卷",
            "补充更多症状后再做一轮",
            "退出中医辨证模式，回到普通咨询",
        ]

    return jsonify(
        {
            "done": bool(session.get("done", False)),
            "message": round_state.get("message", "阶段性辨证完成。"),
            "round": int(session.get("round", 0)),
            "confidence": float(session.get("confidence", 0.0)),
            "result": session.get("result", {}),
            "symptom_profile": session.get("symptom_profile", {}),
            "extraction_ok": bool(session.get("symptom_profile", {}).get("extraction_ok", True)),
            "extraction_error": str(session.get("symptom_profile", {}).get("extraction_error", "")),
            "llm_invoked": bool(session.get("symptom_profile", {}).get("llm_invoked", False)),
            "candidates": session.get("candidates", []),
            "questionnaire": session.get("questionnaire", []),
            "case_refs": _case_refs_preview(session.get("case_refs", []), limit=6),
            "red_flags": session.get("red_flags", []),
            "follow_ups": follow_ups,
        }
    )


def main() -> None:
    load_dotenv()
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8000"))
    web_app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
