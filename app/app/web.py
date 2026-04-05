from __future__ import annotations

import json
import hashlib
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from .async_pipeline import run_agent_async_pipeline_sync
from .llm_chains import generate_followups_with_llm, generate_memory_fact_with_llm, stream_response_with_llm
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
CHAT_FEEDBACK_DIR = BASE_DIR / "data" / "chat_feedback"


def _to_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        value = int(default)
    return max(minimum, min(maximum, value))


def _to_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, str(default))).strip())
    except ValueError:
        value = float(default)
    return max(minimum, min(maximum, value))


GENERAL_MEMORY_MAX_CHARS = _to_int("CHAT_SHORT_MEMORY_MAX_CHARS", 2200, 480, 12000)
GENERAL_M0_TURNS = _to_int("CHAT_MEMORY_M0_TURNS", 3, 2, 8)
GENERAL_M1_MAX_CHARS = _to_int("CHAT_MEMORY_M1_MAX_CHARS", 1600, 320, 8000)
GENERAL_MAX_SEGMENTS = _to_int("CHAT_MEMORY_MAX_SEGMENTS", 12, 4, 48)
GENERAL_SEGMENT_SWITCH_STREAK = _to_int("CHAT_MEMORY_SEGMENT_SWITCH_STREAK", 2, 1, 4)
GENERAL_SEGMENT_SIM_THRESHOLD = _to_float("CHAT_MEMORY_SEGMENT_SIM_THRESHOLD", 0.55, 0.2, 0.95)
GENERAL_SEGMENT_INTENT_CONF_THRESHOLD = _to_float("CHAT_MEMORY_INTENT_CONF_THRESHOLD", 0.75, 0.2, 0.99)
GENERAL_TURN_USER_MAX = _to_int("CHAT_MEMORY_TURN_USER_MAX", 2000, 300, 6000)
GENERAL_TURN_ASSISTANT_MAX = _to_int("CHAT_MEMORY_TURN_ASSISTANT_MAX", 4000, 500, 12000)
GENERAL_SUMMARY_USER_MAX = _to_int("CHAT_MEMORY_SUMMARY_USER_MAX", 80, 20, 240)
GENERAL_SUMMARY_ASSISTANT_MAX = _to_int("CHAT_MEMORY_SUMMARY_ASSISTANT_MAX", 110, 30, 320)
GENERAL_M3_MAX_ITEMS = _to_int("CHAT_MEMORY_M3_MAX_ITEMS", 6, 2, 12)
GENERAL_M3_ITEM_MAX_CHARS = _to_int("CHAT_MEMORY_M3_ITEM_MAX_CHARS", 36, 10, 120)
GENERAL_M3_BLOCK_MAX_CHARS = _to_int("CHAT_MEMORY_M3_BLOCK_MAX_CHARS", 520, 120, 2400)
GENERAL_M2_ENABLED = str(os.getenv("CHAT_MEMORY_M2_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
GENERAL_M2_VECTOR_DIM = _to_int("CHAT_MEMORY_M2_VECTOR_DIM", 192, 64, 768)
GENERAL_M2_MAX_RECORDS = _to_int("CHAT_MEMORY_M2_MAX_RECORDS", 80, 20, 300)
GENERAL_M2_BOOTSTRAP_LIMIT = _to_int("CHAT_MEMORY_M2_BOOTSTRAP_LIMIT", 120, 20, 600)
GENERAL_M2_RETRIEVE_TOPK = _to_int("CHAT_MEMORY_M2_RETRIEVE_TOPK", 3, 1, 8)
GENERAL_M2_MIN_SIM = _to_float("CHAT_MEMORY_M2_MIN_SIM", 0.12, 0.05, 0.95)
GENERAL_M2_FACT_MAX_CHARS = _to_int("CHAT_MEMORY_M2_FACT_MAX_CHARS", 150, 40, 400)
GENERAL_M2_BLOCK_MAX_CHARS = _to_int("CHAT_MEMORY_M2_BLOCK_MAX_CHARS", 420, 100, 1800)
GENERAL_M2_LLM_CLEAN_ENABLED = str(os.getenv("CHAT_MEMORY_M2_LLM_CLEAN_ENABLED", "false")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GENERAL_M2_LLM_MIN_SALIENCE = _to_float("CHAT_MEMORY_M2_LLM_MIN_SALIENCE", 0.45, 0.1, 0.95)
GENERAL_M2_LLM_FAIL_STREAK_LIMIT = _to_int("CHAT_MEMORY_M2_LLM_FAIL_STREAK_LIMIT", 3, 1, 8)
GENERAL_M2_LLM_COOLDOWN_SEC = _to_int("CHAT_MEMORY_M2_LLM_COOLDOWN_SEC", 300, 30, 7200)
GENERAL_M2_LLM_MAX_QUERY_CHARS = _to_int("CHAT_MEMORY_M2_LLM_MAX_QUERY_CHARS", 300, 80, 800)
GENERAL_M2_INTENT_SAME_BONUS = _to_float("CHAT_MEMORY_M2_INTENT_SAME_BONUS", 0.14, 0.0, 0.6)
GENERAL_M2_INTENT_CROSS_PENALTY = _to_float("CHAT_MEMORY_M2_INTENT_CROSS_PENALTY", 0.08, 0.0, 0.6)
GENERAL_M2_CROSS_INTENT_MIN_SIM = _to_float("CHAT_MEMORY_M2_CROSS_INTENT_MIN_SIM", 0.18, 0.05, 0.95)
GENERAL_M2_MARKERS = (
    "我有",
    "我对",
    "我在",
    "我曾",
    "我一直",
    "过敏",
    "慢性病",
    "基础病",
    "既往",
    "手术",
    "住院",
    "家族史",
    "怀孕",
    "哺乳",
    "服用",
    "在吃",
    "用药",
    "血压",
    "血糖",
    "体温",
    "检查",
    "报告",
)
EXPLICIT_TOPIC_SWITCH_MARKERS = (
    "换个问题",
    "另外一个问题",
    "另外我想问",
    "另一个症状",
    "再问一个",
    "顺便问下",
)
VOLCENGINE_ALLOWED_MODELS = {
    "deepseek-v3-2-251201",
    "doubao-seed-2-0-pro-260215",
}
M3_BUCKET_ORDER = (
    "allergy",
    "chronic_disease",
    "medication",
    "pregnancy_lactation",
    "surgery_history",
    "family_history",
)
M3_BUCKET_LABELS = {
    "allergy": "过敏史",
    "chronic_disease": "慢病/基础病",
    "medication": "当前用药",
    "pregnancy_lactation": "妊娠/哺乳",
    "surgery_history": "手术/住院史",
    "family_history": "家族史",
}
M3_DISEASE_HINTS = (
    "高血压",
    "糖尿病",
    "冠心病",
    "脑梗",
    "脑卒中",
    "心衰",
    "哮喘",
    "慢阻肺",
    "乙肝",
    "丙肝",
    "肝硬化",
    "肾病",
    "肾炎",
    "甲亢",
    "甲减",
    "痛风",
    "类风湿",
    "肿瘤",
    "癌",
    "抑郁",
    "焦虑",
)
M3_MEDICATION_HINTS = (
    "阿司匹林",
    "二甲双胍",
    "胰岛素",
    "华法林",
    "氯吡格雷",
    "硝酸甘油",
    "美托洛尔",
    "氨氯地平",
    "缬沙坦",
    "瑞舒伐他汀",
    "阿托伐他汀",
    "布洛芬",
    "对乙酰氨基酚",
    "头孢",
    "阿莫西林",
)
M3_RELATION_HINTS = ("父亲", "母亲", "父母", "兄弟", "姐妹", "家族", "家里人")
GENERAL_LONG_MEMORY_FILE = BASE_DIR / "data" / "general_long_memory.jsonl"
GENERAL_LONG_MEMORY_FILE_LOCK = threading.Lock()
GENERAL_M2_LLM_STATE_LOCK = threading.Lock()
GENERAL_M2_LLM_STATE = {"fail_streak": 0, "disabled_until_ts": 0.0}


@web_app.get("/")
def index():
    return render_template("index.html")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clean_feedback_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit]


def _serialize_result(result: dict) -> dict:
    return {
        "answer": result.get("answer", ""),
        "intent": result.get("intent", "other"),
        "secondary_intents": result.get("secondary_intents", []),
        "intent_candidates": result.get("intent_candidates", []),
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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clean_turn_text(text: object, limit: int) -> str:
    merged = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(merged) <= limit:
        return merged
    return f"{merged[:max(0, limit - 1)]}…"


def _new_m2_state() -> dict[str, Any]:
    return {
        "records": [],
        "loaded_from_disk": False,
        "updated_at": "",
    }


def _ensure_general_m2_shape(session: dict[str, Any]) -> dict[str, Any]:
    state = session.get("m2")
    if not isinstance(state, dict):
        state = _new_m2_state()
        session["m2"] = state

    records = state.get("records", [])
    if not isinstance(records, list):
        records = []
    clean_records: list[dict[str, Any]] = []
    for row in records[-GENERAL_M2_MAX_RECORDS:]:
        if not isinstance(row, dict):
            continue
        text = _clean_turn_text(row.get("text", ""), GENERAL_M2_FACT_MAX_CHARS)
        if not text:
            continue
        vector = row.get("vector", [])
        if not isinstance(vector, list):
            vector = []
        clean_records.append(
            {
                "id": str(row.get("id", "")).strip() or uuid.uuid4().hex,
                "session_id": str(row.get("session_id", "")).strip(),
                "intent": str(row.get("intent", "")).strip(),
                "text": text,
                "vector": [_safe_float(x, 0.0) for x in vector[:GENERAL_M2_VECTOR_DIM]] if vector else [],
                "created_at": str(row.get("created_at", "")).strip(),
            }
        )
    state["records"] = clean_records[-GENERAL_M2_MAX_RECORDS:]
    state["loaded_from_disk"] = bool(state.get("loaded_from_disk", False))
    state["updated_at"] = str(state.get("updated_at", "")).strip()
    session["m2"] = state
    return state


def _m2_hash_index(token: str, dim: int) -> int:
    raw = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(raw, byteorder="big", signed=False) % max(1, int(dim))


def _m2_vectorize(text: object, dim: int = GENERAL_M2_VECTOR_DIM) -> list[float]:
    grams = _char_ngrams(str(text or ""), n=2)
    if not grams:
        return []
    vec = [0.0] * max(1, int(dim))
    for gram in grams:
        idx = _m2_hash_index(gram, len(vec))
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-8:
        return []
    return [v / norm for v in vec]


def _m2_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    size = min(len(vec_a), len(vec_b))
    return sum(float(vec_a[i]) * float(vec_b[i]) for i in range(size))


def _m2_disk_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(record.get("id", "")).strip() or uuid.uuid4().hex,
        "session_id": str(record.get("session_id", "")).strip(),
        "intent": str(record.get("intent", "")).strip(),
        "text": _clean_turn_text(record.get("text", ""), GENERAL_M2_FACT_MAX_CHARS),
        "created_at": str(record.get("created_at", "")).strip() or datetime.now().isoformat(timespec="seconds"),
    }


def _append_m2_record_to_disk(record: dict[str, Any]) -> None:
    try:
        row = _m2_disk_record(record)
        if not row.get("text"):
            return
        GENERAL_LONG_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with GENERAL_LONG_MEMORY_FILE_LOCK:
            with GENERAL_LONG_MEMORY_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        return


def _load_m2_records_from_disk(session_id: str, limit: int = GENERAL_M2_BOOTSTRAP_LIMIT) -> list[dict[str, Any]]:
    sid = str(session_id or "").strip()
    if not sid:
        return []
    if not GENERAL_LONG_MEMORY_FILE.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with GENERAL_LONG_MEMORY_FILE_LOCK:
            lines = GENERAL_LONG_MEMORY_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    for line in reversed(lines):
        if len(rows) >= max(1, int(limit)):
            break
        text = str(line or "").strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if str(row.get("session_id", "")).strip() != sid:
            continue
        fact_text = _clean_turn_text(row.get("text", ""), GENERAL_M2_FACT_MAX_CHARS)
        if not fact_text:
            continue
        rows.append(
            {
                "id": str(row.get("id", "")).strip() or uuid.uuid4().hex,
                "session_id": sid,
                "intent": str(row.get("intent", "")).strip(),
                "text": fact_text,
                "vector": _m2_vectorize(fact_text),
                "created_at": str(row.get("created_at", "")).strip(),
            }
        )
    rows.reverse()
    return rows[-GENERAL_M2_MAX_RECORDS:]


def _normalize_m2_fact(text: object) -> str:
    merged = _clean_turn_text(text, GENERAL_M2_FACT_MAX_CHARS)
    merged = re.sub(r"[。；;]+$", "", merged).strip()
    return merged


def _normalize_intent_key(intent: object) -> str:
    key = str(intent or "").strip().lower()
    if not key or key == "other":
        return ""
    return key


def _is_m2_candidate_query(query: str) -> bool:
    text = str(query or "").strip()
    if len(text) < 6:
        return False
    if any(mark in text for mark in GENERAL_M2_MARKERS):
        return True
    if re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", text):
        return True
    if re.search(r"(?:\d+(\.\d+)?)(?:天|周|月|年)", text):
        return True
    return False


def _select_m2_fact(query: str, m3_events: list[str], intent: str) -> str:
    cleaned_query = _normalize_m2_fact(query)
    if m3_events:
        joined = "；".join(str(x) for x in m3_events if str(x).strip())
        merged = f"用户医疗事实更新: {joined}"
        if cleaned_query and cleaned_query not in joined:
            merged = f"{merged}（原话: {cleaned_query}）"
        return _normalize_m2_fact(merged)
    if not _is_m2_candidate_query(cleaned_query):
        return ""
    if re.search(r"(?:^|[，。；\s])(怎么|如何|为什么|可以吗|能不能|是否|吗|嘛|\?)", cleaned_query) and "我" not in cleaned_query:
        return ""
    return cleaned_query


def _m2_llm_clean_allowed() -> bool:
    if not GENERAL_M2_ENABLED or not GENERAL_M2_LLM_CLEAN_ENABLED:
        return False
    now = time.time()
    with GENERAL_M2_LLM_STATE_LOCK:
        disabled_until = _safe_float(GENERAL_M2_LLM_STATE.get("disabled_until_ts", 0.0), 0.0)
        if disabled_until > now:
            return False
    return True


def _record_m2_llm_success() -> None:
    with GENERAL_M2_LLM_STATE_LOCK:
        GENERAL_M2_LLM_STATE["fail_streak"] = 0
        GENERAL_M2_LLM_STATE["disabled_until_ts"] = 0.0


def _record_m2_llm_failure() -> None:
    now = time.time()
    with GENERAL_M2_LLM_STATE_LOCK:
        next_streak = int(_safe_float(GENERAL_M2_LLM_STATE.get("fail_streak", 0), 0)) + 1
        GENERAL_M2_LLM_STATE["fail_streak"] = next_streak
        if next_streak >= GENERAL_M2_LLM_FAIL_STREAK_LIMIT:
            GENERAL_M2_LLM_STATE["disabled_until_ts"] = now + float(GENERAL_M2_LLM_COOLDOWN_SEC)
            GENERAL_M2_LLM_STATE["fail_streak"] = 0


def _clean_m2_fact_with_llm(
    *,
    query: str,
    intent: str,
    rule_fact: str,
    m3_events: list[str],
    llm_runtime: dict[str, Any] | None = None,
) -> str:
    baseline = _normalize_m2_fact(rule_fact)
    if not baseline:
        return ""
    if not _m2_llm_clean_allowed():
        return baseline

    query_for_llm = _clean_turn_text(query, GENERAL_M2_LLM_MAX_QUERY_CHARS)
    if not query_for_llm:
        return baseline

    result = generate_memory_fact_with_llm(
        query=query_for_llm,
        intent=str(intent or "").strip() or "other",
        rule_fact=baseline,
        m3_events=m3_events,
        llm_runtime=llm_runtime,
    )
    if not isinstance(result, dict):
        _record_m2_llm_failure()
        return baseline

    _record_m2_llm_success()
    cleaned = _normalize_m2_fact(result.get("fact", ""))
    salience = _safe_float(result.get("salience", 0.0), 0.0)
    is_profile_fact = _to_bool(result.get("is_profile_fact", False))
    if cleaned and is_profile_fact and salience >= GENERAL_M2_LLM_MIN_SALIENCE:
        return cleaned
    return baseline


def _upsert_general_m2(session: dict[str, Any], fact_text: str, *, intent: str = "") -> None:
    if not GENERAL_M2_ENABLED:
        return
    normalized = _normalize_m2_fact(fact_text)
    if not normalized:
        return

    state = _ensure_general_m2_shape(session)
    records = state.get("records", [])
    if not isinstance(records, list):
        records = []

    candidate_vec = _m2_vectorize(normalized)
    for row in reversed(records[-8:]):
        if not isinstance(row, dict):
            continue
        old_text = str(row.get("text", "")).strip()
        if not old_text:
            continue
        if old_text == normalized:
            return
        old_vec = row.get("vector", [])
        if not isinstance(old_vec, list) or not old_vec:
            old_vec = _m2_vectorize(old_text)
        if _m2_similarity(candidate_vec, old_vec) >= 0.96:
            return

    session_id = str(session.get("session_id", "")).strip()
    if not session_id:
        session_id = uuid.uuid4().hex
        session["session_id"] = session_id
    row = {
        "id": uuid.uuid4().hex,
        "session_id": session_id,
        "intent": str(intent or "").strip(),
        "text": normalized,
        "vector": candidate_vec,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    records.append(row)
    records = records[-GENERAL_M2_MAX_RECORDS:]
    state["records"] = records
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    session["m2"] = state
    _append_m2_record_to_disk(row)


def _retrieve_general_m2_records(
    session: dict[str, Any],
    query: str,
    top_k: int = GENERAL_M2_RETRIEVE_TOPK,
    *,
    intent_hint: str = "",
) -> list[dict[str, Any]]:
    if not GENERAL_M2_ENABLED:
        return []

    state = _ensure_general_m2_shape(session)
    if not bool(state.get("loaded_from_disk", False)):
        loaded = _load_m2_records_from_disk(str(session.get("session_id", "")).strip(), limit=GENERAL_M2_BOOTSTRAP_LIMIT)
        current = state.get("records", [])
        if not isinstance(current, list):
            current = []
        all_rows = loaded + [x for x in current if isinstance(x, dict)]
        dedup: dict[str, dict[str, Any]] = {}
        for row in all_rows:
            text = _normalize_m2_fact(row.get("text", ""))
            if not text:
                continue
            dedup[text] = {
                "id": str(row.get("id", "")).strip() or uuid.uuid4().hex,
                "session_id": str(row.get("session_id", "")).strip(),
                "intent": str(row.get("intent", "")).strip(),
                "text": text,
                "vector": row.get("vector", []) if isinstance(row.get("vector", []), list) else [],
                "created_at": str(row.get("created_at", "")).strip(),
            }
        merged = list(dedup.values())[-GENERAL_M2_MAX_RECORDS:]
        state["records"] = merged
        state["loaded_from_disk"] = True
        session["m2"] = state

    records = state.get("records", [])
    if not isinstance(records, list) or not records:
        return []

    query_vec = _m2_vectorize(query)
    if not query_vec:
        return []

    hint_key = _normalize_intent_key(intent_hint)
    same_intent_scored: list[tuple[float, int, dict[str, Any]]] = []
    cross_intent_scored: list[tuple[float, int, dict[str, Any]]] = []
    neutral_scored: list[tuple[float, int, dict[str, Any]]] = []
    total = max(1, len(records))
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        vec = row.get("vector", [])
        if not isinstance(vec, list) or not vec:
            vec = _m2_vectorize(text)
            row["vector"] = vec
        sim = _m2_similarity(query_vec, vec)
        if sim < GENERAL_M2_MIN_SIM:
            continue
        recency_boost = 0.05 * ((idx + 1) / total)
        row_intent_key = _normalize_intent_key(row.get("intent", ""))

        if hint_key and row_intent_key:
            if row_intent_key == hint_key:
                score = sim + recency_boost + GENERAL_M2_INTENT_SAME_BONUS
                same_intent_scored.append((score, idx, row))
            else:
                if sim < GENERAL_M2_CROSS_INTENT_MIN_SIM:
                    continue
                score = sim + recency_boost - GENERAL_M2_INTENT_CROSS_PENALTY
                cross_intent_scored.append((score, idx, row))
        else:
            score = sim + recency_boost
            neutral_scored.append((score, idx, row))

    if not same_intent_scored and not cross_intent_scored and not neutral_scored:
        return []
    same_intent_scored.sort(key=lambda x: x[0], reverse=True)
    cross_intent_scored.sort(key=lambda x: x[0], reverse=True)
    neutral_scored.sort(key=lambda x: x[0], reverse=True)

    picked: list[dict[str, Any]] = []
    for _, _, row in same_intent_scored:
        picked.append(row)
        if len(picked) >= max(1, int(top_k)):
            return picked
    for _, _, row in neutral_scored:
        picked.append(row)
        if len(picked) >= max(1, int(top_k)):
            return picked
    for _, _, row in cross_intent_scored:
        picked.append(row)
        if len(picked) >= max(1, int(top_k)):
            return picked
    return picked


def _build_general_m2_text(session: dict[str, Any], query: str, *, intent_hint: str = "") -> str:
    rows = _retrieve_general_m2_records(
        session,
        query,
        top_k=GENERAL_M2_RETRIEVE_TOPK,
        intent_hint=intent_hint,
    )
    if not rows:
        return ""
    lines = ["长期记忆(M2):"]
    for row in rows:
        text = _clean_turn_text(row.get("text", ""), GENERAL_M2_FACT_MAX_CHARS)
        if not text:
            continue
        lines.append(f"- {text}")
    block = "\n".join(lines).strip()
    if len(block) > GENERAL_M2_BLOCK_MAX_CHARS:
        return "..." + block[-GENERAL_M2_BLOCK_MAX_CHARS:]
    return block


def _new_m3_bucket() -> dict[str, Any]:
    return {
        "status": "unknown",  # unknown / confirmed / negated
        "items": [],
        "note": "",
        "updated_at": "",
    }


def _new_m3_profile() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "entities": {name: _new_m3_bucket() for name in M3_BUCKET_ORDER},
        "updated_at": "",
    }


def _ensure_general_m3_shape(session: dict[str, Any]) -> dict[str, Any]:
    profile = session.get("m3")
    if not isinstance(profile, dict):
        profile = _new_m3_profile()
        session["m3"] = profile

    entities = profile.get("entities")
    if not isinstance(entities, dict):
        entities = {}
    for name in M3_BUCKET_ORDER:
        bucket = entities.get(name)
        if not isinstance(bucket, dict):
            entities[name] = _new_m3_bucket()
            continue
        status = str(bucket.get("status", "unknown")).strip().lower()
        if status not in {"unknown", "confirmed", "negated"}:
            status = "unknown"
        items = bucket.get("items", [])
        if not isinstance(items, list):
            items = []
        clean_items: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = _clean_turn_text(item, GENERAL_M3_ITEM_MAX_CHARS)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            clean_items.append(normalized)
            if len(clean_items) >= GENERAL_M3_MAX_ITEMS:
                break
        entities[name] = {
            "status": status,
            "items": clean_items,
            "note": _clean_turn_text(bucket.get("note", ""), 120),
            "updated_at": str(bucket.get("updated_at", "")).strip(),
        }
    profile["entities"] = entities
    profile["schema_version"] = 1
    profile["updated_at"] = str(profile.get("updated_at", "")).strip()
    session["m3"] = profile
    return profile


def _normalize_m3_item(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.strip("，。；;:：,、")
    text = re.sub(
        r"^(?:我(?:现在|目前|正在)?|目前|现在|正在|有|患有|既往|家族史|过敏史|手术史|住院史|用药情况)[:：]?",
        "",
        text,
    ).strip()
    if not text:
        return ""
    if len(text) > GENERAL_M3_ITEM_MAX_CHARS:
        return text[:GENERAL_M3_ITEM_MAX_CHARS]
    return text


def _split_m3_items(raw: object, limit: int = GENERAL_M3_MAX_ITEMS) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    compact = text.replace("；", "，").replace(";", "，")
    parts = re.split(r"[，,、/|+＋和及与]", compact)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = _normalize_m3_item(part)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _is_negated_near(text: str, token: str, window: int = 8) -> bool:
    if not token:
        return False
    for matched in re.finditer(re.escape(token), text):
        start = max(0, matched.start() - window)
        left = text[start : matched.start()]
        if re.search(r"(无|没有|未|并未|否认|不)", left):
            return True
    return False


def _has_first_person_near(text: str, token: str, window: int = 8) -> bool:
    if not token:
        return False
    for matched in re.finditer(re.escape(token), text):
        start = max(0, matched.start() - window)
        left = text[start : matched.start()]
        if any(mark in left for mark in ("我", "本人", "自己")):
            return True
    return False


def _extract_allergy_entities(text: str) -> tuple[list[str], bool]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return [], False
    if re.search(r"(无|没有|未见|否认)(?:药物|食物)?过敏|不过敏", compact):
        return [], True

    items: list[str] = []
    for pattern in (
        r"对([^，。；;\n]{1,20}?)过敏",
        r"(?:过敏史|药物过敏|食物过敏)[:：]?([^，。；;\n]{1,32})",
        r"过敏[:：]?([^，。；;\n]{1,24})",
    ):
        for matched in re.finditer(pattern, compact):
            chunk = matched.group(1)
            for item in _split_m3_items(chunk):
                if any(word in item for word in ("怎么办", "怎么", "如何", "吗", "？", "?")):
                    continue
                items.append(item)
    dedup: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_m3_item(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
        if len(dedup) >= GENERAL_M3_MAX_ITEMS:
            break
    return dedup, False


def _extract_chronic_entities(text: str) -> tuple[list[str], bool]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return [], False
    if re.search(r"(无|没有|否认)(?:慢性病|基础病|既往病史)|既往体健", compact):
        return [], True

    items: list[str] = []
    for pattern in (
        r"(?:慢性病|基础病|既往病史|既往史)[:：]?([^，。；;\n]{1,40})",
        r"(?:我有|本人有|一直有|患有)([^，。；;\n]{1,28})",
    ):
        for matched in re.finditer(pattern, compact):
            items.extend(_split_m3_items(matched.group(1)))

    for disease in M3_DISEASE_HINTS:
        if disease not in compact:
            continue
        if _is_negated_near(compact, disease):
            continue
        if _has_first_person_near(compact, disease) or any(
            marker in compact for marker in ("慢性病", "基础病", "既往病史", "既往史", "还有", "伴有", "并有", "合并")
        ):
            items.append(disease)

    dedup: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_m3_item(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
        if len(dedup) >= GENERAL_M3_MAX_ITEMS:
            break
    return dedup, False


def _extract_medication_entities(text: str) -> tuple[list[str], bool]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return [], False
    if re.search(r"(目前|现在)?(?:没有|未|不)(?:在)?(?:服用|吃|用)(?:药|任何药)", compact):
        return [], True

    items: list[str] = []
    for pattern in (
        r"(?:我|本人)?(?:目前|现在|正在|平时)?(?:在吃|服用|长期服用|长期吃|一直在吃|用)([^，。；;\n]{1,36})",
        r"(?:用药情况|目前用药|服药情况)[:：]?([^，。；;\n]{1,36})",
    ):
        for matched in re.finditer(pattern, compact):
            items.extend(_split_m3_items(matched.group(1)))

    for drug in M3_MEDICATION_HINTS:
        if drug not in compact:
            continue
        if _is_negated_near(compact, drug):
            continue
        if _has_first_person_near(compact, drug) or any(mark in compact for mark in ("服用", "在吃", "用药", "长期")):
            items.append(drug)

    dedup: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_m3_item(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
        if len(dedup) >= GENERAL_M3_MAX_ITEMS:
            break
    return dedup, False


def _extract_pregnancy_lactation_entities(text: str) -> tuple[list[str], bool]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return [], False

    items: list[str] = []
    preg_neg = bool(re.search(r"(未|没有|并未|否认)(?:怀孕|妊娠)|非孕|不是孕妇", compact))
    lact_neg = bool(re.search(r"(不|未|没有)(?:在)?哺乳|已断奶|非哺乳期", compact))

    for matched in re.finditer(r"(怀孕\d{1,2}周|孕\d{1,2}周|怀孕|妊娠|备孕|孕期)", compact):
        items.append(matched.group(1))
    for matched in re.finditer(r"(哺乳期|哺乳中|正在哺乳|母乳喂养|喂奶中)", compact):
        items.append(matched.group(1))

    if preg_neg:
        items.append("非孕")
    if lact_neg:
        items.append("非哺乳期")

    dedup: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_m3_item(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
        if len(dedup) >= GENERAL_M3_MAX_ITEMS:
            break
    return dedup, False


def _extract_surgery_entities(text: str) -> tuple[list[str], bool]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return [], False
    if re.search(r"(无|没有|否认)(?:手术史|住院史)|未手术", compact):
        return [], True

    items: list[str] = []
    for pattern in (
        r"(?:手术史|住院史)[:：]?([^，。；;\n]{1,36})",
        r"(?:我|本人|既往|曾|之前)([^，。；;\n]{0,20}(?:手术|住院|切除|置换|剖宫产|支架))",
    ):
        for matched in re.finditer(pattern, compact):
            items.extend(_split_m3_items(matched.group(1)))

    dedup: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_m3_item(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
        if len(dedup) >= GENERAL_M3_MAX_ITEMS:
            break
    return dedup, False


def _extract_family_entities(text: str) -> tuple[list[str], bool]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return [], False
    if re.search(r"(无|没有|否认)家族史|家族中无", compact):
        return [], True

    items: list[str] = []
    for pattern in (
        r"(?:家族史|家族中|家里人)[:：]?([^。；;\n]{1,40})",
        r"(父亲|母亲|父母|兄弟|姐妹|家族)([^，。；;\n]{0,18})",
    ):
        for matched in re.finditer(pattern, compact):
            merged = "".join(str(x) for x in matched.groups() if x)
            candidates = _split_m3_items(merged)
            for item in candidates:
                if any(rel in item for rel in M3_RELATION_HINTS) or any(d in item for d in M3_DISEASE_HINTS):
                    items.append(item)

    dedup: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_m3_item(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
        if len(dedup) >= GENERAL_M3_MAX_ITEMS:
            break
    return dedup, False


def _upsert_m3_bucket(
    profile: dict[str, Any],
    *,
    bucket_name: str,
    items: list[str],
    negated: bool,
    note: str = "",
) -> bool:
    entities = profile.get("entities", {})
    bucket = entities.get(bucket_name)
    if not isinstance(bucket, dict):
        bucket = _new_m3_bucket()
        entities[bucket_name] = bucket
    changed = False
    normalized_note = _clean_turn_text(note, 120)

    if negated:
        if (
            bucket.get("status") != "negated"
            or bool(bucket.get("items"))
            or str(bucket.get("note", "")) != normalized_note
        ):
            bucket["status"] = "negated"
            bucket["items"] = []
            bucket["note"] = normalized_note
            changed = True
    elif items:
        current = [str(item) for item in bucket.get("items", []) if str(item).strip()]
        merged: list[str] = []
        seen: set[str] = set()
        for item in current + items:
            normalized = _normalize_m3_item(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        merged = merged[-GENERAL_M3_MAX_ITEMS:]
        if (
            bucket.get("status") != "confirmed"
            or bucket.get("items") != merged
            or (normalized_note and str(bucket.get("note", "")) != normalized_note)
        ):
            bucket["status"] = "confirmed"
            bucket["items"] = merged
            if normalized_note:
                bucket["note"] = normalized_note
            changed = True

    if changed:
        bucket["updated_at"] = datetime.now().isoformat(timespec="seconds")
        entities[bucket_name] = bucket
        profile["entities"] = entities
    return changed


def _update_general_m3(session: dict[str, Any], user_query: str) -> list[str]:
    profile = _ensure_general_m3_shape(session)
    text = str(user_query or "").strip()
    if not text:
        return []

    before: dict[str, tuple[str, tuple[str, ...]]] = {}
    entities_before = profile.get("entities", {})
    if isinstance(entities_before, dict):
        for bucket_name in M3_BUCKET_ORDER:
            bucket = entities_before.get(bucket_name, {})
            if not isinstance(bucket, dict):
                continue
            status = str(bucket.get("status", "unknown")).strip().lower()
            items = bucket.get("items", [])
            if not isinstance(items, list):
                items = []
            before[bucket_name] = (status, tuple(str(x) for x in items if str(x).strip()))

    updates = {
        "allergy": _extract_allergy_entities(text),
        "chronic_disease": _extract_chronic_entities(text),
        "medication": _extract_medication_entities(text),
        "pregnancy_lactation": _extract_pregnancy_lactation_entities(text),
        "surgery_history": _extract_surgery_entities(text),
        "family_history": _extract_family_entities(text),
    }

    changed = False
    for bucket_name, value in updates.items():
        items, negated = value
        changed = _upsert_m3_bucket(
            profile,
            bucket_name=bucket_name,
            items=items,
            negated=bool(negated),
            note="用户最新说明" if (items or negated) else "",
        ) or changed

    events: list[str] = []
    if changed:
        entities_after = profile.get("entities", {})
        if isinstance(entities_after, dict):
            for bucket_name in M3_BUCKET_ORDER:
                bucket = entities_after.get(bucket_name, {})
                if not isinstance(bucket, dict):
                    continue
                now_status = str(bucket.get("status", "unknown")).strip().lower()
                items = bucket.get("items", [])
                if not isinstance(items, list):
                    items = []
                now_items = tuple(str(x) for x in items if str(x).strip())
                prev_status, prev_items = before.get(bucket_name, ("unknown", tuple()))
                if now_status == prev_status and now_items == prev_items:
                    continue
                label = M3_BUCKET_LABELS.get(bucket_name, bucket_name)
                if now_status == "negated":
                    events.append(f"{label}=否认")
                elif now_status == "confirmed" and now_items:
                    events.append(f"{label}=" + "、".join(now_items[:3]))
        profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
        session["m3"] = profile
    return events


def _build_general_m3_text(session: dict[str, Any]) -> str:
    profile = _ensure_general_m3_shape(session)
    entities = profile.get("entities", {})
    lines: list[str] = []
    for bucket_name in M3_BUCKET_ORDER:
        bucket = entities.get(bucket_name, {})
        if not isinstance(bucket, dict):
            continue
        status = str(bucket.get("status", "unknown")).strip().lower()
        items = bucket.get("items", [])
        if not isinstance(items, list):
            items = []
        label = M3_BUCKET_LABELS.get(bucket_name, bucket_name)
        if status == "confirmed" and items:
            safe_items = [_clean_turn_text(item, GENERAL_M3_ITEM_MAX_CHARS) for item in items if str(item).strip()]
            if safe_items:
                lines.append(f"- {label}: {'、'.join(safe_items[:GENERAL_M3_MAX_ITEMS])}")
        elif status == "negated":
            lines.append(f"- {label}: 用户否认")

    block = "\n".join(lines).strip()
    if len(block) > GENERAL_M3_BLOCK_MAX_CHARS:
        return "..." + block[-GENERAL_M3_BLOCK_MAX_CHARS:]
    return block


def _normalize_similarity_text(text: object) -> str:
    merged = re.sub(r"\s+", "", str(text or "").lower())
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", merged)


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    clean = _normalize_similarity_text(text)
    if not clean:
        return set()
    if len(clean) < n:
        return {clean}
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def _topic_similarity(a: object, b: object) -> float:
    grams_a = _char_ngrams(str(a or ""))
    grams_b = _char_ngrams(str(b or ""))
    if not grams_a or not grams_b:
        return 1.0
    union = grams_a | grams_b
    if not union:
        return 1.0
    return len(grams_a & grams_b) / len(union)


def _is_explicit_topic_switch(query: str) -> bool:
    text = str(query or "").strip()
    return any(marker in text for marker in EXPLICIT_TOPIC_SWITCH_MARKERS)


def _new_general_segment(seq: int, *, intent: str = "", topic_seed: str = "") -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": f"seg-{int(seq)}",
        "intent": str(intent or "").strip(),
        "topic_hint": _clean_turn_text(topic_seed, 240),
        "summary_text": "",
        "summary_lines": [],
        "turns": [],
        "created_at": now,
        "updated_at": now,
    }


def _ensure_general_session_shape(session: dict[str, Any]) -> None:
    session["session_id"] = str(session.get("session_id", "")).strip()
    _ensure_general_m3_shape(session)
    _ensure_general_m2_shape(session)
    if not isinstance(session.get("segments"), list):
        session["segments"] = []
    if not isinstance(session.get("turns"), list):
        session["turns"] = []

    pending = session.get("pending_switch")
    if not isinstance(pending, dict):
        pending = {}
    session["pending_switch"] = {
        "intent": str(pending.get("intent", "")).strip(),
        "streak": max(0, int(_safe_float(pending.get("streak", 0), 0))),
    }

    segments: list[dict[str, Any]] = [seg for seg in session["segments"] if isinstance(seg, dict)]
    if not segments:
        segments = [_new_general_segment(1)]
    max_seq = 1
    for segment in segments:
        matched = re.fullmatch(r"seg-(\d+)", str(segment.get("id", "")).strip())
        if matched:
            max_seq = max(max_seq, int(matched.group(1)))
    current_seq = int(_safe_float(session.get("segment_seq", 0), 0))
    if current_seq <= 0:
        current_seq = max_seq
    session["segment_seq"] = max(current_seq, max_seq)
    session["segments"] = segments

    active_id = str(session.get("active_segment_id", "")).strip()
    if not any(str(seg.get("id", "")) == active_id for seg in segments):
        session["active_segment_id"] = str(segments[-1].get("id", "seg-1"))


def _create_general_segment(session: dict[str, Any], *, intent: str = "", topic_seed: str = "") -> dict[str, Any]:
    _ensure_general_session_shape(session)
    next_seq = int(session.get("segment_seq", 1)) + 1
    segment = _new_general_segment(next_seq, intent=intent, topic_seed=topic_seed)
    segments: list[dict[str, Any]] = session.get("segments", [])
    segments.append(segment)
    session["segment_seq"] = next_seq

    if len(segments) > GENERAL_MAX_SEGMENTS:
        segments = segments[-GENERAL_MAX_SEGMENTS:]
    session["segments"] = segments
    session["active_segment_id"] = str(segment.get("id", ""))
    return segment


def _get_active_general_segment(session: dict[str, Any]) -> dict[str, Any]:
    _ensure_general_session_shape(session)
    active_id = str(session.get("active_segment_id", "")).strip()
    segments: list[dict[str, Any]] = session.get("segments", [])
    for segment in segments:
        if str(segment.get("id", "")).strip() == active_id:
            return segment
    segment = segments[-1]
    session["active_segment_id"] = str(segment.get("id", ""))
    return segment


def _segment_summary_line(turn: dict[str, Any]) -> str:
    user_text = _clean_turn_text(turn.get("user", ""), GENERAL_SUMMARY_USER_MAX)
    assistant_text = _clean_turn_text(turn.get("assistant", ""), GENERAL_SUMMARY_ASSISTANT_MAX)
    if user_text and assistant_text:
        return f"- 用户提到: {user_text}；助手回应: {assistant_text}"
    if user_text:
        return f"- 用户提到: {user_text}"
    if assistant_text:
        return f"- 助手回应: {assistant_text}"
    return ""


def _refresh_segment_memory(segment: dict[str, Any]) -> None:
    turns = segment.get("turns", [])
    if not isinstance(turns, list):
        turns = []
    prefix_count = max(0, len(turns) - GENERAL_M0_TURNS)

    summary_lines = segment.get("summary_lines", [])
    if not isinstance(summary_lines, list):
        summary_lines = []
    summary_lines = [str(line) for line in summary_lines[:prefix_count]]

    for idx in range(len(summary_lines), prefix_count):
        turn = turns[idx] if idx < len(turns) and isinstance(turns[idx], dict) else {}
        line = _segment_summary_line(turn)
        if line:
            summary_lines.append(line)

    summary_text = "\n".join(summary_lines).strip()
    if len(summary_text) > GENERAL_M1_MAX_CHARS:
        summary_text = "..." + summary_text[-GENERAL_M1_MAX_CHARS:]

    segment["summary_lines"] = summary_lines
    segment["summary_text"] = summary_text


def _choose_segment_for_turn(
    session: dict[str, Any],
    *,
    query: str,
    intent: str,
    intent_confidence: float,
) -> dict[str, Any]:
    active = _get_active_general_segment(session)
    active_turns = active.get("turns", [])
    if not isinstance(active_turns, list) or not active_turns:
        return active

    target_intent = str(intent or "").strip()
    active_intent = str(active.get("intent", "")).strip()
    topic_hint = str(active.get("topic_hint", "")).strip()
    confidence = _safe_float(intent_confidence, 0.0)
    similarity = _topic_similarity(query, topic_hint)

    explicit_switch = _is_explicit_topic_switch(query)
    intent_changed = bool(active_intent and target_intent and target_intent != active_intent)
    drifted = similarity < GENERAL_SEGMENT_SIM_THRESHOLD
    conf_ok = confidence >= GENERAL_SEGMENT_INTENT_CONF_THRESHOLD

    pending = session.get("pending_switch", {"intent": "", "streak": 0})
    if not isinstance(pending, dict):
        pending = {"intent": "", "streak": 0}

    if explicit_switch:
        session["pending_switch"] = {"intent": "", "streak": 0}
        return _create_general_segment(session, intent=target_intent, topic_seed=query)

    if intent_changed and drifted and conf_ok:
        if str(pending.get("intent", "")).strip() == target_intent:
            next_streak = max(1, int(_safe_float(pending.get("streak", 0), 0)) + 1)
        else:
            next_streak = 1
        session["pending_switch"] = {"intent": target_intent, "streak": next_streak}
        if next_streak >= GENERAL_SEGMENT_SWITCH_STREAK:
            session["pending_switch"] = {"intent": "", "streak": 0}
            return _create_general_segment(session, intent=target_intent, topic_seed=query)
    else:
        session["pending_switch"] = {"intent": "", "streak": 0}

    return active


def _build_general_history_text(session: dict[str, Any], query: str = "") -> str:
    segment = _get_active_general_segment(session)
    turns = segment.get("turns", [])
    if not isinstance(turns, list):
        turns = []
    m0_turns = [turn for turn in turns[-GENERAL_M0_TURNS:] if isinstance(turn, dict)]
    summary_text = str(segment.get("summary_text", "") or "").strip()
    segment_intent = str(segment.get("intent", "") or "").strip()

    m0_lines: list[str] = []
    if m0_turns:
        m0_lines.append("最近对话(M0):")
        for turn in m0_turns:
            user_text = _clean_turn_text(turn.get("user", ""), 180)
            assistant_text = _clean_turn_text(turn.get("assistant", ""), 260)
            if user_text:
                m0_lines.append(f"用户: {user_text}")
            if assistant_text:
                m0_lines.append(f"助手: {assistant_text}")
    m0_block = "\n".join(m0_lines).strip()

    prefix_parts: list[str] = []
    m2_text = _build_general_m2_text(session, query, intent_hint=segment_intent)
    if m2_text:
        prefix_parts.append(m2_text)
    m3_text = _build_general_m3_text(session)
    if m3_text:
        prefix_parts.append("医疗核心事实(M3):")
        prefix_parts.append(m3_text)
    if segment_intent:
        prefix_parts.append(f"当前段意图: {segment_intent}")
    if summary_text:
        prefix_parts.append("段内摘要(M1):")
        prefix_parts.append(summary_text)
    prefix_block = "\n".join(prefix_parts).strip()

    if prefix_block and m0_block:
        merged = f"{prefix_block}\n{m0_block}"
    else:
        merged = prefix_block or m0_block

    if len(merged) <= GENERAL_MEMORY_MAX_CHARS:
        return merged

    if not m0_block:
        return "..." + merged[-GENERAL_MEMORY_MAX_CHARS:]
    if len(m0_block) >= GENERAL_MEMORY_MAX_CHARS:
        return "..." + m0_block[-GENERAL_MEMORY_MAX_CHARS:]

    budget = max(0, GENERAL_MEMORY_MAX_CHARS - len(m0_block) - 1)
    if budget <= 0:
        return "..." + m0_block[-GENERAL_MEMORY_MAX_CHARS:]

    if len(prefix_block) > budget:
        prefix_block = "..." + prefix_block[-max(0, budget - 3) :]
    if not prefix_block:
        return m0_block
    return f"{prefix_block}\n{m0_block}"


def _append_general_turn(
    session: dict[str, Any],
    query: str,
    answer: str,
    *,
    intent: str = "",
    intent_confidence: float = 0.0,
    llm_runtime: dict[str, Any] | None = None,
) -> None:
    normalized_query = _clean_turn_text(query, GENERAL_TURN_USER_MAX)
    normalized_answer = _clean_turn_text(answer, GENERAL_TURN_ASSISTANT_MAX)
    normalized_intent = str(intent or "").strip()
    m3_events = _update_general_m3(session, normalized_query)
    m2_fact = _select_m2_fact(normalized_query, m3_events, normalized_intent)
    if m2_fact:
        m2_fact = _clean_m2_fact_with_llm(
            query=normalized_query,
            intent=normalized_intent,
            rule_fact=m2_fact,
            m3_events=m3_events,
            llm_runtime=llm_runtime,
        )
        _upsert_general_m2(session, m2_fact, intent=normalized_intent)

    segment = _choose_segment_for_turn(
        session,
        query=normalized_query,
        intent=normalized_intent,
        intent_confidence=_safe_float(intent_confidence, 0.0),
    )

    turns = segment.get("turns", [])
    if not isinstance(turns, list):
        turns = []
    turns.append(
        {
            "user": normalized_query,
            "assistant": normalized_answer,
            "intent": normalized_intent,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    segment["turns"] = turns
    segment["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if normalized_intent and not str(segment.get("intent", "")).strip():
        segment["intent"] = normalized_intent

    topic_seed = " ".join(str(turn.get("user", "")) for turn in turns[-2:])
    if topic_seed:
        segment["topic_hint"] = _clean_turn_text(topic_seed, 240)

    _refresh_segment_memory(segment)
    session["active_segment_id"] = str(segment.get("id", ""))

    legacy_turns = session.get("turns", [])
    if not isinstance(legacy_turns, list):
        legacy_turns = []
    legacy_turns.append(
        {
            "user": normalized_query,
            "assistant": normalized_answer,
            "intent": normalized_intent,
            "segment_id": str(segment.get("id", "")),
        }
    )
    session["turns"] = legacy_turns[-max(12, GENERAL_M0_TURNS * 4) :]


def _new_general_session(session_id: str = "") -> tuple[str, dict[str, Any]]:
    sid = str(session_id or "").strip() or str(uuid.uuid4())
    session: dict[str, Any] = {
        "session_id": sid,
        "m2": _new_m2_state(),
        "m3": _new_m3_profile(),
        "segments": [],
        "active_segment_id": "",
        "segment_seq": 1,
        "pending_switch": {"intent": "", "streak": 0},
        "turns": [],
    }
    _ensure_general_session_shape(session)
    GENERAL_SESSIONS[sid] = session
    return sid, session


def _get_or_create_general_session(session_id: str) -> tuple[str, dict[str, Any]]:
    sid = str(session_id or "").strip()
    if sid and sid in GENERAL_SESSIONS:
        session = GENERAL_SESSIONS[sid]
        session["session_id"] = sid
        _ensure_general_session_shape(session)
        return sid, session
    if sid:
        return _new_general_session(sid)
    return _new_general_session()


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


@web_app.post("/api/chat/feedback")
def chat_feedback():
    payload = request.get_json(silent=True) or {}
    reaction = str(payload.get("reaction", "")).strip().lower()
    if reaction not in {"up", "down"}:
        return jsonify({"error": "reaction must be up or down"}), 400

    user_query = _clean_feedback_text(payload.get("user_query", ""), 8000)
    assistant_answer = _clean_feedback_text(payload.get("assistant_answer", ""), 12000)
    if not assistant_answer:
        return jsonify({"error": "assistant_answer is required"}), 400

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    record = {
        "reaction": reaction,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": _clean_feedback_text(payload.get("session_id", ""), 128),
        "message_id": _clean_feedback_text(payload.get("message_id", ""), 128),
        "user_query": user_query,
        "assistant_answer": assistant_answer,
        "meta": {
            "intent": _clean_feedback_text(meta.get("intent", ""), 64),
            "intent_source": _clean_feedback_text(meta.get("intent_source", ""), 32),
            "risk_level": _clean_feedback_text(meta.get("risk_level", ""), 32),
            "handoff": bool(meta.get("handoff", False)),
            "confidence": float(meta.get("confidence", 0.0) or 0.0),
            "citations": list(meta.get("citations", []))[:20] if isinstance(meta.get("citations", []), list) else [],
        },
    }

    bucket = "thumb_up" if reaction == "up" else "thumb_down"
    target_dir = CHAT_FEEDBACK_DIR / bucket
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{stamp}_{uuid.uuid4().hex[:8]}.json"
    path = target_dir / filename
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    rel = path.relative_to(BASE_DIR).as_posix()
    return jsonify({"ok": True, "saved_to": rel})


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
    history_text = _build_general_history_text(session, query=query)

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

    _append_general_turn(
        session,
        query,
        str(result.get("answer", "")),
        intent=str(result.get("intent", "")),
        intent_confidence=_safe_float(result.get("intent_confidence", 0.0), 0.0),
        llm_runtime=llm_runtime,
    )
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
    history_text = _build_general_history_text(session, query=query)

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
                secondary_intents=list(state.get("secondary_intents", []) or []),
                risk_level=state.get("risk_level", "low"),
                conversation_history_text=history_text,
                llm_runtime=llm_runtime,
            )

            stream_iter = stream_response_with_llm(
                query=query,
                intent=state.get("intent", "other"),
                secondary_intents=list(state.get("secondary_intents", []) or []),
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

            _append_general_turn(
                session,
                query,
                final_answer,
                intent=str(state.get("intent", "")),
                intent_confidence=_safe_float(state.get("intent_confidence", 0.0), 0.0),
                llm_runtime=llm_runtime,
            )
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
            "退出中医辨证模式，回到医疗助手",
            "继续补充症状，重新开始辨证",
        ]
    else:
        follow_ups = [
            "继续完成下一轮问卷",
            "补充更多症状后再做一轮",
            "退出中医辨证模式，回到医疗助手",
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
