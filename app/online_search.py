from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any


MEDICAL_KEYWORDS = [
    "医疗",
    "健康",
    "就医",
    "门诊",
    "挂号",
    "医院",
    "医生",
    "药",
    "用药",
    "副作用",
    "说明书",
    "症状",
    "发热",
    "咳嗽",
    "头痛",
    "胸痛",
    "腹痛",
    "腹泻",
    "报告",
    "化验",
    "指标",
    "检查",
    "中医",
    "辨证",
    "舌苔",
    "脉象",
]

SMALL_TALK_KEYWORDS = [
    "你好",
    "您好",
    "在吗",
    "你是谁",
    "你能做什么",
    "功能",
    "介绍一下你",
]

MEDICAL_INTENTS = {
    "symptom_consult",
    "appointment_process",
    "medication_question",
    "report_interpretation",
    "emergency",
    "tcm",
}


def _safe_strip(value: Any) -> str:
    return str(value or "").strip()


def _truthy_env(name: str) -> bool:
    val = str(os.getenv(name, "")).strip().lower()
    return val in {"1", "true", "yes", "on"}


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _query_tokens(text: str) -> list[str]:
    chunks = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}", text.lower())
    tokens: list[str] = []
    for chunk in chunks:
        tokens.append(chunk)
        if re.match(r"^[\u4e00-\u9fff]+$", chunk) and len(chunk) > 2:
            for i in range(len(chunk) - 1):
                tokens.append(chunk[i : i + 2])
    stop = {"什么", "一下", "这个", "那个", "可以", "怎么", "今天", "最近", "情况", "问题"}
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip()
        if not token or token in stop or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _is_medical_intent(intent_hint: str) -> bool:
    return str(intent_hint or "").strip().lower() in MEDICAL_INTENTS


def _is_medical_query(query: str, intent_hint: str) -> bool:
    if _is_medical_intent(intent_hint):
        return True
    return _contains_any(query, MEDICAL_KEYWORDS)


def _is_small_talk_query(query: str, intent_hint: str) -> bool:
    if _is_medical_intent(intent_hint):
        return False
    q = query.strip()
    if len(q) <= 2:
        return True
    if _contains_any(q, SMALL_TALK_KEYWORDS):
        return True
    return False


def _should_search(query: str, intent_hint: str) -> tuple[bool, str]:
    if _truthy_env("TAVILY_FORCE_SEARCH"):
        return True, "force_search"

    q = query.strip()
    if not q:
        return False, "no_query"
    if _is_small_talk_query(q, intent_hint):
        return False, "small_talk"
    if _is_medical_query(q, intent_hint):
        return True, "medical_query"
    if len(_query_tokens(q)) >= 4:
        return True, "generic_long_query"
    return False, "generic_short_query"


def _rewrite_query_for_search(query: str, intent_hint: str) -> str:
    q = query.strip()
    if _is_medical_query(q, intent_hint):
        return f"{q} 医疗 健康 就医建议"
    return q


def _is_low_relevance(query: str, answer: str, rows: list[dict[str, str]], intent_hint: str) -> bool:
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return True

    corpus_parts: list[str] = [answer]
    for item in rows:
        corpus_parts.append(item.get("title", ""))
        corpus_parts.append(item.get("snippet", ""))
    corpus = "\n".join(corpus_parts).lower()

    hit_count = sum(1 for t in q_tokens if t in corpus)
    ratio = hit_count / max(1, len(q_tokens))

    if _is_medical_query(query, intent_hint):
        return hit_count < 1 or ratio < 0.06
    return hit_count < 2 or ratio < 0.12


@lru_cache(maxsize=1)
def _tavily_client():
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from tavily import TavilyClient

        return TavilyClient(api_key=api_key)
    except Exception:
        return None


def tavily_is_enabled() -> bool:
    return _tavily_client() is not None


def fetch_tavily_context(
    query: str,
    *,
    max_results: int = 3,
    topic: str = "general",
    intent_hint: str = "",
    enable_online: bool = False,
) -> str:
    q = _safe_strip(query)
    if not q:
        return "(no_query)"
    if not enable_online:
        return "(tavily_disabled)"

    do_search, reason = _should_search(q, intent_hint)
    if not do_search:
        return f"(tavily_skipped: {reason})"

    client = _tavily_client()
    if client is None:
        return "(tavily_unavailable)"

    search_query = _rewrite_query_for_search(q, intent_hint)

    try:
        response = client.search(
            query=search_query,
            topic=topic,
            max_results=max_results,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
        )

        answer = _safe_strip(response.get("answer"))
        structured_rows: list[dict[str, str]] = []
        rows: list[str] = []
        if answer:
            rows.append(f"answer: {answer}")

        for idx, item in enumerate(response.get("results", [])[:max_results], start=1):
            title = _safe_strip(item.get("title"))
            url = _safe_strip(item.get("url"))
            snippet = _safe_strip(item.get("content"))
            structured_rows.append({"title": title, "snippet": snippet, "url": url})
            rows.append(f"[{idx}] {title}")
            if snippet:
                rows.append(f"snippet: {snippet}")
            if url:
                rows.append(f"url: {url}")

        if not rows:
            return "(tavily_no_results)"

        if not _truthy_env("TAVILY_DISABLE_RELEVANCE_FILTER") and _is_low_relevance(
            q,
            answer,
            structured_rows,
            intent_hint,
        ):
            return "(tavily_low_relevance_filtered)"

        return "\n".join(rows)
    except Exception as exc:
        return f"(tavily_error: {exc})"
