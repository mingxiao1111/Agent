from __future__ import annotations

import json
import os
from typing import Any


def _trace_enabled() -> bool:
    flag = str(os.getenv("LLM_TRACE_ENABLED", "1")).strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _max_chars() -> int:
    raw = str(os.getenv("LLM_TRACE_MAX_CHARS", "12000")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 12000
    return max(800, min(value, 100000))


def _clip(text: str) -> str:
    max_len = _max_chars()
    if len(text) <= max_len:
        return text
    hidden = len(text) - max_len
    return f"{text[:max_len]}\n... [truncated {hidden} chars]"


def _json_safe(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(data)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _format_messages(messages: list[Any]) -> str:
    blocks: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        role = str(getattr(msg, "type", "") or getattr(msg, "role", "") or msg.__class__.__name__)
        content = _message_content_to_text(getattr(msg, "content", msg))
        blocks.append(f"[{idx}] {role}\n{content}")
    return "\n\n".join(blocks)


def _emit(tag: str, section: str, content: str) -> None:
    if not _trace_enabled():
        return
    print(f"\n[LLM_TRACE] {tag} | {section}\n{_clip(content)}\n", flush=True)


def log_prompt(tag: str, prompt: Any, payload: dict[str, Any]) -> None:
    if not _trace_enabled():
        return
    payload_text = _json_safe(payload)
    try:
        messages = prompt.format_messages(**payload)
        msg_text = _format_messages(messages)
    except Exception as exc:
        msg_text = f"(prompt_format_failed: {exc})"
    _emit(tag, "PROMPT_PAYLOAD", payload_text)
    _emit(tag, "PROMPT_MESSAGES", msg_text)


def log_response(tag: str, text: str) -> None:
    _emit(tag, "RESPONSE_TEXT", str(text))


def log_error(tag: str, exc: Exception) -> None:
    _emit(tag, "ERROR", f"{exc.__class__.__name__}: {exc}")


def log_model_route(tag: str, payload: dict[str, Any]) -> None:
    """记录模型路由决策，便于排查多模型编排是否按预期生效。"""

    _emit(tag, "MODEL_ROUTE", _json_safe(payload))
