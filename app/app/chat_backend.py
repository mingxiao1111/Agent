from __future__ import annotations

import os
from typing import Any

from langchain_community.chat_models import ChatTongyi

from .model_compat import normalize_dashscope_chat_model


def _to_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return int(default)


def _to_float_env(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return float(default)


def chat_backend_name() -> str:
    """聊天模型后端：dashscope_sdk / openai_compatible。"""

    raw = str(os.getenv("CHAT_BACKEND", "dashscope_sdk")).strip().lower()
    if raw in {"dashscope_sdk", "openai_compatible"}:
        return raw
    return "dashscope_sdk"


def _openai_compat_api_key() -> str:
    return str(
        os.getenv("OPENAI_COMPAT_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or ""
    ).strip()


def _openai_compat_base_url() -> str:
    return str(
        os.getenv("OPENAI_COMPAT_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip()


def is_chat_enabled() -> bool:
    """是否具备聊天模型调用条件。"""

    if chat_backend_name() == "openai_compatible":
        return bool(_openai_compat_api_key())
    return bool(os.getenv("DASHSCOPE_API_KEY"))


def build_chat_model(*, model_name: str, temperature: float, streaming: bool = False) -> Any:
    """按后端配置构建聊天模型实例。"""

    backend = chat_backend_name()
    if backend == "openai_compatible":
        from langchain_openai import ChatOpenAI

        api_key = _openai_compat_api_key()
        base_url = _openai_compat_base_url()
        timeout = _to_float_env("OPENAI_COMPAT_TIMEOUT_SEC", 60)
        max_retries = _to_int_env("OPENAI_COMPAT_MAX_RETRIES", 2)
        return ChatOpenAI(
            model=str(model_name or "").strip(),
            temperature=float(temperature),
            streaming=bool(streaming),
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    normalized = normalize_dashscope_chat_model(model_name)
    return ChatTongyi(
        model=normalized,
        temperature=float(temperature),
        streaming=bool(streaming),
    )

