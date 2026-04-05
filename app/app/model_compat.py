from __future__ import annotations

import os
from typing import Any


def _is_deepseek_v32_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return "deepseek-v3-2" in name


def _is_doubao_seed_20_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return "doubao-seed-2-0" in name


def supports_volcengine_reasoning(model_name: str) -> bool:
    return _is_deepseek_v32_model(model_name) or _is_doubao_seed_20_model(model_name)


def _normalize_reasoning_effort(value: str, fallback: str) -> str:
    effort = str(value or "").strip().lower()
    allowed = {"minimal", "low", "medium", "high"}
    if effort in allowed:
        return effort
    return fallback


def volcengine_reasoning_kwargs(model_name: str, thinking: bool) -> dict[str, Any]:
    """按模型适配 Volcengine 的思考参数。

    - DeepSeek: 使用 extra_body.thinking.type = enabled/disabled
    - Doubao Seed 2.0: 使用 reasoning_effort，关思考时明确 minimal
    """

    if _is_deepseek_v32_model(model_name):
        return {"extra_body": {"thinking": {"type": "enabled" if bool(thinking) else "disabled"}}}

    if _is_doubao_seed_20_model(model_name):
        on_effort = _normalize_reasoning_effort(
            os.getenv("VOLCENGINE_DOUBAO_REASONING_ON", "high"),
            "high",
        )
        off_effort = _normalize_reasoning_effort(
            os.getenv("VOLCENGINE_DOUBAO_REASONING_OFF", "minimal"),
            "minimal",
        )
        return {"reasoning_effort": on_effort if bool(thinking) else off_effort}

    return {}


def normalize_dashscope_chat_model(model_name: str) -> str:
    """Normalize model name for DashScope chat compatibility.

    Current project defaults to Volcengine model ids; here we keep a no-op
    normalizer so legacy call sites remain stable.
    """

    raw = str(model_name or "").strip()
    if not raw:
        return raw

    return raw
