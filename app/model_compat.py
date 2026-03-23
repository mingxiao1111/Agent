from __future__ import annotations


def normalize_dashscope_chat_model(model_name: str) -> str:
    """Normalize model name for DashScope chat compatibility.

    Current project defaults to Volcengine model ids; here we keep a no-op
    normalizer so legacy call sites remain stable.
    """

    raw = str(model_name or "").strip()
    if not raw:
        return raw

    return raw
