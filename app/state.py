from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    llm_provider: str
    llm_model: str
    llm_thinking: bool
    conversation_history_text: str
    normalized_input: str
    risk_level: str
    risk_reason: str
    intent: str
    secondary_intents: list[str]
    intent_candidates: list[str]
    intent_confidence: float
    intent_source: str
    handoff_hint: bool
    context_docs: list[dict[str, Any]]
    tool_results: dict[str, Any]
    confidence: float
    handoff: bool
    answer: str
    citations: list[str]
    handoff_summary: str
    follow_ups: list[str]
    prefetched_followups: list[str]
    disable_llm_response: bool
    disable_llm_followups: bool
    enable_online_search: bool
