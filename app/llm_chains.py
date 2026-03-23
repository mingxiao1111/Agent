from __future__ import annotations

"""LLM 能力层（意图识别、主回复、流式输出、猜你想问）。

这个模块专门负责“模型调用”相关工作，保持两点：
1) 结构化：尽量要求模型输出 JSON，便于稳定解析；
2) 可回退：任何异常都可以退回规则逻辑，不影响主链路可用性。
"""

import json
import os
import re
from typing import Any, Iterator

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .chat_backend import build_chat_model, is_chat_enabled
from .llm_trace import log_error, log_model_route, log_prompt, log_response
from .online_search import fetch_tavily_context

INTENT_LABELS = {
    "daily_chat",
    "symptom_consult",
    "medical_knowledge",
    "medication_question",
    "lifestyle_guidance",
    "appointment_process",
    "report_interpretation",
    "after_sales",
    "human_service",
    "non_medical",
    "other",
}

DEFAULT_MODEL = "deepseek-v3-2-251201"
DEFAULT_ASSISTANT_PROFILE = (
    "你是智能医疗助手，可提供健康问题与症状咨询、就医流程指导、"
    "用药与健康品安全科普、检查报告解读与健康生活方式建议。"
    "你不能做临床诊断、不能替代医生、不能提供个体化处方。"
)

INTENT_SOFT_GUIDANCE = {
    "symptom_consult": "优先给分诊建议、观察要点与何时线下就医；避免诊断化措辞。",
    "appointment_process": "优先说明就医/挂号步骤、所需材料与时间安排注意事项。",
    "report_interpretation": "优先解释指标含义与常见影响因素，提醒需结合医生面诊判断。",
    "medication_question": "优先给说明书级风险提醒、禁忌与相互作用，不做个体化处方建议。",
    "medical_knowledge": "优先做医学科普，强调一般性信息与适用边界。",
    "lifestyle_guidance": "优先给可执行、保守的饮食作息运动建议，必要时提醒就医。",
    "after_sales": "优先说明转人工售后的原因与下一步所需信息。",
    "human_service": "优先确认已转人工并告知下一步对接方式。",
    "non_medical": "简要说明能力边界，再引导用户提出医疗相关问题。",
    "daily_chat": "以简洁友好方式承接，并自然引导到医疗咨询能力。",
    "other": "先澄清用户核心诉求，再引导到可处理的医疗场景。",
}

VOLCENGINE_MODELS = {
    "deepseek-v3-2-251201",
    "doubao-seed-2-0-pro-260215",
    "doubao-seed-2-0-mini-260215",
}
VOLCENGINE_DEFAULT_CHAT_MODEL = "deepseek-v3-2-251201"
VOLCENGINE_DEFAULT_INTENT_MODEL = "doubao-seed-2-0-mini-260215"
VOLCENGINE_FAST_MODEL = "doubao-seed-2-0-mini-260215"
FAST_PATH_INTENTS = {
    "daily_chat",
    "after_sales",
    "human_service",
    "non_medical",
    "other",
}


class IntentRouteOutput(BaseModel):
    """意图路由结构化输出。

    用 Pydantic 约束字段，减少模型自由文本导致的解析失败。
    """

    intent: str = Field(description="意图标签")
    confidence: float = Field(description="意图置信度，0到1")
    needs_handoff: bool = Field(description="是否建议人工接管")


class FollowupOutput(BaseModel):
    """猜你想问结构化输出。"""

    questions: list[str] = Field(description="3条用户下一步可能追问的问题")


def _extract_text(raw_output: Any) -> str:
    """统一提取模型返回文本（兼容 str / list / message 对象）。"""

    content = getattr(raw_output, "content", raw_output)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_chunk_text(raw_output: Any) -> str:
    """流式场景提取 chunk 文本，不做 strip，避免拼接变形。"""

    content = getattr(raw_output, "content", raw_output)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _to_confidence(value: Any, fallback: float) -> float:
    """将输入安全转换为 [0,1] 置信度。"""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, score))


def _normalize_questions(items: list[Any]) -> list[str]:
    """清洗猜你想问：去空、去重、最多 3 条。"""

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) == 3:
            break
    return out


def _summarize_history(history_text: str, max_len: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(history_text or "")).strip()
    if not text:
        return "(无历史对话)"
    if len(text) <= max_len:
        return text
    return "..." + text[-max_len:]


def _intent_soft_guidance(intent: str) -> str:
    key = str(intent or "").strip().lower()
    return INTENT_SOFT_GUIDANCE.get(key, INTENT_SOFT_GUIDANCE["other"])


def _runtime_provider(llm_runtime: dict[str, Any] | None) -> str:
    provider = str((llm_runtime or {}).get("provider", "default")).strip().lower()
    if provider in {"volcengine", "default"}:
        return provider
    return "default"


def _runtime_model(llm_runtime: dict[str, Any] | None) -> str:
    model = str((llm_runtime or {}).get("model", "")).strip()
    if not model:
        return ""
    return model


def _runtime_thinking(llm_runtime: dict[str, Any] | None) -> bool:
    value = (llm_runtime or {}).get("thinking", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _with_forced_volcengine_model(
    llm_runtime: dict[str, Any] | None,
    *,
    forced_model: str,
    force_thinking_false: bool = True,
    reason: str = "",
    intent: str = "",
) -> dict[str, Any]:
    """按需强制覆写 Volcengine 运行时模型（用于速度优先路径）。"""

    runtime = dict(llm_runtime or {})
    if _runtime_provider(runtime) != "volcengine":
        return runtime

    model = str(forced_model or "").strip()
    if not model or model not in VOLCENGINE_MODELS:
        return runtime

    before_model = str(runtime.get("model", "")).strip()
    before_thinking = _runtime_thinking(runtime)
    runtime["model"] = model
    if force_thinking_false:
        runtime["thinking"] = False

    log_model_route(
        "llm_chains.route.override",
        {
            "reason": str(reason or ""),
            "intent": str(intent or ""),
            "before_model": before_model,
            "after_model": model,
            "before_thinking": bool(before_thinking),
            "after_thinking": bool(_runtime_thinking(runtime)),
        },
    )
    return runtime


def _response_runtime_for_intent(
    *,
    llm_runtime: dict[str, Any] | None,
    intent: str,
) -> dict[str, Any]:
    key = str(intent or "").strip().lower()
    if key in FAST_PATH_INTENTS:
        return _with_forced_volcengine_model(
            llm_runtime,
            forced_model=VOLCENGINE_FAST_MODEL,
            force_thinking_false=True,
            reason="fast_path_intent",
            intent=key,
        )
    return dict(llm_runtime or {})


def _volcengine_api_key() -> str:
    return str(os.getenv("VOLCENGINE_API_KEY", "")).strip()


def _volcengine_base_url() -> str:
    return str(os.getenv("VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")).strip()


def _supports_thinking(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return bool(name and ("doubao-seed-2-0-pro" in name or "deepseek-v3-2" in name))


def _build_volcengine_chat_model(
    *,
    model_name: str,
    temperature: float,
    streaming: bool,
    thinking: bool,
) -> Any:
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {}
    # 对 OpenAI-compatible 客户端，供应商扩展参数应放在 extra_body。
    # 直接塞到 model_kwargs 会被提升为 create() 顶层参数，触发 unexpected keyword。
    if _supports_thinking(model_name) and bool(thinking):
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    return ChatOpenAI(
        model=model_name,
        temperature=float(temperature),
        streaming=bool(streaming),
        api_key=_volcengine_api_key(),
        base_url=_volcengine_base_url(),
        timeout=float(os.getenv("VOLCENGINE_TIMEOUT_SEC", "60")),
        max_retries=int(str(os.getenv("VOLCENGINE_MAX_RETRIES", "2")).strip() or "2"),
        **kwargs,
    )


def _build_runtime_chat_model(
    *,
    model_name: str,
    temperature: float,
    streaming: bool,
    llm_runtime: dict[str, Any] | None = None,
    preferred_volcengine_model: str = "",
    route_stage: str = "",
) -> Any:
    provider = _runtime_provider(llm_runtime)
    runtime_model = _runtime_model(llm_runtime)
    thinking = _runtime_thinking(llm_runtime)
    route_tag = f"llm_chains.route.{route_stage or 'unknown'}"

    if provider == "volcengine":
        key = _volcengine_api_key()
        if key:
            preferred = str(preferred_volcengine_model or "").strip()
            picked = runtime_model or preferred or str(os.getenv("VOLCENGINE_MODEL", VOLCENGINE_DEFAULT_CHAT_MODEL)).strip()
            if picked not in VOLCENGINE_MODELS:
                fallback = preferred or VOLCENGINE_DEFAULT_CHAT_MODEL
                picked = fallback if fallback in VOLCENGINE_MODELS else VOLCENGINE_DEFAULT_CHAT_MODEL
            log_model_route(
                route_tag,
                {
                    "provider": "volcengine",
                    "streaming": bool(streaming),
                    "thinking": bool(thinking),
                    "requested_model": str(model_name or ""),
                    "runtime_model": runtime_model,
                    "preferred_model": preferred,
                    "picked_model": picked,
                    "temperature": float(temperature),
                },
            )
            return _build_volcengine_chat_model(
                model_name=picked,
                temperature=temperature,
                streaming=streaming,
                thinking=thinking,
            )

    picked_model = runtime_model or model_name
    log_model_route(
        route_tag,
        {
            "provider": "default",
            "streaming": bool(streaming),
            "thinking": False,
            "requested_model": str(model_name or ""),
            "runtime_model": runtime_model,
            "preferred_model": str(preferred_volcengine_model or ""),
            "picked_model": picked_model,
            "temperature": float(temperature),
        },
    )
    return build_chat_model(model_name=picked_model, temperature=temperature, streaming=streaming)


def is_tongyi_enabled(llm_runtime: dict[str, Any] | None = None) -> bool:
    """检查是否已配置聊天模型凭证。"""

    if _runtime_provider(llm_runtime) == "volcengine":
        return bool(_volcengine_api_key())
    return is_chat_enabled()


def _router_llm(llm_runtime: dict[str, Any] | None = None) -> Any:
    """意图识别模型（低温，追求稳定）。"""

    model = os.getenv("TONGYI_ROUTER_MODEL") or os.getenv("TONGYI_MODEL") or DEFAULT_MODEL
    temperature = float(os.getenv("TONGYI_ROUTER_TEMPERATURE", "0.1"))
    preferred_intent_model = str(
        os.getenv("VOLCENGINE_INTENT_MODEL", VOLCENGINE_DEFAULT_INTENT_MODEL)
    ).strip() or VOLCENGINE_DEFAULT_INTENT_MODEL
    # 意图识别走独立模型，不被前端“主对话模型”选择覆盖。
    router_runtime = dict(llm_runtime or {})
    if _runtime_provider(router_runtime) == "volcengine":
        router_runtime["model"] = ""
    return _build_runtime_chat_model(
        model_name=model,
        temperature=temperature,
        streaming=False,
        llm_runtime=router_runtime,
        preferred_volcengine_model=preferred_intent_model,
        route_stage="intent",
    )


def _generator_llm(llm_runtime: dict[str, Any] | None = None) -> Any:
    """主回复模型（温度略高，表达更自然）。"""

    model = os.getenv("TONGYI_MODEL") or DEFAULT_MODEL
    temperature = float(os.getenv("TONGYI_TEMPERATURE", "0.9"))
    preferred_chat_model = str(
        os.getenv("VOLCENGINE_CHAT_MODEL") or os.getenv("VOLCENGINE_MODEL") or VOLCENGINE_DEFAULT_CHAT_MODEL
    ).strip() or VOLCENGINE_DEFAULT_CHAT_MODEL
    return _build_runtime_chat_model(
        model_name=model,
        temperature=temperature,
        streaming=False,
        llm_runtime=llm_runtime,
        preferred_volcengine_model=preferred_chat_model,
        route_stage="response",
    )


def _followup_llm(llm_runtime: dict[str, Any] | None = None) -> Any:
    """猜你想问模型（低温，强调结构化输出）。"""

    model = os.getenv("TONGYI_FOLLOWUP_MODEL") or os.getenv("TONGYI_MODEL") or DEFAULT_MODEL
    temperature = float(os.getenv("TONGYI_FOLLOWUP_TEMPERATURE", "0.2"))
    preferred_chat_model = str(
        os.getenv("VOLCENGINE_CHAT_MODEL") or os.getenv("VOLCENGINE_MODEL") or VOLCENGINE_DEFAULT_CHAT_MODEL
    ).strip() or VOLCENGINE_DEFAULT_CHAT_MODEL
    return _build_runtime_chat_model(
        model_name=model,
        temperature=temperature,
        streaming=False,
        llm_runtime=llm_runtime,
        preferred_volcengine_model=preferred_chat_model,
        route_stage="followup",
    )


def _generator_stream_llm(llm_runtime: dict[str, Any] | None = None) -> Any:
    """流式主回复模型（与主回复同模型，开启 streaming）。"""

    model = os.getenv("TONGYI_MODEL") or DEFAULT_MODEL
    temperature = float(os.getenv("TONGYI_TEMPERATURE", "0.9"))
    preferred_chat_model = str(
        os.getenv("VOLCENGINE_CHAT_MODEL") or os.getenv("VOLCENGINE_MODEL") or VOLCENGINE_DEFAULT_CHAT_MODEL
    ).strip() or VOLCENGINE_DEFAULT_CHAT_MODEL
    return _build_runtime_chat_model(
        model_name=model,
        temperature=temperature,
        streaming=True,
        llm_runtime=llm_runtime,
        preferred_volcengine_model=preferred_chat_model,
        route_stage="stream",
    )


def classify_intent_with_llm(
    query: str,
    fallback_intent: str,
    llm_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """调用 LLM 做意图精修，并保留规则兜底。

    - fallback_intent 来自规则分类；
    - LLM 可用时尝试输出结构化结果；
    - 任意异常时回退到规则，保证线上稳定。
    """

    if not is_tongyi_enabled(llm_runtime):
        # 无模型配置时直接返回规则结果。
        return {
            "intent": fallback_intent,
            "confidence": 0.66 if fallback_intent != "other" else 0.45,
            "needs_handoff": fallback_intent in {"human_service", "after_sales"},
            "source": "rule",
        }

    try:
        parser = PydanticOutputParser(pydantic_object=IntentRouteOutput)
        # 用格式化指令强约束模型输出为可解析 JSON。
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是医疗客服路由器。先判断用户请求属于医疗还是非医疗，再映射到指定意图。"
                    "不做医疗诊断。严格按输出格式返回，不要输出其他文本。"
                    "可选 intent: daily_chat|symptom_consult|medical_knowledge|medication_question|"
                    "lifestyle_guidance|appointment_process|report_interpretation|after_sales|"
                    "human_service|non_medical|other。"
                    "其中 symptom_consult 表示健康问题与症状咨询（非诊断）。",
                ),
                (
                    "human",
                    "用户输入: {query}\n"
                    "规则候选意图: {fallback_intent}\n"
                    "输出格式要求:\n{format_instructions}\n",
                ),
            ]
        )
        payload = {
            "query": query,
            "fallback_intent": fallback_intent,
            "format_instructions": parser.get_format_instructions(),
        }
        log_prompt("llm_chains.classify_intent_with_llm", prompt, payload)

        chain = prompt | _router_llm(llm_runtime)
        raw = chain.invoke(payload)
        raw_text = _extract_text(raw)
        log_response("llm_chains.classify_intent_with_llm", raw_text)

        # 结构化解析失败会进入 except，触发规则兜底。
        parsed = parser.parse(raw_text)
        intent = str(parsed.intent or fallback_intent).strip()
        if intent not in INTENT_LABELS:
            intent = fallback_intent

        confidence = _to_confidence(parsed.confidence, 0.62)
        needs_handoff = bool(parsed.needs_handoff)

        return {
            "intent": intent,
            "confidence": confidence,
            "needs_handoff": needs_handoff,
            "source": "llm",
        }
    except Exception as exc:
        # 模型异常不影响链路可用性：回退规则分类。
        log_error("llm_chains.classify_intent_with_llm", exc)
        return {
            "intent": fallback_intent,
            "confidence": 0.63 if fallback_intent != "other" else 0.45,
            "needs_handoff": fallback_intent in {"human_service", "after_sales"},
            "source": "rule",
        }


def _response_prompt() -> ChatPromptTemplate:
    """主回复提示词模板。

    把 intent 处理边界、安全边界、输出格式集中写在 system prompt，
    让不同场景行为更一致、可控。
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是医疗客服助手。"
                "你会收到意图(intent)、风险等级(risk_level)、工具结果、知识库和在线信息。"
                "请优先参考当前 intent 的建议组织回复，不要机械套模板。"
                "\n"
                "当前 intent 的优先建议（软约束，可自然表达）:"
                "\n{intent_guidance}"
                "\n"
                "安全边界："
                "\n- 严禁做诊断、开处方、替代医生。"
                "\n- 若 risk_level=high 或信息不足且存在风险，必须明确建议线下就医或转人工。"
                "\n- 若 handoff=true，应在结尾明确建议人工复核。"
                "\n"
                "输出要求："
                "\n- 仅输出中文。"
                "\n- 结构清晰，允许自然段或分点，不强制固定条数。"
                "\n- 先给结论/建议，再给依据或注意事项。"
                "\n- 不要输出与医疗安全无关的冗余内容。",
            ),
            (
                "human",
                "用户问题: {query}\n"
                "意图: {intent}\n"
                "风险等级: {risk_level}\n"
                "工具结果: {tool_results}\n"
                "知识库: {context_text}\n"
                "引用ID: {citations}\n"
                "短期记忆: {conversation_history}\n"
                "Tavily在线信息: {online_context}\n"
                "是否建议转人工: {handoff}\n"
                "请给出最终回复。",
            ),
        ]
    )


def _build_response_inputs(
    *,
    query: str,
    intent: str,
    risk_level: str,
    tool_results: dict[str, Any],
    context_docs: list[dict[str, Any]],
    citations: list[str],
    handoff: bool,
    enable_online_search: bool,
    conversation_history_text: str = "",
) -> dict[str, str]:
    """将工作流状态整理成主回复链输入。"""
    # 只拼接前 3 条知识库，控制提示词长度，避免噪声过多。
    context_lines = [
        f"- {doc.get('title', '')}: {doc.get('content', '')}" for doc in context_docs[:3]
    ]
    context_text = "\n".join(context_lines) if context_lines else "(无检索结果)"
    # 是否联网由前端开关控制；关闭时 online_search 模块会返回占位说明。
    online_context = fetch_tavily_context(
        query,
        max_results=3,
        topic="general",
        intent_hint=intent,
        enable_online=enable_online_search,
    )

    return {
        "query": query,
        "intent": intent,
        "risk_level": risk_level,
        "intent_guidance": _intent_soft_guidance(intent),
        "tool_results": json.dumps(tool_results, ensure_ascii=False),
        "context_text": context_text,
        "citations": ", ".join(citations) if citations else "none",
        "online_context": online_context,
        "handoff": str(handoff).lower(),
        "conversation_history": _summarize_history(conversation_history_text, max_len=520),
    }


def generate_response_with_llm(
    *,
    query: str,
    intent: str,
    risk_level: str,
    tool_results: dict[str, Any],
    context_docs: list[dict[str, Any]],
    citations: list[str],
    handoff: bool,
    enable_online_search: bool = False,
    conversation_history_text: str = "",
    llm_runtime: dict[str, Any] | None = None,
) -> str | None:
    """非流式生成主回复；失败返回 None 让上层规则答案兜底。"""

    runtime_for_response = _response_runtime_for_intent(llm_runtime=llm_runtime, intent=intent)
    if not is_tongyi_enabled(runtime_for_response):
        return None

    try:
        prompt = _response_prompt()
        payload = _build_response_inputs(
            query=query,
            intent=intent,
            risk_level=risk_level,
            tool_results=tool_results,
            context_docs=context_docs,
            citations=citations,
            handoff=handoff,
            enable_online_search=enable_online_search,
            conversation_history_text=conversation_history_text,
        )
        log_prompt("llm_chains.generate_response_with_llm", prompt, payload)

        chain = prompt | _generator_llm(runtime_for_response)
        raw = chain.invoke(payload)
        answer = _extract_text(raw)
        log_response("llm_chains.generate_response_with_llm", answer)
        return answer or None
    except Exception as exc:
        # 只记录错误，不抛出，避免中断主流程。
        log_error("llm_chains.generate_response_with_llm", exc)
        return None


def stream_response_with_llm(
    *,
    query: str,
    intent: str,
    risk_level: str,
    tool_results: dict[str, Any],
    context_docs: list[dict[str, Any]],
    citations: list[str],
    handoff: bool,
    enable_online_search: bool = False,
    conversation_history_text: str = "",
    llm_runtime: dict[str, Any] | None = None,
) -> Iterator[str] | None:
    """流式生成主回复，返回 token 迭代器。"""

    runtime_for_response = _response_runtime_for_intent(llm_runtime=llm_runtime, intent=intent)
    if not is_tongyi_enabled(runtime_for_response):
        return None

    def _gen() -> Iterator[str]:
        # 与非流式复用同一模板与输入，确保语义一致。
        prompt = _response_prompt()
        payload = _build_response_inputs(
            query=query,
            intent=intent,
            risk_level=risk_level,
            tool_results=tool_results,
            context_docs=context_docs,
            citations=citations,
            handoff=handoff,
            enable_online_search=enable_online_search,
            conversation_history_text=conversation_history_text,
        )
        log_prompt("llm_chains.stream_response_with_llm", prompt, payload)

        chain = prompt | _generator_stream_llm(runtime_for_response)
        collected: list[str] = []
        try:
            for chunk in chain.stream(payload):
                text = _extract_chunk_text(chunk)
                if text:
                    collected.append(text)
                    yield text
        except Exception as exc:
            log_error("llm_chains.stream_response_with_llm", exc)
            raise
        finally:
            # 收尾时记录完整文本，方便排查流式输出问题。
            if collected:
                log_response("llm_chains.stream_response_with_llm", "".join(collected))

    try:
        return _gen()
    except Exception as exc:
        log_error("llm_chains.stream_response_with_llm", exc)
        return None


def generate_followups_with_llm(
    *,
    query: str,
    intent: str,
    risk_level: str,
    conversation_history_text: str = "",
    assistant_profile: str = DEFAULT_ASSISTANT_PROFILE,
    llm_runtime: dict[str, Any] | None = None,
) -> list[str] | None:
    """独立链路生成“猜你想问”。

    只输入：用户问题 + 意图/风险 + 短期记忆。
    不灌完整历史，避免污染主对话链路。
    """

    followup_runtime = _with_forced_volcengine_model(
        llm_runtime,
        forced_model=VOLCENGINE_FAST_MODEL,
        force_thinking_false=True,
        reason="followup_fast_path",
        intent=str(intent or "").strip().lower(),
    )
    if not is_tongyi_enabled(followup_runtime):
        return None

    try:
        parser = PydanticOutputParser(pydantic_object=FollowupOutput)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是医疗客服助手的追问生成模块。"
                    "只负责生成用户下一步最可能追问你的3个问题。"
                    "必须严格按输出格式返回 JSON，不要任何额外文本。"
                    "问题要简短、自然。注意是用户的提问，要站在用户的角度。",
                ),
                (
                    "human",
                    "助手能力范围: {assistant_profile}\n"
                    "用户问题: {query}\n"
                    "意图: {intent}\n"
                    "风险等级: {risk_level}\n"
                    "短期记忆: {history_summary}\n"
                    "输出格式要求:\n{format_instructions}\n",
                ),
            ]
        )

        payload = {
            "query": query,
            "intent": intent,
            "risk_level": risk_level,
            "assistant_profile": str(assistant_profile or DEFAULT_ASSISTANT_PROFILE),
            "history_summary": _summarize_history(conversation_history_text, max_len=420),
            "format_instructions": parser.get_format_instructions(),
        }
        log_prompt("llm_chains.generate_followups_with_llm", prompt, payload)

        chain = prompt | _followup_llm(followup_runtime)
        raw = chain.invoke(payload)
        raw_text = _extract_text(raw)
        log_response("llm_chains.generate_followups_with_llm", raw_text)

        parsed = parser.parse(raw_text)
        questions = _normalize_questions(list(parsed.questions or []))
        return questions if questions else None
    except Exception as exc:
        # 追问失败不影响主回答，返回 None 交给上层默认追问。
        log_error("llm_chains.generate_followups_with_llm", exc)
        return None
