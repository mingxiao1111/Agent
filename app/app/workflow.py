from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from langgraph.graph import END, START, StateGraph

from .guardrails import classify_intent, classify_intent_candidates, detect_high_risk, normalize_text
from .llm_chains import (
    INTENT_LABELS,
    classify_intent_with_llm,
    generate_followups_with_llm,
    generate_response_with_llm,
)
from .state import AgentState
from .tcm import search_tcm_patent_medicines
from .tools import (
    extract_drug_names,
    get_doctor_schedule,
    get_drug_leaflet,
    kb_search,
    recommend_department,
)

LOW_CONFIDENCE_HANDOFF_THRESHOLD = 0.40
HANDOFF_INTENTS = {"human_service", "after_sales"}
MEDICATION_SAFETY_KEYWORDS = (
    "药",
    "用药",
    "服用",
    "同服",
    "联用",
    "相互作用",
    "禁忌",
    "副作用",
    "不良反应",
    "保健品",
    "健康品",
    "中成药",
    "处方药",
    "非处方",
)


def _merge_secondary_intents(primary: str, *groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen = {str(primary or "").strip()}
    for group in groups:
        for item in group:
            intent = str(item or "").strip()
            if intent not in INTENT_LABELS or intent in seen:
                continue
            seen.add(intent)
            merged.append(intent)
            if len(merged) >= 2:
                return merged
    return merged


def _active_intents(state: AgentState) -> list[str]:
    primary = str(state.get("intent", "other") or "other").strip() or "other"
    secondary = state.get("secondary_intents", [])
    if not isinstance(secondary, list):
        secondary = []
    return [primary] + _merge_secondary_intents(primary, secondary)


def _compact_patent_hits(items: list[dict], limit: int = 3) -> list[dict]:
    compact: list[dict] = []
    seen: set[str] = set()

    for item in items:
        name = str(item.get("title", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)

        fit_for = str(item.get("indication", "")).strip() or "请结合辨证与症状判断适用性"
        cautions = (
            str(item.get("contraindications", "")).strip()
            or str(item.get("cautions", "")).strip()
            or "请先阅读说明书并咨询医生/药师"
        )
        source = f"{item.get('file', '')}:{item.get('line_no', 0)}"

        compact.append(
            {
                "name": name,
                "fit_for": fit_for[:160],
                "caution": cautions[:180],
                "source": source.strip(":"),
            }
        )
        if len(compact) >= max(1, min(limit, 5)):
            break
    return compact


def _is_medication_safety_query(*texts: str) -> bool:
    merged = " ".join(str(t or "").lower() for t in texts)
    return any(keyword in merged for keyword in MEDICATION_SAFETY_KEYWORDS)


def normalize_node(state: AgentState) -> AgentState:
    text = state.get("user_input", "")
    return {
        "normalized_input": normalize_text(text),
        "tool_results": {},
        "context_docs": [],
        "citations": [],
    }


def risk_node(state: AgentState) -> AgentState:
    text = state.get("normalized_input", "")
    is_high, reason = detect_high_risk(text)
    return {
        "risk_level": "high" if is_high else "low",
        "risk_reason": reason,
    }


def risk_route(state: AgentState) -> Literal["emergency", "continue"]:
    return "emergency" if state.get("risk_level") == "high" else "continue"


def emergency_node(state: AgentState) -> AgentState:
    raw = state.get("user_input", "")
    reason = state.get("risk_reason", "high_risk")
    summary = f"高危分诊触发({reason})，用户原话: {raw}"
    if str(reason).startswith(("keyword_risk:", "regex_risk:")):
        answer = (
            "检测到高风险关键词内容，当前无法继续该请求。"
            "请调整为合规、医疗相关的咨询表述，或直接转人工处理。"
        )
        follow_ups = [
            "请删去高风险关键词后重试",
            "请仅保留医疗健康咨询内容",
            "如需人工协助，请直接输入“转人工”",
        ]
        citations = ["risk_policy_v1"]
    else:
        answer = (
            "检测到高风险症状。请立即前往最近急诊或拨打当地急救电话。"
            "在等待专业救治期间，避免自行用药。"
            "我已为你生成人工接管摘要。"
        )
        follow_ups = [
            "去急诊前需要准备哪些资料？",
            "急诊就诊时我该如何描述症状？",
            "现在在家等待时有什么注意事项？",
        ]
        citations = ["safety_protocol_v1"]
    return {
        "intent": "emergency",
        "intent_source": "rule",
        "intent_confidence": 1.0,
        "handoff_hint": True,
        "confidence": 1.0,
        "handoff": True,
        "handoff_summary": summary,
        "answer": answer,
        "citations": citations,
        "follow_ups": follow_ups,
    }


def intent_node(state: AgentState) -> AgentState:
    text = state.get("normalized_input", "")
    fallback_intent = classify_intent(text)
    fallback_candidates = classify_intent_candidates(text, max_items=3)
    routed = classify_intent_with_llm(
        query=text,
        fallback_intent=fallback_intent,
        llm_runtime={
            "provider": state.get("llm_provider", "default"),
            "model": state.get("llm_model", ""),
            "thinking": bool(state.get("llm_thinking", False)),
        },
    )
    primary_intent = str(routed.get("intent", fallback_intent) or fallback_intent).strip() or fallback_intent
    secondary_intents = _merge_secondary_intents(
        primary_intent,
        list(routed.get("secondary_intents", []) or []),
        [item for item in fallback_candidates if item != primary_intent],
    )
    return {
        "intent": primary_intent,
        "secondary_intents": secondary_intents,
        "intent_candidates": [primary_intent] + secondary_intents,
        "intent_confidence": routed["confidence"],
        "handoff_hint": routed["needs_handoff"],
        "intent_source": routed["source"],
    }


def retrieve_node(state: AgentState) -> AgentState:
    docs = kb_search(state.get("normalized_input", ""), k=3)
    citations = [doc["id"] for doc in docs]
    return {
        "context_docs": docs,
        "citations": citations,
    }


def tools_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "other")
    active_intents = _active_intents(state)
    intent_set = set(active_intents)
    text = state.get("normalized_input", "")
    user_text = state.get("user_input", "")

    results: dict = {}
    next_context_docs = list(state.get("context_docs", []))
    next_citations = list(state.get("citations", []))
    if "symptom_consult" in intent_set:
        department = recommend_department(text)
        results["department"] = department
        results["schedule"] = get_doctor_schedule(department)

    if "appointment_process" in intent_set:
        department = recommend_department(text)
        results["department"] = department
        results["schedule"] = get_doctor_schedule(department)
        results["booking_steps"] = "先完成实名认证，再选择科室与时段，最后支付挂号费。"

    if "medication_question" in intent_set or (
        "medical_knowledge" in intent_set and _is_medication_safety_query(text, user_text)
    ):
        try:
            patent_hits = search_tcm_patent_medicines(
                query=user_text or text,
                symptoms=[],
                top_k=8,
            )
        except Exception:
            patent_hits = []

        compact_hits = _compact_patent_hits(patent_hits, limit=3)
        if compact_hits:
            results["patent_medicine_candidates"] = compact_hits

            for row in compact_hits:
                src = str(row.get("source", "")).strip() or "patent_db"
                cid = f"patent:{src}"
                next_citations.append(cid)
                next_context_docs.append(
                    {
                        "id": cid,
                        "title": f"中成药: {row.get('name', '')}",
                        "content": f"适用={row.get('fit_for', '')}；禁忌/注意={row.get('caution', '')}",
                    }
                )

        drug_names = extract_drug_names(state.get("user_input", ""))
        if drug_names:
            results["drug_leaflets"] = {name: get_drug_leaflet(name) for name in drug_names}
        else:
            results["drug_leaflets"] = {
                "unknown": {
                    "indication": "请提供药品通用名",
                    "caution": "避免自行联合用药",
                    "interaction": "需药师复核",
                }
            }

    if "report_interpretation" in intent_set:
        results["report_notice"] = "可提供指标含义解释，但不能替代医生诊断。"

    return {
        "tool_results": results,
        "context_docs": next_context_docs,
        "citations": list(dict.fromkeys(next_citations)),
    }


def _calculate_confidence(state: AgentState) -> float:
    confidence = 0.40
    if state.get("intent") and state.get("intent") != "other":
        confidence += 0.20
    confidence += min(0.24, len(state.get("context_docs", [])) * 0.08)
    if state.get("tool_results"):
        confidence += 0.15

    intent_conf = state.get("intent_confidence")
    if isinstance(intent_conf, (int, float)):
        intent_conf = max(0.0, min(1.0, float(intent_conf)))
        confidence = confidence * 0.7 + intent_conf * 0.3

    return min(confidence, 0.95)


def _render_context_lines(state: AgentState) -> list[str]:
    lines: list[str] = []
    for doc in state.get("context_docs", []):
        lines.append(f"- {doc['title']}: {doc['content']}")
    return lines


def _build_rule_answer(state: AgentState, intent: str, handoff: bool) -> str:
    lines = ["以下是基于规则与知识库的建议，仅供分诊和就医流程参考:"]
    secondary_intents = _merge_secondary_intents(intent, list(state.get("secondary_intents", []) or []))

    if secondary_intents:
        lines.append(f"补充识别到复合诉求，除主问题外还涉及: {'、'.join(secondary_intents)}。")

    if intent == "symptom_consult":
        department = state.get("tool_results", {}).get("department", "全科")
        schedule = state.get("tool_results", {}).get("schedule", "请在医院小程序查看实时号源")
        lines.append("1) 你当前属于健康问题与症状咨询（非诊断）场景。")
        lines.append(f"2) 建议先挂 {department}。")
        lines.append(f"3) 该科室常规排班: {schedule}。")
        lines.append("4) 若症状加重或出现呼吸困难/持续胸痛，请立即急诊。")

    elif intent == "appointment_process":
        steps = state.get("tool_results", {}).get("booking_steps", "先实名认证，再选科室和时间。")
        department = state.get("tool_results", {}).get("department", "全科")
        lines.append("1) 你当前是就医咨询/挂号流程场景。")
        lines.append(f"2) 推荐科室: {department}。")
        lines.append(f"3) 挂号步骤: {steps}")
        if "symptom_consult" in secondary_intents:
            lines.append("4) 结合你提到的不适症状，若出现持续加重、呼吸困难或胸痛，请及时线下就医。")

    elif intent == "medication_question":
        patent_candidates = state.get("tool_results", {}).get("patent_medicine_candidates", [])
        leaflets = state.get("tool_results", {}).get("drug_leaflets", {})
        lines.append("1) 我先检索了中成药知识库，以下仅用于用药与健康品安全科普。")
        if patent_candidates:
            lines.append("2) 中成药库相关条目：")
            for item in patent_candidates:
                lines.append(
                    f"- {item.get('name', '未命名药品')}："
                    f"适用={item.get('fit_for', '请结合说明书')}；"
                    f"禁忌/注意={item.get('caution', '请先咨询医生/药师')}"
                )
        else:
            lines.append("2) 中成药库未检索到高相关条目，建议补充具体药名或核心症状。")

        lines.append("3) 已识别药品说明书要点：")
        for drug_name, details in leaflets.items():
            lines.append(
                f"- {drug_name}: 适应症={details['indication']}；注意={details['caution']}；相互作用={details['interaction']}"
            )
        lines.append("4) 不提供个体化处方建议；如涉及孕期/慢病/儿童，请转人工药师。")
        if "report_interpretation" in secondary_intents:
            lines.append("5) 若你还想结合检查结果判断用药风险，请继续补充关键指标或报告结论。")

    elif intent == "report_interpretation":
        lines.append("1) 你当前是报告解读场景，我可以解释指标含义和常见影响因素。")
        lines.append("2) 检查结果异常是否需要治疗，需由临床医生结合病史判断。")
        if "appointment_process" in secondary_intents:
            lines.append("3) 如果你希望继续安排就诊，我也可以顺带帮你梳理挂号方向。")

    elif intent == "medical_knowledge":
        patent_candidates = state.get("tool_results", {}).get("patent_medicine_candidates", [])
        leaflets = state.get("tool_results", {}).get("drug_leaflets", {})

        if patent_candidates:
            lines.append("1) 检测到你在咨询用药/健康品安全科普，我已先检索中成药库。")
            lines.append("2) 中成药库相关条目：")
            for item in patent_candidates:
                lines.append(
                    f"- {item.get('name', '未命名药品')}："
                    f"适用={item.get('fit_for', '请结合说明书')}；"
                    f"禁忌/注意={item.get('caution', '请先咨询医生/药师')}"
                )
            if leaflets:
                lines.append("3) 药品说明书要点：")
                for drug_name, details in leaflets.items():
                    lines.append(
                        f"- {drug_name}: 适应症={details['indication']}；注意={details['caution']}；相互作用={details['interaction']}"
                    )
            lines.append("4) 以上仅供安全科普，不替代医生面诊或个体化处方。")
        else:
            lines.append("1) 我可以做医学科普与疾病知识查询。")
            lines.append("2) 若你有具体症状，请补充主诉和持续时间，我会转到症状咨询流程。")
        if "appointment_process" in secondary_intents:
            lines.append("3) 如果你同时想了解就诊流程，我也可以继续帮你判断挂哪个科。")

    elif intent == "lifestyle_guidance":
        lines.append("1) 我可以提供健康生活方式建议（饮食/睡眠/运动/作息）。")
        lines.append("2) 若伴随明显不适症状，建议同步线下就医评估。")

    elif intent == "after_sales":
        lines.append("1) 你当前是产品/服务售后咨询场景。")
        lines.append("2) 建议转人工客服处理订单、退款、发票或投诉问题。")

    elif intent == "human_service":
        lines.append("1) 你已请求人工服务。")
        lines.append("2) 我将优先为你触发人工接管。")

    elif intent == "non_medical":
        lines.append("1) 当前问题不属于医疗健康范围。")
        lines.append("2) 如需医疗咨询，请描述症状、就医流程或报告指标。")

    elif intent == "daily_chat":
        lines.append("1) 我可以协助健康问题与症状咨询、就医咨询、报告解读和用药科普。")
        lines.append("2) 你可以直接描述具体问题，我会按医疗客服流程处理。")

    else:
        lines.append("1) 我暂时无法稳定识别你的问题类型，建议转人工客服。")

    context_lines = _render_context_lines(state)
    if context_lines:
        lines.append("参考依据:")
        lines.extend(context_lines)

    if handoff:
        lines.append("系统建议: 已触发人工复核，以降低医疗风险。")

    return "\n".join(lines)


def _default_followups(intent: str, risk_level: str, query: str) -> list[str]:
    if risk_level == "high":
        return [
            "去急诊前需要准备哪些资料？",
            "急诊就诊时我该如何描述症状？",
            "现在在家等待时有什么注意事项？",
        ]

    if intent == "symptom_consult":
        return [
            "需要先做哪些基础检查？",
            "什么情况下必须立刻去急诊？",
            "线上复诊和线下就诊怎么选？",
        ]

    if intent == "appointment_process":
        return [
            "挂号后错过时间可以改约吗？",
            "初诊和复诊挂号有什么区别？",
            "医保卡和身份证都要带吗？",
        ]

    if intent == "medication_question":
        return [
            "两种药间隔多久服用更安全？",
            "出现哪些副作用需要立刻停药就医？",
            "慢病/孕期/儿童用药需要转人工药师吗？",
        ]

    if intent == "report_interpretation":
        return [
            "这个指标异常通常还要复查哪些项目？",
            "报告异常但没症状需要马上就医吗？",
            "复查前饮食和作息要注意什么？",
        ]

    if intent == "medical_knowledge":
        return [
            "这个病常见早期症状有哪些？",
            "什么情况下需要尽快去医院？",
            "日常该怎么预防复发？",
        ]

    if intent == "lifestyle_guidance":
        return [
            "能给我一个一周作息和运动建议吗？",
            "饮食上我优先改哪三件事？",
            "哪些信号说明需要线下就医？",
        ]

    if intent == "after_sales":
        return [
            "帮我转人工客服处理售后",
            "需要提供哪些订单信息？",
            "售后处理一般多久？",
        ]

    if intent == "human_service":
        return [
            "帮我尽快转人工",
            "人工服务时间是几点到几点？",
            "转人工前我需要准备什么信息？",
        ]

    if intent == "non_medical":
        return [
            "我有健康问题，应该怎么描述更清楚？",
            "你可以做哪些医疗相关咨询？",
            "可以帮我做就医流程咨询吗？",
        ]

    if intent == "daily_chat":
        return [
            "你可以先帮我判断该挂哪个科吗？",
            "我有症状想咨询，怎么描述最有效？",
            "我有报告想看，应该发哪些信息？",
        ]

    return [
        "我该挂哪个科室更合适？",
        "这种情况需要转人工客服吗？",
        "就诊前需要准备什么材料？",
    ]


def _merged_followups(intents: list[str], risk_level: str, query: str, limit: int = 4) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for intent in intents:
        for item in _default_followups(intent, risk_level, query):
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
            if len(ordered) >= limit:
                return ordered
    return ordered or _default_followups("other", risk_level, query)


def _ensure_tcm_entry(follow_ups: list[str], intent: str) -> list[str]:
    tcm_entry = "进入中医辨证问诊模式"
    if intent not in {"symptom_consult", "other", "report_interpretation"}:
        return follow_ups
    if tcm_entry not in follow_ups:
        follow_ups = [tcm_entry] + follow_ups
    return follow_ups[:4]


def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "other")
    active_intents = _active_intents(state)
    risk_level = state.get("risk_level", "low")
    confidence = _calculate_confidence(state)

    handoff = (
        risk_level == "high"
        or any(item in HANDOFF_INTENTS for item in active_intents)
        or confidence < LOW_CONFIDENCE_HANDOFF_THRESHOLD
    )

    answer = _build_rule_answer(state, intent, handoff)

    disable_llm_response = bool(state.get("disable_llm_response", False))
    disable_llm_followups = bool(state.get("disable_llm_followups", False))
    history_text = str(state.get("conversation_history_text", "") or "")
    llm_runtime = {
        "provider": state.get("llm_provider", "default"),
        "model": state.get("llm_model", ""),
        "thinking": bool(state.get("llm_thinking", False)),
    }

    prefetched_followups = state.get("prefetched_followups", [])
    followups = (
        list(prefetched_followups)
        if isinstance(prefetched_followups, list) and prefetched_followups
        else _merged_followups(active_intents, risk_level, state.get("user_input", ""))
    )
    followups_future = None
    followups_executor = None
    if (not disable_llm_followups) and not (
        isinstance(prefetched_followups, list) and prefetched_followups
    ):
        # 猜你想问不依赖主回复内容，提前并行生成以减少整体等待。
        followups_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="followup_prefetch")
        followups_future = followups_executor.submit(
            generate_followups_with_llm,
            query=state.get("user_input", ""),
            intent=intent,
            secondary_intents=active_intents[1:],
            risk_level=risk_level,
            conversation_history_text=history_text,
            llm_runtime=llm_runtime,
        )

    if not disable_llm_response:
        llm_answer = generate_response_with_llm(
            query=state.get("user_input", ""),
            intent=intent,
            secondary_intents=active_intents[1:],
            risk_level=risk_level,
            tool_results=state.get("tool_results", {}),
            context_docs=state.get("context_docs", []),
            citations=state.get("citations", []),
            handoff=handoff,
            enable_online_search=bool(state.get("enable_online_search", False)),
            conversation_history_text=history_text,
            llm_runtime=llm_runtime,
        )
        if llm_answer:
            answer = llm_answer

    citations = state.get("citations", [])
    if citations and "参考来源ID:" not in answer:
        answer = f"{answer}\n\n参考来源ID: {', '.join(citations)}"

    if handoff and "人工" not in answer:
        answer = f"{answer}\n系统建议: 已触发人工复核，以降低医疗风险。"

    if followups_future is not None:
        try:
            llm_followups = followups_future.result(timeout=0.9)
            if llm_followups:
                followups = llm_followups
        except Exception:
            pass
        finally:
            if followups_executor is not None:
                followups_executor.shutdown(wait=False)

    follow_ups = _ensure_tcm_entry(followups, intent)
    summary = (
        f"intent={intent}; secondary={','.join(active_intents[1:]) or 'none'}; "
        f"source={state.get('intent_source', 'rule')}; risk={risk_level}; "
        f"query={state.get('user_input', '')}; confidence={confidence:.2f}"
    )
    return {
        "confidence": confidence,
        "handoff": handoff,
        "answer": answer,
        "handoff_summary": summary,
        "follow_ups": follow_ups,
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("normalize", normalize_node)
    graph.add_node("risk", risk_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("intent", intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tools", tools_node)
    graph.add_node("respond", response_node)

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "risk")
    graph.add_conditional_edges(
        "risk",
        risk_route,
        {
            "emergency": "emergency",
            "continue": "intent",
        },
    )
    graph.add_edge("emergency", END)
    graph.add_edge("intent", "retrieve")
    graph.add_edge("retrieve", "tools")
    graph.add_edge("tools", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


APP = build_graph()


def run_agent(
    user_input: str,
    *,
    llm_provider: str = "default",
    llm_model: str = "",
    llm_thinking: bool = False,
    conversation_history_text: str = "",
    disable_llm_response: bool = False,
    disable_llm_followups: bool = False,
    enable_online_search: bool = False,
) -> AgentState:
    payload = {
        "user_input": user_input,
        "llm_provider": str(llm_provider or "default"),
        "llm_model": str(llm_model or ""),
        "llm_thinking": bool(llm_thinking),
        "conversation_history_text": str(conversation_history_text or ""),
        "disable_llm_response": disable_llm_response,
        "disable_llm_followups": disable_llm_followups,
        "enable_online_search": enable_online_search,
    }
    return APP.invoke(payload)
