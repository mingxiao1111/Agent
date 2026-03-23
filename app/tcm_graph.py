from __future__ import annotations

"""中医辨证工作流（LangGraph）。

本文件把流程拆为两张图：
1) 首轮收集图：症状收集 -> 红旗筛查 -> RAG -> 候选证候 -> 首轮问卷；
2) 多轮复判图：合并答案 -> 复检索复评分 -> 继续问卷或输出阶段结果。
"""

import os
from typing import Any, Iterator, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .tcm import (
    build_tcm_questionnaire,
    collect_question_keys,
    compute_tcm_confidence,
    detect_tcm_red_flags,
    extract_tcm_symptoms,
    finalize_tcm_assessment,
    infer_tcm_syndrome_candidates,
    normalize_tcm_answer,
    search_tcm_cases_tool,
    summarize_questionnaire_answers,
)


class TcmCollectState(TypedDict, total=False):
    """首轮（收集阶段）状态。"""

    history: list[str]
    user_input: str
    round: int
    asked_question_keys: list[str]
    combined_text: str
    symptom_profile: dict[str, Any]
    extraction_failed: bool
    extraction_error: str
    need_more: bool
    red_flags: list[str]
    case_refs: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    questionnaire: list[dict[str, Any]]
    confidence: float
    done: bool
    message: str


class TcmRoundState(TypedDict, total=False):
    """多轮（问卷复判阶段）状态。"""

    history: list[str]
    round: int
    asked_question_keys: list[str]
    answers_history: list[str]
    combined_text: str
    symptom_profile: dict[str, Any]
    candidates: list[dict[str, Any]]
    questionnaire: list[dict[str, Any]]
    answers: dict[str, Any]
    case_refs: list[dict[str, Any]]
    red_flags: list[str]
    confidence: float
    done: bool
    stop_now: bool
    result: dict[str, Any]
    message: str


def _confidence_threshold() -> float:
    """读取置信度阈值，达到后可结束流程。"""

    try:
        val = float(os.getenv("TCM_CONFIDENCE_THRESHOLD", "0.75"))
    except ValueError:
        val = 0.75
    return max(0.5, min(0.95, val))


def _max_rounds() -> int:
    """读取最大轮次，防止问诊无限循环。"""

    try:
        val = int(os.getenv("TCM_MAX_ROUNDS", "3"))
    except ValueError:
        val = 3
    return max(1, min(6, val))


def _first_question_count() -> int:
    """首轮问卷题数。"""

    try:
        val = int(os.getenv("TCM_QUESTION_COUNT", "10"))
    except ValueError:
        val = 10
    return max(6, min(12, val))


def _next_question_count(round_no: int) -> int:
    """后续轮次题数策略（当前与首轮一致）。"""

    base = _first_question_count()
    if round_no <= 1:
        return base
    return max(6, min(12, base))


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    """格式化候选证候，供用户侧消息展示。"""

    if not candidates:
        return "- 暂无稳定候选证候"
    rows: list[str] = []
    for c in candidates[:3]:
        name = str(c.get("name", "待进一步辨证"))
        try:
            score = float(c.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        reason = str(c.get("reason", "")).strip() or "暂无说明"
        rows.append(f"- {name} (score={score:.2f})：{reason}")
    return "\n".join(rows)


def merge_history_node(state: TcmCollectState) -> TcmCollectState:
    """合并历史与当前输入，产出本轮统一文本。"""

    history = list(state.get("history", []))
    user_input = str(state.get("user_input", "")).strip()
    if user_input:
        history.append(user_input)
    return {
        "history": history,
        "combined_text": "；".join(history[-10:]),
        "round": int(state.get("round", 0)),
        "asked_question_keys": list(state.get("asked_question_keys", [])),
    }


def extract_symptoms_node(state: TcmCollectState) -> TcmCollectState:
    """调用症状提取并判断是否“症状不足”。"""

    combined_text = state.get("combined_text", "")
    profile = extract_tcm_symptoms(combined_text)
    symptoms = profile.get("symptoms", []) if isinstance(profile, dict) else []
    extraction_failed = not bool(profile.get("extraction_ok", True))
    need_more = len(symptoms) < 3
    return {
        "symptom_profile": profile,
        "need_more": need_more,
        "extraction_failed": extraction_failed,
        "extraction_error": str(profile.get("extraction_error", "")),
    }


def precheck_node(state: TcmCollectState) -> TcmCollectState:
    """首轮红旗筛查。"""

    red_flags = detect_tcm_red_flags(state.get("combined_text", ""))
    return {"red_flags": red_flags}


def collect_route(state: TcmCollectState) -> Literal["red_flag", "extract_fail", "need_more", "diagnose"]:
    """首轮路由：高风险/提取失败/信息不足/进入辨证。"""

    if state.get("red_flags"):
        return "red_flag"
    if state.get("extraction_failed", False):
        return "extract_fail"
    return "need_more" if state.get("need_more", True) else "diagnose"


def red_flag_node(state: TcmCollectState) -> TcmCollectState:
    """命中高风险后立即终止线上辨证。"""

    red_flags = state.get("red_flags", [])
    msg = (
        "检测到您可能伴有高风险症状："
        f"{'、'.join(red_flags)}。\n"
        "建议您优先线下就医或急诊处理。当前中医辨证流程暂停。"
    )
    return {
        "done": True,
        "message": msg,
        "questionnaire": [],
        "candidates": [],
        "case_refs": [],
        "confidence": 0.0,
    }


def extract_fail_node(state: TcmCollectState) -> TcmCollectState:
    """症状提取失败时的兜底提示节点。"""

    error = str(state.get("extraction_error", "")).strip() or "未知错误"
    msg = (
        "症状提取模型调用失败，当前未进入辨证检索。\n"
        f"失败原因：{error}\n"
        "请检查 DASHSCOPE_API_KEY 与网络/代理/防火墙后重试。"
    )
    return {
        "done": False,
        "message": msg,
        "questionnaire": [],
        "candidates": [],
        "case_refs": [],
        "confidence": 0.0,
    }


def need_more_node(state: TcmCollectState) -> TcmCollectState:
    """信息不足时，引导补充症状。"""

    profile = state.get("symptom_profile", {})
    symptoms = profile.get("symptoms", []) if isinstance(profile, dict) else []
    symptom_text = "、".join(symptoms) if symptoms else "（当前未提取到可用症状，请按口语再描述一次）"
    msg = (
        "我先记录到这些症状："
        f"{symptom_text}。\n"
        "为了提高辨证的准确性，请您再补充几个具体的症状，比如睡眠、口渴、食欲、二便、畏寒发热等。"
    )
    return {
        "done": False,
        "message": msg,
        "questionnaire": [],
        "candidates": [],
        "case_refs": [],
        "confidence": 0.0,
    }


def retrieve_cases_node(state: TcmCollectState) -> TcmCollectState:
    """RAG 检索中医医案证据。"""

    combined_text = state.get("combined_text", "")
    profile = state.get("symptom_profile", {})
    symptoms = profile.get("symptoms", []) if isinstance(profile, dict) else []

    case_refs = search_tcm_cases_tool.invoke(
        {
            "query": combined_text,
            "symptoms": symptoms,
            "top_k": 10,
        }
    )
    return {"case_refs": case_refs}


def infer_candidates_node(state: TcmCollectState) -> TcmCollectState:
    """基于症状+医案推断候选证候。"""

    candidates = infer_tcm_syndrome_candidates(
        user_text=state.get("combined_text", ""),
        symptom_profile=state.get("symptom_profile", {}),
        case_refs=state.get("case_refs", []),
    )
    return {"candidates": candidates}


def collect_score_node(state: TcmCollectState) -> TcmCollectState:
    """首轮粗评分（未含问卷答案）。"""

    confidence = compute_tcm_confidence(
        candidates=state.get("candidates", []),
        answers={},
        questionnaire=[],
    )
    return {"confidence": confidence}


def questionnaire_node(state: TcmCollectState) -> TcmCollectState:
    """生成首轮区分问卷。"""

    questionnaire = build_tcm_questionnaire(
        user_text=state.get("combined_text", ""),
        symptom_profile=state.get("symptom_profile", {}),
        candidates=state.get("candidates", []),
        case_refs=state.get("case_refs", []),
        asked_question_keys=state.get("asked_question_keys", []),
        target_count=_first_question_count(),
    )
    return {"questionnaire": questionnaire}


def ready_node(state: TcmCollectState) -> TcmCollectState:
    """首轮完成后输出给前端的消息。"""

    candidates = state.get("candidates", [])
    confidence = float(state.get("confidence", 0.0))
    new_keys = collect_question_keys(state.get("questionnaire", []))
    asked = list(dict.fromkeys(list(state.get("asked_question_keys", [])) + new_keys))

    msg = "已完成首轮辨证，候选证候如下（按可能性排序）：\n"
    msg += _format_candidates(candidates)
    msg += f"\n当前置信度: {confidence:.2f}"
    msg += "\n\n为了更加准确的诊断，请完成下方的问卷，我会基于你的选择给你更准确的诊断信息。"

    return {
        "done": False,
        "message": msg,
        "asked_question_keys": asked,
    }


def merge_answers_node(state: TcmRoundState) -> TcmRoundState:
    """合并并标准化问卷答案，推进轮次并更新历史。"""

    raw_answers = state.get("answers", {})
    normalized_answers: dict[str, Any] = {}
    for k, v in raw_answers.items():
        # 前端可能传字符串或对象，这里统一转成标准答案结构。
        normalized_answers[str(k)] = normalize_tcm_answer(v)

    history = list(state.get("history", []))
    answers_history = list(state.get("answers_history", []))

    answer_summary = summarize_questionnaire_answers(state.get("questionnaire", []), normalized_answers)
    round_no = int(state.get("round", 0)) + 1
    if answer_summary:
        history.append(f"第{round_no}轮问卷回答：{answer_summary}")
        answers_history.append(answer_summary)

    asked_keys = list(
        dict.fromkeys(list(state.get("asked_question_keys", [])) + collect_question_keys(state.get("questionnaire", [])))
    )

    return {
        "answers": normalized_answers,
        "history": history,
        "answers_history": answers_history,
        "combined_text": "；".join(history[-16:]),
        "round": round_no,
        "asked_question_keys": asked_keys,
    }


def round_precheck_node(state: TcmRoundState) -> TcmRoundState:
    """每轮都重新做红旗筛查。"""

    red_flags = detect_tcm_red_flags(state.get("combined_text", ""))
    return {"red_flags": red_flags}


def round_extract_symptoms_node(state: TcmRoundState) -> TcmRoundState:
    """每轮都基于最新上下文再提取症状。"""

    profile = extract_tcm_symptoms(state.get("combined_text", ""))
    if not profile.get("symptoms"):
        profile = state.get("symptom_profile", {})
    return {"symptom_profile": profile}


def round_retrieve_cases_node(state: TcmRoundState) -> TcmRoundState:
    """多轮复判阶段的医案检索。"""

    combined_text = state.get("combined_text", "")
    profile = state.get("symptom_profile", {})
    symptoms = profile.get("symptoms", []) if isinstance(profile, dict) else []
    case_refs = search_tcm_cases_tool.invoke({"query": combined_text, "symptoms": symptoms, "top_k": 10})
    return {"case_refs": case_refs}


def round_infer_candidates_node(state: TcmRoundState) -> TcmRoundState:
    """多轮复判阶段的候选证候推断。"""

    candidates = infer_tcm_syndrome_candidates(
        user_text=state.get("combined_text", ""),
        symptom_profile=state.get("symptom_profile", {}),
        case_refs=state.get("case_refs", []),
    )
    return {"candidates": candidates}


def round_score_node(state: TcmRoundState) -> TcmRoundState:
    """多轮置信度评分（含问卷答案信息）。"""

    confidence = compute_tcm_confidence(
        candidates=state.get("candidates", []),
        answers=state.get("answers", {}),
        questionnaire=state.get("questionnaire", []),
    )
    return {"confidence": confidence}


def round_route(state: TcmRoundState) -> Literal["red_flag_finalize", "finalize", "continue_ask"]:
    """多轮路由决策函数。"""

    if state.get("red_flags"):
        return "red_flag_finalize"

    round_no = int(state.get("round", 1))
    confidence = float(state.get("confidence", 0.0))
    # 达到阈值或轮次上限时收敛，避免用户反复循环答题。
    if confidence >= _confidence_threshold() or round_no >= _max_rounds():
        return "finalize"
    return "continue_ask"


def continue_questionnaire_node(state: TcmRoundState) -> TcmRoundState:
    """生成下一轮问卷。"""

    round_no = int(state.get("round", 1))
    questionnaire = build_tcm_questionnaire(
        user_text=state.get("combined_text", ""),
        symptom_profile=state.get("symptom_profile", {}),
        candidates=state.get("candidates", []),
        case_refs=state.get("case_refs", []),
        asked_question_keys=state.get("asked_question_keys", []),
        target_count=_next_question_count(round_no),
    )
    stop_now = len(questionnaire) == 0
    return {"questionnaire": questionnaire, "stop_now": stop_now}


def continue_questionnaire_route(state: TcmRoundState) -> Literal["finalize", "continue_ready"]:
    """若问卷为空则直接结束，否则继续提问。"""

    return "finalize" if state.get("stop_now", False) else "continue_ready"


def continue_ready_node(state: TcmRoundState) -> TcmRoundState:
    """继续提问前，输出阶段提示信息。"""

    confidence = float(state.get("confidence", 0.0))
    round_no = int(state.get("round", 1))
    asked_keys = list(
        dict.fromkeys(list(state.get("asked_question_keys", [])) + collect_question_keys(state.get("questionnaire", [])))
    )

    msg = "已完成本轮辨证更新，当前候选证候：\n"
    msg += _format_candidates(state.get("candidates", []))
    msg += f"\n当前置信度: {confidence:.2f}"
    msg += f"\n当前轮次: {round_no}/{_max_rounds()}"
    msg += "\n\n已结合历史对话与本轮答案生成新问卷，请继续作答。"

    return {
        "done": False,
        "asked_question_keys": asked_keys,
        "message": msg,
    }


def finalize_node(state: TcmRoundState) -> TcmRoundState:
    """输出阶段性辨证结果。"""

    result = finalize_tcm_assessment(
        user_text=state.get("combined_text", ""),
        symptom_profile=state.get("symptom_profile", {}),
        candidates=state.get("candidates", []),
        questionnaire=state.get("questionnaire", []),
        answers=state.get("answers", {}),
        case_refs=state.get("case_refs", []),
        confidence=float(state.get("confidence", 0.0)),
        round_no=int(state.get("round", 1)),
        answers_history=state.get("answers_history", []),
        red_flags=state.get("red_flags", []),
    )

    refs = state.get("case_refs", [])[:5]
    ref_text = ", ".join([f"line#{x['line_no']}" for x in refs]) if refs else "无"
    confidence = float(state.get("confidence", 0.0))

    message = (
        f"阶段性最可能证候: {result.get('final_syndrome', '待进一步辨证')}\n"
        f"备选证候: {', '.join(result.get('second_choices', [])) or '无'}\n"
        f"置信度: {confidence:.2f}\n"
        f"分析: {result.get('analysis', '')}\n"
        f"建议: {result.get('advice', '')}\n"
        f"参考医案行: {ref_text}"
    )

    return {"done": True, "result": result, "message": message, "questionnaire": []}


def red_flag_finalize_node(state: TcmRoundState) -> TcmRoundState:
    """多轮阶段命中红旗时的最终安全出口。"""

    red_flags = state.get("red_flags", [])
    message = (
        "检测到可能高风险症状："
        f"{'、'.join(red_flags)}。\n"
        "建议立即线下就医或急诊，不建议继续线上辨证问卷。"
    )
    return {
        "done": True,
        "message": message,
        "result": {
            "final_syndrome": "高风险分流",
            "second_choices": [],
            "analysis": "触发红旗症状安全策略。",
            "advice": "请尽快至线下医疗机构就诊。",
        },
        "questionnaire": [],
        "confidence": 0.0,
    }


def build_tcm_collect_graph():
    """构建首轮收集图。"""

    graph = StateGraph(TcmCollectState)

    graph.add_node("merge_history", merge_history_node)
    graph.add_node("extract_symptoms", extract_symptoms_node)
    graph.add_node("precheck", precheck_node)
    graph.add_node("red_flag", red_flag_node)
    graph.add_node("extract_fail", extract_fail_node)
    graph.add_node("need_more", need_more_node)
    graph.add_node("retrieve_cases", retrieve_cases_node)
    graph.add_node("infer_candidates", infer_candidates_node)
    graph.add_node("score", collect_score_node)
    graph.add_node("build_questionnaire", questionnaire_node)
    graph.add_node("ready", ready_node)

    graph.add_edge(START, "merge_history")
    graph.add_edge("merge_history", "extract_symptoms")
    graph.add_edge("extract_symptoms", "precheck")
    graph.add_conditional_edges(
        "precheck",
        collect_route,
        {
            # 四个分支：高风险 / 提取失败 / 信息不足 / 进入辨证主链
            "red_flag": "red_flag",
            "extract_fail": "extract_fail",
            "need_more": "need_more",
            "diagnose": "retrieve_cases",
        },
    )
    graph.add_edge("red_flag", END)
    graph.add_edge("extract_fail", END)
    graph.add_edge("need_more", END)
    graph.add_edge("retrieve_cases", "infer_candidates")
    graph.add_edge("infer_candidates", "score")
    graph.add_edge("score", "build_questionnaire")
    graph.add_edge("build_questionnaire", "ready")
    graph.add_edge("ready", END)

    return graph.compile()


def build_tcm_round_graph():
    """构建多轮复判图。"""

    graph = StateGraph(TcmRoundState)

    graph.add_node("merge_answers", merge_answers_node)
    graph.add_node("precheck", round_precheck_node)
    graph.add_node("extract_symptoms", round_extract_symptoms_node)
    graph.add_node("retrieve_cases", round_retrieve_cases_node)
    graph.add_node("infer_candidates", round_infer_candidates_node)
    graph.add_node("score", round_score_node)
    graph.add_node("red_flag_finalize", red_flag_finalize_node)
    graph.add_node("continue_ask", continue_questionnaire_node)
    graph.add_node("continue_ready", continue_ready_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "merge_answers")
    graph.add_edge("merge_answers", "precheck")
    graph.add_edge("precheck", "extract_symptoms")
    graph.add_edge("extract_symptoms", "retrieve_cases")
    graph.add_edge("retrieve_cases", "infer_candidates")
    graph.add_edge("infer_candidates", "score")
    graph.add_conditional_edges(
        "score",
        round_route,
        {
            # 红旗优先，其次看置信度和轮次是否达到收敛条件
            "red_flag_finalize": "red_flag_finalize",
            "finalize": "finalize",
            "continue_ask": "continue_ask",
        },
    )
    graph.add_conditional_edges(
        "continue_ask",
        continue_questionnaire_route,
        {
            "finalize": "finalize",
            "continue_ready": "continue_ready",
        },
    )
    graph.add_edge("continue_ready", END)
    graph.add_edge("finalize", END)
    graph.add_edge("red_flag_finalize", END)

    return graph.compile()


TCM_COLLECT_APP = build_tcm_collect_graph()
TCM_ROUND_APP = build_tcm_round_graph()


def run_tcm_collect(
    history: list[str],
    user_input: str,
    asked_question_keys: list[str] | None = None,
    round_no: int = 0,
) -> TcmCollectState:
    """首轮入口（Web API 调用）。"""

    return TCM_COLLECT_APP.invoke(
        {
            "history": history,
            "user_input": user_input,
            "asked_question_keys": asked_question_keys or [],
            "round": round_no,
        }
    )


def stream_tcm_collect(
    history: list[str],
    user_input: str,
    asked_question_keys: list[str] | None = None,
    round_no: int = 0,
) -> Iterator[dict[str, Any]]:
    """首轮收集流式版本（按阶段回传进度 + 最终结果）。"""

    state: TcmCollectState = {
        "history": history,
        "user_input": user_input,
        "asked_question_keys": asked_question_keys or [],
        "round": round_no,
    }

    yield {"event": "stage", "data": {"stage": "merge", "text": "正在整理症状描述..."}}
    state.update(merge_history_node(state))

    yield {"event": "stage", "data": {"stage": "extract", "text": "正在提取关键症状..."}}
    state.update(extract_symptoms_node(state))

    yield {"event": "stage", "data": {"stage": "precheck", "text": "正在进行风险筛查..."}}
    state.update(precheck_node(state))

    route = collect_route(state)
    if route == "red_flag":
        yield {"event": "stage", "data": {"stage": "red_flag", "text": "检测到高风险症状，正在生成安全建议..."}}
        state.update(red_flag_node(state))
        yield {"event": "result", "data": state}
        return

    if route == "extract_fail":
        yield {"event": "stage", "data": {"stage": "extract_fail", "text": "症状提取失败，正在生成诊断提示..."}}
        state.update(extract_fail_node(state))
        yield {"event": "result", "data": state}
        return

    if route == "need_more":
        yield {"event": "stage", "data": {"stage": "need_more", "text": "信息不足，正在生成补充引导..."}}
        state.update(need_more_node(state))
        yield {"event": "result", "data": state}
        return

    yield {"event": "stage", "data": {"stage": "retrieve", "text": "正在检索中医医案..."}}
    state.update(retrieve_cases_node(state))

    yield {"event": "stage", "data": {"stage": "infer", "text": "正在进行辨证分析..."}}
    state.update(infer_candidates_node(state))

    yield {"event": "stage", "data": {"stage": "score", "text": "正在计算置信度..."}}
    state.update(collect_score_node(state))

    # 先返回辨证分析，避免用户等待问卷生成期间“无结果感”。
    candidates = state.get("candidates", [])
    confidence = float(state.get("confidence", 0.0))
    analysis_message = "已完成辨证分析，当前候选证候：\n"
    analysis_message += _format_candidates(candidates)
    analysis_message += f"\n当前置信度: {confidence:.2f}"
    analysis_message += "\n\n正在生成个性化问卷，请稍候..."
    yield {
        "event": "analysis",
        "data": {
            "message": analysis_message,
            "confidence": confidence,
            "candidates": candidates,
            "case_refs": state.get("case_refs", [])[:6],
            "red_flags": state.get("red_flags", []),
        },
    }

    yield {"event": "stage", "data": {"stage": "questionnaire", "text": "正在生成辨证问卷..."}}
    state.update(questionnaire_node(state))
    state.update(ready_node(state))

    yield {"event": "stage", "data": {"stage": "ready", "text": "问卷已生成，正在返回结果..."}}
    yield {"event": "result", "data": state}


def run_tcm_round(
    *,
    history: list[str],
    round_no: int,
    asked_question_keys: list[str],
    answers_history: list[str],
    symptom_profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    questionnaire: list[dict[str, Any]],
    answers: dict[str, Any],
    case_refs: list[dict[str, Any]],
) -> TcmRoundState:
    """多轮入口（提交问卷后调用）。"""

    return TCM_ROUND_APP.invoke(
        {
            "history": history,
            "round": round_no,
            "asked_question_keys": asked_question_keys,
            "answers_history": answers_history,
            "symptom_profile": symptom_profile,
            "candidates": candidates,
            "questionnaire": questionnaire,
            "answers": answers,
            "case_refs": case_refs,
        }
    )


