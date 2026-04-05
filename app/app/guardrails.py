from __future__ import annotations

import os
import re


HIGH_RISK_PATTERNS = [
    ("chest_pain_breathing", ["胸痛", "呼吸困难"]),
    ("stroke_warning", ["言语不清", "单侧无力"]),
    ("severe_allergy", ["喉头紧", "皮疹", "呼吸困难"]),
    ("heavy_bleeding", ["大出血"]),
    ("unconscious", ["昏迷"]),
]

HIGH_RISK_CONTENT_KEYWORDS = [
    "炸弹制作",
    "自制炸药",
    "做枪",
    "改枪",
    "杀人教程",
    "自杀教程",
    "无痛自杀",
    "鼓励自残",
    "未成年人色情",
    "幼女色情",
    "儿童色情",
    "种族清洗",
    "恐怖袭击教程",
    "洗钱教程",
    "伪造证件",
    "黑客入侵教程",
]

HIGH_RISK_CONTENT_PATTERNS = [
    r"(?:制作|自制).{0,6}(?:炸弹|炸药|枪|武器)",
    r"(?:如何|怎么).{0,8}(?:杀人|投毒|纵火)",
    r"(?:未成年|儿童).{0,6}(?:色情|成人视频|裸照)",
]


MEDICATION_KEYWORDS = [
    "药",
    "同服",
    "联用",
    "一起吃",
    "能一起吃",
    "能吃吗",
    "还能吃吗",
    "副作用",
    "剂量",
    "饭前",
    "饭后",
    "用药",
    "禁忌",
    "保健品",
    "健康品",
]


REPORT_KEYWORDS = ["报告", "指标", "化验", "检查结果", "化验单", "检验单", "片子", "ct", "核磁", "mri"]
SYMPTOM_KEYWORDS = ["咳嗽", "发热", "发烧", "头痛", "头晕", "腹痛", "喉咙痛", "症状", "不舒服", "不太舒服", "难受", "不适", "恶心", "拉肚子"]
APPOINTMENT_KEYWORDS = ["挂号", "预约", "门诊", "排班", "医生什么时候", "就医", "看哪个科", "挂哪个科", "挂什么科", "先看什么科", "去医院看看"]
MEDICAL_KNOWLEDGE_KEYWORDS = ["什么是", "病因", "并发症", "预后", "科普", "疾病知识", "病理", "会不会传染"]
LIFESTYLE_KEYWORDS = ["饮食", "作息", "运动", "睡眠建议", "减重", "戒烟", "控糖", "生活方式", "怎么调理"]
AFTER_SALES_KEYWORDS = ["售后", "退款", "退费", "投诉", "订单", "发票", "客服"]
HUMAN_SERVICE_KEYWORDS = ["转人工", "人工客服", "人工服务", "找人工", "接人工"]
NON_MEDICAL_KEYWORDS = ["天气", "股票", "旅游", "编程", "游戏", "电影", "音乐", "新闻"]
DAILY_CHAT_KEYWORDS = ["你好", "在吗", "你是谁", "你能做什么", "谢谢", "早上好", "晚上好", "哈喽"]

INTENT_REGEX_RULES = {
    "appointment_process": [
        r"(?:挂|看|去)(?:[\u4e00-\u9fff]{0,8})科",
        r"挂.{0,6}号",
        r"去门诊看看",
    ],
    "medication_question": [
        r"(?:能|可以|还能).{0,6}(?:吃|服用)",
    ],
}

INTENT_RULES = [
    ("human_service", HUMAN_SERVICE_KEYWORDS),
    ("after_sales", AFTER_SALES_KEYWORDS),
    ("appointment_process", APPOINTMENT_KEYWORDS),
    ("medication_question", MEDICATION_KEYWORDS),
    ("report_interpretation", REPORT_KEYWORDS),
    ("lifestyle_guidance", LIFESTYLE_KEYWORDS),
    ("medical_knowledge", MEDICAL_KNOWLEDGE_KEYWORDS),
    ("symptom_consult", SYMPTOM_KEYWORDS),
    ("non_medical", NON_MEDICAL_KEYWORDS),
    ("daily_chat", DAILY_CHAT_KEYWORDS),
]


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned.lower()


def _split_csv(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,;\n|]+", str(raw))
    return [x.strip() for x in parts if x and x.strip()]


def _content_risk_keywords() -> list[str]:
    extra = _split_csv(os.getenv("HIGH_RISK_EXTRA_KEYWORDS", ""))
    return [x.lower() for x in (HIGH_RISK_CONTENT_KEYWORDS + extra)]


def _content_risk_patterns() -> list[re.Pattern[str]]:
    patterns = list(HIGH_RISK_CONTENT_PATTERNS)
    extra = _split_csv(os.getenv("HIGH_RISK_EXTRA_PATTERNS", ""))
    patterns.extend(extra)

    compiled: list[re.Pattern[str]] = []
    for item in patterns:
        try:
            compiled.append(re.compile(item, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def detect_high_risk(text: str) -> tuple[bool, str]:
    for reason, keywords in HIGH_RISK_PATTERNS:
        if all(keyword in text for keyword in keywords):
            return True, reason

    direct_keywords = ["胸痛", "呼吸困难", "昏迷", "抽搐", "大出血"]
    if any(keyword in text for keyword in direct_keywords):
        return True, "high_risk_keyword"

    lowered = str(text or "").lower()
    compact = re.sub(r"\s+", "", lowered)
    for keyword in _content_risk_keywords():
        if keyword and (keyword in lowered or keyword in compact):
            return True, f"keyword_risk:{keyword}"

    for pattern in _content_risk_patterns():
        if pattern.search(text):
            return True, f"regex_risk:{pattern.pattern}"

    return False, "none"


def _intent_keyword_hits(text: str, keywords: list[str]) -> tuple[int, int]:
    hits = [text.find(keyword) for keyword in keywords if keyword and text.find(keyword) >= 0]
    if not hits:
        return 0, 10**6
    return len(hits), min(hits)


def _intent_regex_hits(text: str, intent: str) -> tuple[int, int]:
    patterns = INTENT_REGEX_RULES.get(intent, [])
    hits: list[int] = []
    for pattern in patterns:
        matched = re.search(pattern, text)
        if matched:
            hits.append(matched.start())
    if not hits:
        return 0, 10**6
    return len(hits), min(hits)


def classify_intent_candidates(text: str, max_items: int = 3) -> list[str]:
    matches: list[tuple[int, int, int, str]] = []
    normalized = str(text or "")
    for priority, (intent, keywords) in enumerate(INTENT_RULES):
        keyword_hits, keyword_pos = _intent_keyword_hits(normalized, list(keywords))
        regex_hits, regex_pos = _intent_regex_hits(normalized, intent)
        hit_count = keyword_hits + regex_hits
        first_pos = min(keyword_pos, regex_pos)
        if hit_count <= 0:
            continue
        matches.append((priority, -hit_count, first_pos, intent))

    if not matches:
        return ["other"]

    matches.sort()
    primary = matches[0][3]
    secondary = sorted(matches[1:], key=lambda item: (item[1], item[2], item[0]))

    ordered = [primary]
    for _, _, _, intent in secondary:
        if intent not in ordered:
            ordered.append(intent)
        if len(ordered) >= max(1, int(max_items)):
            break
    return ordered


def classify_intent(text: str) -> str:
    candidates = classify_intent_candidates(text, max_items=1)
    return candidates[0] if candidates else "other"
