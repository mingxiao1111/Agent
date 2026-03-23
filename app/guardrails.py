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
    "副作用",
    "剂量",
    "饭前",
    "饭后",
    "用药",
    "禁忌",
    "保健品",
    "健康品",
]


REPORT_KEYWORDS = ["报告", "指标", "化验", "检查结果", "ct", "核磁", "mri"]
SYMPTOM_KEYWORDS = ["咳嗽", "发热", "头痛", "腹痛", "喉咙痛", "症状", "不舒服", "难受"]
APPOINTMENT_KEYWORDS = ["挂号", "预约", "门诊", "排班", "医生什么时候", "就医", "看哪个科"]
MEDICAL_KNOWLEDGE_KEYWORDS = ["什么是", "病因", "并发症", "预后", "科普", "疾病知识", "病理"]
LIFESTYLE_KEYWORDS = ["饮食", "作息", "运动", "睡眠建议", "减重", "戒烟", "控糖", "生活方式"]
AFTER_SALES_KEYWORDS = ["售后", "退款", "退费", "投诉", "订单", "发票", "客服"]
HUMAN_SERVICE_KEYWORDS = ["转人工", "人工客服", "人工服务", "找人工", "接人工"]
NON_MEDICAL_KEYWORDS = ["天气", "股票", "旅游", "编程", "游戏", "电影", "音乐", "新闻"]
DAILY_CHAT_KEYWORDS = ["你好", "在吗", "你是谁", "你能做什么", "谢谢", "早上好", "晚上好", "哈喽"]


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


def classify_intent(text: str) -> str:
    if any(k in text for k in HUMAN_SERVICE_KEYWORDS):
        return "human_service"
    if any(k in text for k in AFTER_SALES_KEYWORDS):
        return "after_sales"
    if any(k in text for k in APPOINTMENT_KEYWORDS):
        return "appointment_process"
    if any(k in text for k in MEDICATION_KEYWORDS):
        return "medication_question"
    if any(k in text for k in REPORT_KEYWORDS):
        return "report_interpretation"
    if any(k in text for k in LIFESTYLE_KEYWORDS):
        return "lifestyle_guidance"
    if any(k in text for k in MEDICAL_KNOWLEDGE_KEYWORDS):
        return "medical_knowledge"
    if any(k in text for k in SYMPTOM_KEYWORDS):
        return "symptom_consult"
    if any(k in text for k in NON_MEDICAL_KEYWORDS):
        return "non_medical"
    if any(k in text for k in DAILY_CHAT_KEYWORDS):
        return "daily_chat"
    return "other"
