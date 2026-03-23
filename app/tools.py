from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KB_PATH = DATA_DIR / "knowledge_base.json"


def _load_kb() -> list[dict]:
    with KB_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


KB_CACHE = _load_kb()

DRUG_LEAFLETS = {
    "布洛芬": {
        "indication": "用于缓解轻中度疼痛、发热",
        "caution": "胃溃疡、肾功能异常或孕晚期需谨慎",
        "interaction": "与其他NSAIDs同服会增加胃肠道风险",
    },
    "阿莫西林": {
        "indication": "用于细菌感染治疗",
        "caution": "青霉素过敏者禁用",
        "interaction": "与部分抗凝药同用需医生评估",
    },
    "对乙酰氨基酚": {
        "indication": "退热镇痛",
        "caution": "避免超剂量，肝功能异常需谨慎",
        "interaction": "避免与含同成分复方药重复服用",
    },
}

DEPARTMENT_RULES = {
    "呼吸科": ["咳嗽", "发热", "咽痛", "呼吸"],
    "心内科": ["胸痛", "心悸", "胸闷"],
    "消化内科": ["腹痛", "腹泻", "反酸", "恶心"],
    "神经内科": ["头晕", "头痛", "肢体麻木", "言语不清"],
}

DOCTOR_SCHEDULE = {
    "呼吸科": "周一至周五 08:00-17:00，周六上午门诊",
    "心内科": "周一至周五 08:30-17:30",
    "消化内科": "周一至周日 08:00-20:00",
    "神经内科": "周一至周五 09:00-17:00",
    "全科": "每日 08:00-20:00",
}


def kb_search(query: str, k: int = 3) -> list[dict]:
    query_terms = set(query.lower().split())

    scored: list[tuple[int, dict]] = []
    for doc in KB_CACHE:
        score = 0
        text_fields = [doc.get("title", ""), doc.get("content", "")]
        keywords = doc.get("keywords", [])
        for field in text_fields + keywords:
            field_lower = str(field).lower()
            for term in query_terms:
                if term and term in field_lower:
                    score += 1
        for keyword in keywords:
            if keyword in query:
                score += 2
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored if score > 0][:k]


def recommend_department(text: str) -> str:
    best_department = "全科"
    best_score = 0
    for department, keywords in DEPARTMENT_RULES.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > best_score:
            best_score = score
            best_department = department
    return best_department


def get_doctor_schedule(department: str) -> str:
    return DOCTOR_SCHEDULE.get(department, DOCTOR_SCHEDULE["全科"])


def extract_drug_names(text: str) -> list[str]:
    hits = [drug for drug in DRUG_LEAFLETS if drug in text]
    return hits


def get_drug_leaflet(drug_name: str) -> dict:
    return DRUG_LEAFLETS.get(
        drug_name,
        {
            "indication": "未收录该药品，请核对商品名或通用名",
            "caution": "请咨询医生或药师确认",
            "interaction": "暂无本地数据",
        },
    )
