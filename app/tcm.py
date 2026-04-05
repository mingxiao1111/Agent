from __future__ import annotations

"""中医辨证核心能力层（提取、检索、候选推断、问卷生成、结果总结）。

定位说明：
- `tcm_graph.py` 负责编排流程（谁先执行、何时分支）；
- 本文件负责“具体能力实现”（每一步怎么做）。

整体顺序：
1) 抽取并标准化症状；
2) 用本地医案做混合检索（关键词 + 向量）；
3) 生成候选证候；
4) 生成区分问卷并汇总答案；
5) 计算置信度并输出阶段性结论。
"""

import json
import os
import re
import time
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Milvus, SKLearnVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from .chat_backend import build_chat_model
from .llm_trace import log_error, log_prompt, log_response
from .model_compat import supports_volcengine_reasoning, volcengine_reasoning_kwargs
from .online_search import fetch_tavily_context

CASE_FILE = "medical_cases_cleaned.txt"
CASE_MARKDOWN_DIR = "证候"
CASE_DOCX_DIR = "医案"
YES_NO_UNKNOWN = ("是", "否", "不确定")
QUESTION_ANSWER_OPTIONS = ("是", "部分是", "否", "不确定")
TCM_VECTOR_STORE_FILE = "tcm_cases_sklearn_store.json"
TCM_VECTOR_META_FILE = "tcm_cases_sklearn_store.meta.json"
TCM_MILVUS_CASE_COLLECTION = "tcm_cases"
PATENT_DATA_DIR = "中成药"
PATENT_SECTIONS_FILE = "patent_sections_candidates.jsonl"
PATENT_VECTOR_STORE_FILE = "tcm_patent_sklearn_store.json"
PATENT_VECTOR_META_FILE = "tcm_patent_sklearn_store.meta.json"
TCM_MILVUS_PATENT_COLLECTION = "tcm_patent"

# 口语 -> 规范术语映射，用于“提取后标准化”与“问卷关键词规范化”。
SYMPTOM_ALIAS_MAP = {
    "睡不着": "失眠",
    "拉肚子": "腹泻",
    "胃口差": "食欲差",
    "不想吃饭": "食欲差",
    "嘴干": "口干",
    "怕冷": "畏寒",
    "全身没劲": "乏力",
    "没有力气": "乏力",
    "心里慌": "心悸",
    "胸口闷": "胸闷",
    "喘不上气": "呼吸困难",
}

TCM_RED_FLAG_KEYWORDS = [
    "胸部剧痛",
    "双腿浮肿",
    "呼吸困难",
    "喘不上气",
    "晕厥",
    "抽搐",
    "咯血",
    "呕血",
    "便血",
    "高烧",
    "持续高热",
    "心脏剧痛",
    "剧烈头痛",
    "意识不清",
    "肢体麻木无力",
]

# 生成“部分是”关键词时需要过滤掉的泛词，避免给出无意义选项。
QUESTION_KEYWORD_STOPWORDS = {
    "是否",
    "是不是",
    "有无",
    "有没有",
    "症状",
    "情况",
    "表现",
    "明显",
    "持续",
    "加重",
    "更明显",
    "偏多",
    "偏少",
    "偏稀",
    "偏干",
    "若观察到",
    "若有",
}


class OpenAICompatibleEmbeddings(Embeddings):
    """OpenAI-compatible embeddings client (for SiliconFlow style endpoints)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout_sec: int = 60,
        max_retries: int = 3,
        batch_size: int = 48,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "https://api.siliconflow.cn/v1").strip().rstrip("/")
        self.timeout_sec = max(5, int(timeout_sec))
        self.max_retries = max(1, int(max_retries))
        self.batch_size = max(1, int(batch_size))

    @property
    def _endpoint(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"

    def _request_embeddings(self, batch: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": batch,
            "encoding_format": "float",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=self._endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(raw)
                rows = data.get("data", [])
                vectors: list[list[float]] = []
                for item in rows:
                    vec = item.get("embedding", [])
                    if not isinstance(vec, list) or not vec:
                        continue
                    vectors.append([float(x) for x in vec])
                if len(vectors) != len(batch):
                    raise ValueError(f"embedding_count_mismatch expected={len(batch)} got={len(vectors)}")
                return vectors
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.5, 0.4 * attempt))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("embedding_request_failed")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        docs = [str(x or "") for x in texts]
        if not docs:
            return []
        vectors: list[list[float]] = []
        step = self.batch_size
        for i in range(0, len(docs), step):
            batch = docs[i : i + step]
            vectors.extend(self._request_embeddings(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        batch = self._request_embeddings([str(text or "")])
        return batch[0] if batch else []


def _extract_text(raw_output: Any) -> str:
    """统一提取模型输出文本（兼容 str / list / message 对象）。"""

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


def _extract_json(text: str) -> dict[str, Any]:
    """从文本中截取并解析 JSON 对象。"""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("json_not_found")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("json_dict_expected")
    return parsed


def _extract_json_array(text: str) -> list[Any]:
    """从文本中截取并解析 JSON 数组。"""

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("json_array_not_found")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("json_array_expected")
    return parsed


def _normalize_str_list(items: list[Any], limit: int = 12) -> list[str]:
    """字符串列表清洗：去空、别名规范化、去重、限长。"""

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        text = SYMPTOM_ALIAS_MAP.get(text, text)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _ensure_list(value: Any) -> list[Any]:
    """把任意值安全包装成 list，方便后续统一处理。"""

    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _extract_question_symptom_keywords(question: str, limit: int = 6) -> list[str]:
    """从问卷题干中抽取可勾选的症状关键词。

    目标是给“部分是”场景做细粒度补充，避免一题多症状时信息丢失。
    """

    text = str(question or "").strip()
    if not text:
        return []

    text = re.sub(r"[（(].*?[)）]", "", text)
    chunks = re.split(r"[，、,；;。？！?]|或|和|及|并且|并|且|与|/", text)
    out: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        item = str(chunk).strip()
        if not item:
            continue
        item = re.sub(
            r"^(是否|是不是|有无|有没有|会不会|是否有|是否出现|是否伴有|是否明显|是否持续|是否容易|是否经常|请问|会否|可有|有否)+",
            "",
            item,
        )
        item = re.sub(r"(吗|呢|情况|表现|症状)+$", "", item).strip()
        if not item or len(item) < 2 or item in QUESTION_KEYWORD_STOPWORDS:
            continue
        item = SYMPTOM_ALIAS_MAP.get(item, item)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            return out

    if out:
        return out[:limit]

    coarse = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    for token in coarse:
        token = str(token).strip()
        if not token or token in QUESTION_KEYWORD_STOPWORDS:
            continue
        token = re.sub(r"^(是否|是不是|有无|有没有)+", "", token).strip()
        token = re.sub(r"(情况|表现|症状)+$", "", token).strip()
        if not token or len(token) < 2:
            continue
        token = SYMPTOM_ALIAS_MAP.get(token, token)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _normalize_partial_keywords(items: list[Any], question: str, limit: int = 6) -> list[str]:
    """优先使用 LLM 提供的关键词；缺失时从题干自动提取。"""

    normalized = _normalize_str_list(items, limit=limit)
    if normalized:
        return normalized[:limit]
    return _extract_question_symptom_keywords(question, limit=limit)


def normalize_tcm_answer(answer: Any) -> dict[str, Any]:
    """统一问卷答案结构。

    输出固定字段：
    - value: 是/部分是/否/不确定
    - selected_keywords: 仅在“部分是”时保留
    - other_text: 仅在“部分是”时保留
    """

    value = ""
    selected_keywords: list[str] = []
    other_text = ""

    if isinstance(answer, dict):
        value = str(answer.get("value", "")).strip()
        selected_keywords = _normalize_str_list(
            _ensure_list(answer.get("selected_keywords", [])) + _ensure_list(answer.get("partial_keywords", [])),
            limit=8,
        )
        other_text = re.sub(r"\s+", " ", str(answer.get("other_text", "")).strip())[:100]
    else:
        value = str(answer).strip()

    if value not in QUESTION_ANSWER_OPTIONS:
        value = "不确定"

    if value != "部分是":
        selected_keywords = []
        other_text = ""

    return {
        "value": value,
        "selected_keywords": selected_keywords,
        "other_text": other_text,
    }


def build_question_key(question: str) -> str:
    """把题干归一化为“去标点短 key”，用于去重。"""

    text = re.sub(r"[，。！？、,.?!\s]+", "", str(question or ""))
    return text[:32]


def collect_question_keys(questionnaire: list[dict[str, Any]]) -> list[str]:
    """收集问卷 key 列表，用于跨轮避免重复提问。"""

    keys: list[str] = []
    seen: set[str] = set()
    for item in questionnaire:
        key = build_question_key(item.get("question", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _normalize_questionnaire(items: list[Any], asked_question_keys: list[str] | None = None) -> list[dict[str, Any]]:
    """把 LLM 返回的问卷做结构化清洗与去重。"""

    asked = set(asked_question_keys or [])
    seen_keys: set[str] = set()
    questions: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            question = str(item.get("question", "")).strip()
            qid = str(item.get("id", f"q{idx}"))
            purpose = str(item.get("purpose", "")).strip()
            discriminates = _normalize_str_list(_ensure_list(item.get("discriminates", [])), limit=3)
            expected_gain = str(item.get("expected_gain", "")).strip()
            partial_keywords = _normalize_partial_keywords(
                _ensure_list(item.get("partial_keywords", [])) + _ensure_list(item.get("symptom_keywords", [])),
                question,
                limit=6,
            )
        else:
            question = str(item).strip()
            qid = f"q{idx}"
            purpose = ""
            discriminates = []
            expected_gain = ""
            partial_keywords = _normalize_partial_keywords([], question, limit=6)

        if not question:
            continue

        key = build_question_key(question)
        if not key or key in asked or key in seen_keys:
            continue
        seen_keys.add(key)

        questions.append(
            {
                "id": qid,
                "question": question,
                "purpose": purpose,
                "discriminates": discriminates,
                "expected_gain": expected_gain,
                "partial_keywords": partial_keywords,
                "options": list(QUESTION_ANSWER_OPTIONS),
            }
        )

        if len(questions) >= 12:
            break

    return questions


def _tokenize(text: str) -> list[str]:
    """简单分词：英文/数字 + 中文 2-gram，供关键词检索打分使用。"""

    chunks = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}", text.lower())
    tokens: list[str] = []
    for chunk in chunks:
        tokens.append(chunk)
        if re.match(r"^[\u4e00-\u9fff]+$", chunk) and len(chunk) > 2:
            for i in range(len(chunk) - 1):
                tokens.append(chunk[i : i + 2])
    return tokens


@lru_cache(maxsize=1)
def _tcm_llm_provider() -> str:
    """TCM 链路 LLM 提供方：默认 volcengine。"""

    raw = str(os.getenv("TCM_LLM_PROVIDER", "volcengine")).strip().lower()
    if raw in {"volcengine", "default"}:
        return raw
    return "volcengine"


def _tcm_bool_env(name: str, default: bool) -> bool:
    value = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _tcm_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return int(default)


def _tcm_volcengine_api_key() -> str:
    return str(os.getenv("TCM_VOLCENGINE_API_KEY") or os.getenv("VOLCENGINE_API_KEY") or "").strip()


def _tcm_volcengine_base_url() -> str:
    return str(
        os.getenv("TCM_VOLCENGINE_BASE_URL")
        or os.getenv("VOLCENGINE_BASE_URL")
        or "https://ark.cn-beijing.volces.com/api/v3"
    ).strip()


def _tcm_main_model_name() -> str:
    return str(os.getenv("TCM_VOLCENGINE_MAIN_MODEL", "deepseek-v3-2-251201")).strip() or "deepseek-v3-2-251201"


def _tcm_extract_model_name() -> str:
    return str(os.getenv("TCM_VOLCENGINE_EXTRACT_MODEL", "doubao-seed-2-0-mini-260215")).strip() or "doubao-seed-2-0-mini-260215"


def _tcm_thinking_enabled() -> bool:
    return _tcm_bool_env("TCM_VOLCENGINE_THINKING", False)


def _tcm_chat_enabled() -> bool:
    provider = _tcm_llm_provider()
    if provider == "volcengine":
        return bool(_tcm_volcengine_api_key())
    return bool(os.getenv("DASHSCOPE_API_KEY"))


def _build_tcm_chat_model(*, model_name: str, temperature: float) -> Any:
    """构建 TCM 专用聊天模型。

    默认走火山引擎；如果显式改为 default，则走现有聊天后端。
    """

    if _tcm_llm_provider() == "volcengine":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {}
        if supports_volcengine_reasoning(model_name):
            kwargs.update(
                volcengine_reasoning_kwargs(
                    model_name=model_name,
                    thinking=_tcm_thinking_enabled(),
                )
            )

        return ChatOpenAI(
            model=str(model_name or "").strip(),
            temperature=float(temperature),
            streaming=False,
            api_key=_tcm_volcengine_api_key(),
            base_url=_tcm_volcengine_base_url(),
            timeout=float(_tcm_int_env("TCM_VOLCENGINE_TIMEOUT_SEC", _tcm_int_env("VOLCENGINE_TIMEOUT_SEC", 60))),
            max_retries=int(
                max(
                    0,
                    _tcm_int_env("TCM_VOLCENGINE_MAX_RETRIES", _tcm_int_env("VOLCENGINE_MAX_RETRIES", 2)),
                )
            ),
            **kwargs,
        )

    # fallback：沿用现有后端（dashscope/openai_compatible）。
    return build_chat_model(model_name=str(model_name or "").strip(), temperature=float(temperature), streaming=False)


@lru_cache(maxsize=1)
def _llm() -> Any:
    """主推理模型（用于候选证候、问卷、总结等）。"""

    if _tcm_llm_provider() == "volcengine":
        model = _tcm_main_model_name()
        temperature = float(os.getenv("TCM_MAIN_TEMPERATURE", "0.25"))
        return _build_tcm_chat_model(model_name=model, temperature=temperature)

    model = os.getenv("TONGYI_MODEL") or "deepseek-v3-2-251201"
    temperature = float(os.getenv("TONGYI_TEMPERATURE", "0.7"))
    return _build_tcm_chat_model(model_name=model, temperature=temperature)


@lru_cache(maxsize=1)
def _extractor_llm() -> Any:
    """低温提取模型（用于症状抽取与 JSON 修复）。"""

    if _tcm_llm_provider() == "volcengine":
        model = _tcm_extract_model_name()
        temperature = float(os.getenv("TCM_EXTRACTOR_TEMPERATURE", "0.0"))
        return _build_tcm_chat_model(model_name=model, temperature=temperature)

    model = (
        os.getenv("TONGYI_EXTRACTOR_MODEL")
        or os.getenv("TONGYI_ROUTER_MODEL")
        or os.getenv("TONGYI_FOLLOWUP_MODEL")
        or os.getenv("TONGYI_MODEL")
        or "doubao-seed-2-0-mini-260215"
    )
    temperature = float(os.getenv("TONGYI_EXTRACTOR_TEMPERATURE", "0.1"))
    return _build_tcm_chat_model(model_name=model, temperature=temperature)


@lru_cache(maxsize=1)
def _case_records() -> list[dict[str, Any]]:
    """加载并清洗中医语料（txt + md + docx），统一为检索记录。"""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    base_path = _case_file_path()
    if base_path.exists():
        try:
            with base_path.open("r", encoding="utf-8-sig") as f:
                for raw in f:
                    text = _truncate_txt_before_prescription(str(raw))
                    if len(text) < 16 or text in seen:
                        continue
                    seen.add(text)
                    records.append(
                        {
                            "text": text,
                            "source": "txt",
                            "file": base_path.name,
                            "section_line": 0,
                            "paragraph_index": 0,
                            "chunk_index": 0,
                        }
                    )
        except Exception as exc:
            log_error("tcm._case_records.base_txt", exc)

    for item in _markdown_case_records():
        text = str(item.get("text", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        records.append(item)

    for item in _docx_case_records():
        text = str(item.get("text", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        records.append(item)

    return records


@lru_cache(maxsize=1)
def _cases() -> list[str]:
    """返回检索文本列表（兼容旧调用）。"""

    return [str(item.get("text", "")) for item in _case_records()]


def _case_file_path() -> Path:
    """医案数据文件路径。"""

    return Path(__file__).resolve().parents[1] / "data" / CASE_FILE


def _case_markdown_dir_path() -> Path:
    """证候 markdown 数据目录。"""

    return Path(__file__).resolve().parents[1] / "data" / CASE_MARKDOWN_DIR


def _case_docx_dir_path() -> Path:
    """医案 Word 数据目录。"""

    return Path(__file__).resolve().parents[1] / "data" / CASE_DOCX_DIR


def _safe_read_text(path: Path) -> str:
    """读取文本文件，自动尝试常见编码。"""

    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _truncate_txt_before_prescription(text: str) -> str:
    """按行文本只保留到“处方/方药/方剂”前半段，减少噪声上下文。"""

    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if not clean:
        return ""

    markers = ["处方", "中药", "方药", "方剂", "用药", "药物组成"]
    cut_points: list[int] = []
    for marker in markers:
        idx = clean.find(marker)
        if idx >= 0:
            cut_points.append(idx)

    if cut_points:
        cut = min(cut_points)
        if cut > 0:
            clean = clean[: cut + 2]

    return clean.strip(" ，,；;。:：")


def _split_case_text(
    text: str,
    max_chars: int = 900,
    target_chars: int = 760,
    min_chars: int = 420,
    overlap_chars: int = 180,
) -> list[str]:
    """按句切分并聚合成较大 chunk，支持滑动重叠。"""

    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if not clean:
        return []
    max_chars = max(180, int(max_chars))
    target_chars = max(120, min(int(target_chars), max_chars))
    min_chars = max(80, min(int(min_chars), target_chars))
    overlap_chars = max(0, min(int(overlap_chars), max_chars - 40))

    def _hard_cut(seg: str) -> list[str]:
        """长句兜底切分：优先标点，找不到再按长度硬切。"""

        remain = str(seg or "").strip()
        out: list[str] = []
        while len(remain) > max_chars:
            cut = max(
                remain.rfind("。", 0, max_chars),
                remain.rfind("；", 0, max_chars),
                remain.rfind("，", 0, max_chars),
                remain.rfind("、", 0, max_chars),
                remain.rfind("：", 0, max_chars),
            )
            if cut < 24:
                cut = max_chars
            part = remain[:cut].strip(" ，,；;、:：")
            if part:
                out.append(part)
            remain = remain[cut:].strip(" ，,；;、:：")
        if remain:
            out.append(remain)
        return out

    # 先把编号条目打断，再按句号/分号/冒号切，再对超长句二次切。
    numbered = re.sub(
        r"(?<!^)(?=(?:[一二三四五六七八九十]{1,3}[、.．]|[0-9]{1,2}[、.．]))",
        "\n",
        clean,
    )
    rough_units = re.split(r"(?<=[。！？；;!?：:])|\n", numbered)
    units: list[str] = []
    for piece in rough_units:
        seg = str(piece or "").strip(" ，,；;、:：")
        if not seg:
            continue
        if len(seg) <= max_chars:
            units.append(seg)
            continue
        for p in re.split(r"(?<=[，,、])", seg):
            p = str(p or "").strip(" ，,；;、:：")
            if not p:
                continue
            if len(p) <= max_chars:
                units.append(p)
            else:
                units.extend(_hard_cut(p))

    if not units:
        return []

    chunks: list[str] = []
    start = 0
    total = len(units)

    while start < total:
        buf = ""
        end = start

        while end < total:
            unit = units[end]
            if not buf:
                buf = unit
                end += 1
                continue

            merged_len = len(buf) + len(unit)
            if merged_len <= target_chars:
                buf = f"{buf}{unit}"
                end += 1
                continue
            if len(buf) < min_chars and merged_len <= max_chars:
                buf = f"{buf}{unit}"
                end += 1
                continue
            break

        if not buf:
            break

        if end >= total:
            if chunks and len(buf) < min_chars and len(chunks[-1]) + len(buf) <= max_chars:
                chunks[-1] = f"{chunks[-1]}{buf}"
            else:
                chunks.append(buf)
            break

        chunks.append(buf)

        if overlap_chars <= 0:
            start = end
            continue

        overlap_start = end
        covered = 0
        while overlap_start > start and covered < overlap_chars:
            overlap_start -= 1
            covered += len(units[overlap_start])

        # 防止 overlap 过大导致起点不前进。
        if overlap_start <= start:
            start = end
        else:
            start = overlap_start

    return chunks


@lru_cache(maxsize=1)
def _case_chunk_config() -> dict[str, int]:
    """医案切分参数（支持环境变量）。"""

    max_chars = _to_int_env("TCM_CASE_CHUNK_MAX_CHARS", 900)
    target_chars = _to_int_env("TCM_CASE_CHUNK_TARGET_CHARS", 760)
    min_chars = _to_int_env("TCM_CASE_CHUNK_MIN_CHARS", 420)
    overlap_chars = _to_int_env("TCM_CASE_CHUNK_OVERLAP_CHARS", 180)

    max_chars = max(180, min(max_chars, 2600))
    target_chars = max(120, min(target_chars, max_chars))
    min_chars = max(80, min(min_chars, target_chars))
    overlap_chars = max(0, min(overlap_chars, max_chars - 40))

    return {
        "max_chars": int(max_chars),
        "target_chars": int(target_chars),
        "min_chars": int(min_chars),
        "overlap_chars": int(overlap_chars),
    }


def _looks_like_case_chunk(text: str) -> bool:
    """过滤明显噪声段落，保留更像医案/证候描述的文本。"""

    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if len(clean) < 24:
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", clean))
    if cjk_count < 12:
        return False

    noise_words = ["目录", "ISBN", "版权所有", "印张", "定价", "出版社", "责任编辑"]
    if any(word in clean for word in noise_words) and len(clean) < 120:
        return False

    tcm_terms = ["症", "证", "舌", "脉", "方", "治", "寒", "热", "虚", "实", "气", "血", "痰", "湿"]
    if not any(term in clean for term in tcm_terms) and len(clean) < 90:
        return False
    return True


def _markdown_paragraphs(md_text: str) -> list[dict[str, Any]]:
    """把 markdown 按段切分（空行分段），而不是按标题 section。"""

    lines = md_text.splitlines()
    paragraphs: list[dict[str, Any]] = []

    current_heading = ""
    buf: list[str] = []
    start_line = 1

    def flush() -> None:
        nonlocal buf, start_line
        if not buf:
            return
        body = re.sub(r"\s+", " ", " ".join(buf)).strip()
        buf = []
        if not body:
            return
        text = f"{current_heading}。{body}" if current_heading else body
        paragraphs.append({"line_no": start_line, "text": text})

    for idx, raw in enumerate(lines, start=1):
        line = str(raw or "").strip()
        if not line:
            flush()
            continue
        if re.match(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$", line):
            continue
        if re.match(r"^\s*```", line):
            flush()
            continue

        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if heading:
            flush()
            current_heading = re.sub(r"\s+", " ", str(heading.group(1) or "").strip())
            continue

        plain = re.sub(r"^\s*[-*+]\s+", "", line)
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        if not plain:
            continue
        if not buf:
            start_line = idx
        buf.append(plain)

    flush()
    return paragraphs


def _markdown_case_records() -> list[dict[str, Any]]:
    """加载 data/证候 下 markdown 语料（按段切分）。"""

    root = _case_markdown_dir_path()
    if not root.exists():
        return []

    out: list[dict[str, Any]] = []
    chunk_cfg = _case_chunk_config()
    max_per_file = int(max(200, _to_float_env("TCM_MD_CASE_MAX_PER_FILE", 1400)))
    for path in sorted(root.glob("*.md")):
        raw = _safe_read_text(path)
        if not raw.strip():
            continue

        paragraphs = _markdown_paragraphs(raw)
        if not paragraphs:
            paragraphs = [{"line_no": 1, "text": raw}]

        kept = 0
        for para_idx0 in _middle_out_indices(len(paragraphs)):
            para = paragraphs[para_idx0]
            para_idx = int(para_idx0) + 1
            body = str(para.get("text", "")).strip()
            line_no = int(para.get("line_no", 0) or 0)
            for chunk_idx, chunk in enumerate(_split_case_text(body, **chunk_cfg), start=1):
                if not _looks_like_case_chunk(chunk):
                    continue
                out.append(
                    {
                        "text": chunk,
                        "source": "md",
                        "file": path.name,
                        "section_line": line_no,
                        "paragraph_index": para_idx,
                        "chunk_index": chunk_idx,
                    }
                )
                kept += 1
                if kept >= max_per_file:
                    break
            if kept >= max_per_file:
                break
    return out


def _read_docx_paragraphs(path: Path) -> list[str]:
    """从 docx 抽取段落文本（无需额外依赖）。"""

    try:
        with zipfile.ZipFile(path, "r") as zf:
            xml_bytes = zf.read("word/document.xml")
    except Exception:
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return []

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        parts: list[str] = []
        for node in para.findall(".//w:t", ns):
            value = str(node.text or "").strip()
            if value:
                parts.append(value)
        if not parts:
            continue
        text = re.sub(r"\s+", " ", "".join(parts)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _middle_out_indices(total: int) -> list[int]:
    """中轴优先，再向两侧扩散的索引顺序（适合书籍语料覆盖）。"""

    if total <= 0:
        return []
    center = (total - 1) // 2
    order = [center]
    step = 1
    while len(order) < total:
        right = center + step
        left = center - step
        if right < total:
            order.append(right)
        if len(order) >= total:
            break
        if left >= 0:
            order.append(left)
        step += 1
    return order


def _docx_case_records() -> list[dict[str, Any]]:
    """加载 data/医案 下 docx 语料。"""

    root = _case_docx_dir_path()
    if not root.exists():
        return []

    out: list[dict[str, Any]] = []
    chunk_cfg = _case_chunk_config()
    max_per_file = int(max(150, _to_float_env("TCM_DOCX_CASE_MAX_PER_FILE", 700)))
    for path in sorted(root.glob("*.docx")):
        paragraphs = _read_docx_paragraphs(path)
        if not paragraphs:
            continue

        kept = 0
        for paragraph_idx0 in _middle_out_indices(len(paragraphs)):
            para = paragraphs[paragraph_idx0]
            paragraph_index = int(paragraph_idx0) + 1
            for chunk_idx, chunk in enumerate(_split_case_text(para, **chunk_cfg), start=1):
                if not _looks_like_case_chunk(chunk):
                    continue
                out.append(
                    {
                        "text": chunk,
                        "source": "docx",
                        "file": path.name,
                        "section_line": paragraph_index,
                        "paragraph_index": paragraph_index,
                        "chunk_index": chunk_idx,
                    }
                )
                kept += 1
                if kept >= max_per_file:
                    break
            if kept >= max_per_file:
                break
    return out


def _vector_store_path() -> Path:
    """向量库持久化文件路径。"""

    return Path(__file__).resolve().parents[1] / "data" / TCM_VECTOR_STORE_FILE


def _vector_meta_path() -> Path:
    """向量库元信息文件路径。"""

    return Path(__file__).resolve().parents[1] / "data" / TCM_VECTOR_META_FILE


def _patent_data_dir_path() -> Path:
    """中成药数据目录。"""

    return Path(__file__).resolve().parents[1] / "data" / PATENT_DATA_DIR


def _patent_sections_file_path() -> Path:
    """中成药成品分段文件（jsonl，一行一个中成药）。"""

    return _patent_data_dir_path() / PATENT_SECTIONS_FILE


def _patent_vector_store_path() -> Path:
    """中成药向量库持久化文件路径。"""

    return Path(__file__).resolve().parents[1] / "data" / PATENT_VECTOR_STORE_FILE


def _patent_vector_meta_path() -> Path:
    """中成药向量库元信息文件路径。"""

    return Path(__file__).resolve().parents[1] / "data" / PATENT_VECTOR_META_FILE


def _is_vector_enabled() -> bool:
    """是否启用向量检索（可通过环境变量关闭）。"""

    val = str(os.getenv("TCM_ENABLE_VECTOR_RETRIEVAL", "true")).strip().lower()
    return val in {"1", "true", "yes", "on"}


def _embedding_provider() -> str:
    """Embedding 提供方：dashscope / siliconflow。"""

    raw = str(os.getenv("TCM_EMBEDDING_PROVIDER", "dashscope")).strip().lower()
    return raw if raw in {"dashscope", "siliconflow"} else "dashscope"


def _embedding_model_name() -> str:
    """返回当前 provider 对应的 embedding 模型名。"""

    provider = _embedding_provider()
    if provider == "siliconflow":
        model = str(os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")).strip()
        return model or "BAAI/bge-m3"
    model = str(os.getenv("TONGYI_EMBEDDING_MODEL", "text-embedding-v1")).strip()
    return model or "text-embedding-v1"


def _to_bool_env(name: str, default: bool) -> bool:
    """读取 bool 类型环境变量。"""

    val = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return val in {"1", "true", "yes", "on"}


def _vector_backend() -> str:
    """向量后端：milvus / sklearn。"""

    raw = str(os.getenv("TCM_VECTOR_BACKEND", "milvus")).strip().lower()
    return raw if raw in {"milvus", "sklearn"} else "milvus"


def _milvus_connection_args() -> dict[str, Any]:
    """Milvus 连接参数。"""

    uri = str(os.getenv("MILVUS_URI", "http://127.0.0.1:19530")).strip() or "http://127.0.0.1:19530"
    args: dict[str, Any] = {"uri": uri}

    token = str(os.getenv("MILVUS_TOKEN", "")).strip()
    if token:
        args["token"] = token
        return args

    user = str(os.getenv("MILVUS_USER", "")).strip()
    password = str(os.getenv("MILVUS_PASSWORD", "")).strip()
    if user:
        args["user"] = user
    if password:
        args["password"] = password
    return args


def _milvus_case_collection_name() -> str:
    """医案 Milvus 集合名。"""

    return str(os.getenv("TCM_MILVUS_CASE_COLLECTION", TCM_MILVUS_CASE_COLLECTION)).strip() or TCM_MILVUS_CASE_COLLECTION


def _milvus_patent_collection_name() -> str:
    """中成药 Milvus 集合名。"""

    return str(os.getenv("TCM_MILVUS_PATENT_COLLECTION", TCM_MILVUS_PATENT_COLLECTION)).strip() or TCM_MILVUS_PATENT_COLLECTION


def _milvus_use_hnsw() -> bool:
    """Milvus 索引类型开关：默认 HNSW。"""

    return _to_bool_env("TCM_MILVUS_USE_HNSW", True)


def _milvus_index_params() -> dict[str, Any]:
    """Milvus 索引参数。"""

    if _milvus_use_hnsw():
        m = int(max(8, min(64, _to_int_env("TCM_MILVUS_HNSW_M", 16))))
        ef_construction = int(max(40, min(800, _to_int_env("TCM_MILVUS_HNSW_EF_CONSTRUCTION", 200))))
        return {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": m, "efConstruction": ef_construction},
        }

    nlist = int(max(64, min(8192, _to_int_env("TCM_MILVUS_IVF_NLIST", 1024))))
    return {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": nlist},
    }


def _milvus_search_params() -> dict[str, Any]:
    """Milvus 检索参数。"""

    if _milvus_use_hnsw():
        ef = int(max(16, min(1024, _to_int_env("TCM_MILVUS_HNSW_EF_SEARCH", 96))))
        return {"metric_type": "COSINE", "params": {"ef": ef}}

    nprobe = int(max(4, min(256, _to_int_env("TCM_MILVUS_IVF_NPROBE", 32))))
    return {"metric_type": "COSINE", "params": {"nprobe": nprobe}}


def _milvus_index_signature() -> dict[str, Any]:
    """把 Milvus 索引配置纳入签名，参数变化时触发重建。"""

    return {
        "use_hnsw": _milvus_use_hnsw(),
        "index": _milvus_index_params(),
        "search": _milvus_search_params(),
    }


def _vector_build_limit(total: int) -> int:
    """控制构建向量库的样本上限，避免冷启动过慢。"""

    raw = str(os.getenv("TCM_VECTOR_BUILD_LIMIT", "4000")).strip()
    try:
        limit = int(raw)
    except ValueError:
        limit = 4000
    return max(200, min(total, limit))


def _to_float_env(name: str, default: float) -> float:
    """读取 float 类型环境变量，失败时使用默认值。"""

    try:
        return float(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return default


def _to_int_env(name: str, default: int) -> int:
    """读取 int 类型环境变量，失败时使用默认值。"""

    try:
        return int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return int(default)


def _source_weight(source: str) -> float:
    """按语料来源返回检索权重，支持环境变量调参。"""

    src = str(source or "").strip().lower()
    if src == "txt":
        weight = _to_float_env("TCM_SOURCE_WEIGHT_TXT", 1.0)
    elif src == "md":
        weight = _to_float_env("TCM_SOURCE_WEIGHT_MD", 0.9)
    elif src == "docx":
        weight = _to_float_env("TCM_SOURCE_WEIGHT_DOCX", 0.75)
    else:
        weight = _to_float_env("TCM_SOURCE_WEIGHT_OTHER", 0.85)
    return max(0.2, min(2.0, float(weight)))


def _normalize_source(source: str) -> str:
    """统一来源标签，避免大小写/空值导致分组混乱。"""

    text = str(source or "").strip().lower()
    return text if text in {"txt", "md", "docx"} else "other"


def _apply_source_quota(
    items: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """按来源做“保底+上限”重排，提升混合语料稳定性。"""

    if not items or top_k <= 0:
        return []

    ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    groups: dict[str, list[dict[str, Any]]] = {"txt": [], "md": [], "docx": [], "other": []}
    for item in ranked:
        src = _normalize_source(str(item.get("source", "")))
        groups.setdefault(src, []).append(item)

    present_sources = [s for s in ("txt", "md", "docx", "other") if groups.get(s)]
    if not present_sources:
        return ranked[:top_k]

    default_min = {
        "txt": max(1, int(round(top_k * 0.4))),
        "md": int(round(top_k * 0.25)),
        "docx": int(round(top_k * 0.2)),
        "other": 0,
    }
    default_max = {
        "txt": top_k,
        "md": max(1, int(round(top_k * 0.6))),
        "docx": max(1, int(round(top_k * 0.45))),
        "other": max(1, int(round(top_k * 0.35))),
    }
    env_min = {
        "txt": _to_int_env("TCM_SOURCE_MIN_TXT", default_min["txt"]),
        "md": _to_int_env("TCM_SOURCE_MIN_MD", default_min["md"]),
        "docx": _to_int_env("TCM_SOURCE_MIN_DOCX", default_min["docx"]),
        "other": _to_int_env("TCM_SOURCE_MIN_OTHER", default_min["other"]),
    }
    env_max = {
        "txt": _to_int_env("TCM_SOURCE_MAX_TXT", default_max["txt"]),
        "md": _to_int_env("TCM_SOURCE_MAX_MD", default_max["md"]),
        "docx": _to_int_env("TCM_SOURCE_MAX_DOCX", default_max["docx"]),
        "other": _to_int_env("TCM_SOURCE_MAX_OTHER", default_max["other"]),
    }

    min_quota: dict[str, int] = {}
    max_quota: dict[str, int] = {}
    for src in present_sources:
        available = len(groups.get(src, []))
        min_q = max(0, min(top_k, env_min.get(src, 0), available))
        max_q = max(0, min(top_k, env_max.get(src, top_k), available))
        if max_q < min_q:
            max_q = min_q
        min_quota[src] = min_q
        max_quota[src] = max_q

    total_min = sum(min_quota.values())
    if total_min > top_k:
        for src in ("other", "docx", "md", "txt"):
            while total_min > top_k and min_quota.get(src, 0) > 0:
                min_quota[src] -= 1
                total_min -= 1

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, int, str]] = set()
    src_count: dict[str, int] = {src: 0 for src in present_sources}
    ptr: dict[str, int] = {src: 0 for src in present_sources}

    def _item_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
        return (
            _normalize_source(str(row.get("source", ""))),
            str(row.get("file", "")),
            int(row.get("line_no", 0) or 0),
            str(row.get("text", ""))[:60],
        )

    for src in ("txt", "md", "docx", "other"):
        need = min_quota.get(src, 0)
        if need <= 0:
            continue
        bucket = groups.get(src, [])
        while need > 0 and ptr.get(src, 0) < len(bucket):
            row = bucket[ptr[src]]
            ptr[src] += 1
            key = _item_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            src_count[src] = src_count.get(src, 0) + 1
            need -= 1

    if len(selected) < top_k:
        for row in ranked:
            if len(selected) >= top_k:
                break
            src = _normalize_source(str(row.get("source", "")))
            if src_count.get(src, 0) >= max_quota.get(src, top_k):
                continue
            key = _item_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            src_count[src] = src_count.get(src, 0) + 1

    if len(selected) < top_k:
        for row in ranked:
            if len(selected) >= top_k:
                break
            key = _item_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)

    selected.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return selected[:top_k]


def _read_json_file(path: Path) -> dict[str, Any]:
    """安全读取 JSON 文件；出错返回空 dict。"""

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    """安全写入 JSON 文件；失败静默处理（不阻断主流程）。"""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        return


@lru_cache(maxsize=1)
def _tcm_embeddings() -> Embeddings | None:
    """初始化 embedding 模型；缺配置或失败时返回 None。"""

    if not _is_vector_enabled():
        return None

    provider = _embedding_provider()
    model = _embedding_model_name()

    if provider == "siliconflow":
        api_key = str(os.getenv("SILICONFLOW_API_KEY", "")).strip()
        if not api_key:
            return None
        base_url = str(os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")).strip()
        timeout_sec = _to_int_env("TCM_EMBEDDING_TIMEOUT_SEC", 60)
        max_retries = _to_int_env("TCM_EMBEDDING_MAX_RETRIES", 3)
        batch_size = _to_int_env("TCM_EMBEDDING_BATCH_SIZE", 48)
        try:
            return OpenAICompatibleEmbeddings(
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                batch_size=batch_size,
            )
        except Exception as exc:
            log_error("tcm._tcm_embeddings.siliconflow", exc)
            return None

    api_key = str(os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not api_key:
        return None
    try:
        return DashScopeEmbeddings(model=model, dashscope_api_key=api_key)
    except Exception as exc:
        log_error("tcm._tcm_embeddings.dashscope", exc)
        return None


def _case_file_signatures() -> list[dict[str, Any]]:
    """收集医案语料相关文件签名（txt + md + docx）。"""

    files: list[Path] = []
    base = _case_file_path()
    if base.exists():
        files.append(base)

    md_root = _case_markdown_dir_path()
    if md_root.exists():
        files.extend(sorted(md_root.glob("*.md")))

    docx_root = _case_docx_dir_path()
    if docx_root.exists():
        files.extend(sorted(docx_root.glob("*.docx")))

    signatures: list[dict[str, Any]] = []
    for path in files:
        try:
            stat = path.stat()
            signatures.append(
                {
                    "name": path.name,
                    "mtime": int(stat.st_mtime),
                    "size": int(stat.st_size),
                }
            )
        except Exception:
            continue
    return signatures


def _vector_meta_signature(total_cases: int, build_limit: int) -> dict[str, Any]:
    """生成向量库签名，用于判断是否需要重建。"""

    model = _embedding_model_name()
    provider = _embedding_provider()
    backend = _vector_backend()
    return {
        "files": _case_file_signatures(),
        "chunk_config": _case_chunk_config(),
        "total_cases": total_cases,
        "build_limit": build_limit,
        "embedding_provider": provider,
        "embedding_model": model,
        "vector_backend": backend,
        "milvus_case_collection": _milvus_case_collection_name() if backend == "milvus" else "",
        "milvus_index_signature": _milvus_index_signature() if backend == "milvus" else {},
        "corpus_version": 5,
    }


def _is_same_meta(current: dict[str, Any], saved: dict[str, Any]) -> bool:
    """比较当前签名与已保存签名是否一致。"""

    keys = [
        "files",
        "chunk_config",
        "total_cases",
        "build_limit",
        "embedding_provider",
        "embedding_model",
        "vector_backend",
        "milvus_case_collection",
        "milvus_index_signature",
        "corpus_version",
    ]
    return all(current.get(k) == saved.get(k) for k in keys)


@lru_cache(maxsize=1)
def _tcm_vector_store() -> SKLearnVectorStore | Milvus | None:
    """加载或构建本地向量库（带元信息增量判断）。"""

    embeddings = _tcm_embeddings()
    if embeddings is None:
        return None

    case_records = _case_records()
    if not case_records:
        return None

    # 只取前 N 条构建向量库，避免冷启动时间过长。
    build_limit = _vector_build_limit(len(case_records))
    build_records = case_records[:build_limit]
    texts = [str(item.get("text", "")) for item in build_records]
    metadatas = [
        {
            "line_no": idx + 1,
            "source": str(item.get("source", "")),
            "file": str(item.get("file", "")),
            "section_line": int(item.get("section_line", 0) or 0),
            "paragraph_index": int(item.get("paragraph_index", 0) or 0),
            "chunk_index": int(item.get("chunk_index", 0) or 0),
        }
        for idx, item in enumerate(build_records)
    ]

    store_path = _vector_store_path()
    meta_path = _vector_meta_path()
    current_meta = _vector_meta_signature(total_cases=len(case_records), build_limit=build_limit)
    saved_meta = _read_json_file(meta_path)

    backend = _vector_backend()
    if backend == "milvus":
        try:
            collection_name = _milvus_case_collection_name()
            connection_args = _milvus_connection_args()
            index_params = _milvus_index_params()
            search_params = _milvus_search_params()

            if _is_same_meta(current_meta, saved_meta):
                return Milvus(
                    embedding_function=embeddings,
                    collection_name=collection_name,
                    connection_args=connection_args,
                    index_params=index_params,
                    search_params=search_params,
                    auto_id=True,
                    consistency_level="Session",
                    drop_old=False,
                )

            vector_store = Milvus.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                collection_name=collection_name,
                connection_args=connection_args,
                index_params=index_params,
                search_params=search_params,
                drop_old=True,
            )
            _write_json_file(meta_path, current_meta)
            return vector_store
        except Exception as exc:
            # Milvus 不可用时回退 SKLearn，避免线上不可用。
            log_error("tcm._tcm_vector_store.milvus", exc)

    try:
        # 元信息一致时直接复用本地持久化向量库，跳过重建。
        if store_path.exists() and _is_same_meta(current_meta, saved_meta):
            return SKLearnVectorStore(
                embedding=embeddings,
                persist_path=str(store_path),
                serializer="json",
                metric="cosine",
            )

        vector_store = SKLearnVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_path=str(store_path),
            serializer="json",
            metric="cosine",
        )
        vector_store.persist()
        # 每次重建后同步写 meta，供下次启动比较。
        _write_json_file(meta_path, current_meta)
        return vector_store
    except Exception as exc:
        log_error("tcm._tcm_vector_store", exc)
        return None


def _patent_file_signatures() -> list[dict[str, Any]]:
    """收集中成药成品分段文件签名。"""

    signatures: list[dict[str, Any]] = []
    path = _patent_sections_file_path()
    if not path.exists():
        return signatures

    try:
        stat = path.stat()
        signatures.append(
            {
                "name": path.name,
                "mtime": int(stat.st_mtime),
                "size": int(stat.st_size),
            }
        )
    except Exception:
        return []
    return signatures


def _patent_vector_build_limit(total: int) -> int:
    """中成药向量构建上限。"""

    raw = str(os.getenv("TCM_PATENT_VECTOR_BUILD_LIMIT", "5000")).strip()
    try:
        limit = int(raw)
    except ValueError:
        limit = 5000
    return max(200, min(total, limit))


def _patent_meta_signature(total_docs: int, build_limit: int) -> dict[str, Any]:
    """中成药向量库签名。"""

    model = _embedding_model_name()
    provider = _embedding_provider()
    backend = _vector_backend()
    return {
        "files": _patent_file_signatures(),
        "total_docs": total_docs,
        "build_limit": build_limit,
        "embedding_provider": provider,
        "embedding_model": model,
        "vector_backend": backend,
        "milvus_patent_collection": _milvus_patent_collection_name() if backend == "milvus" else "",
        "milvus_index_signature": _milvus_index_signature() if backend == "milvus" else {},
        "corpus_version": 2,
    }


def _is_same_patent_meta(current: dict[str, Any], saved: dict[str, Any]) -> bool:
    """比较中成药向量库签名。"""

    keys = [
        "total_docs",
        "build_limit",
        "embedding_provider",
        "embedding_model",
        "files",
        "vector_backend",
        "milvus_patent_collection",
        "milvus_index_signature",
        "corpus_version",
    ]
    return all(current.get(k) == saved.get(k) for k in keys)


def _embedding_safe_text(text: str, max_chars: int = 1800) -> str:
    """压缩并截断 embedding 输入，避免超过模型长度限制。"""

    compact = re.sub(r"\s+", " ", str(text or "").strip())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars]


def _extract_bracket_field(text: str, field: str, limit: int = 220) -> str:
    """提取形如 `【字段】` 的内容。"""

    pattern = rf"【{re.escape(field)}】\s*(.*?)(?=【[^】]+】|$)"
    m = re.search(pattern, text, re.S)
    if not m:
        return ""
    value = re.sub(r"\s+", " ", str(m.group(1)).strip())
    return value[:limit]


@lru_cache(maxsize=1)
def _patent_medicine_docs() -> list[dict[str, Any]]:
    """加载中成药成品分段 jsonl（一行一个中成药词条）。"""

    path = _patent_sections_file_path()
    if not path.exists():
        return []

    docs: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                raw = str(line or "").strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue

                title = str(row.get("title", "")).strip()
                text = str(row.get("text", "")).strip()
                file_name = str(row.get("file", "")).strip() or path.name
                try:
                    line_no = int(row.get("line_no", 0) or 0)
                except (TypeError, ValueError):
                    line_no = 0
                doc_id = str(row.get("doc_id", "")).strip() or f"{file_name}:{line_no}"
                if not title or not text:
                    continue

                indication = str(row.get("indication", "")).strip()
                contraindications = str(row.get("contraindications", "")).strip()
                cautions = str(row.get("cautions", "")).strip()
                if not indication:
                    indication = _extract_bracket_field(text, "功能与主治", limit=260)
                if not contraindications:
                    contraindications = _extract_bracket_field(text, "禁忌", limit=180)
                if not cautions:
                    cautions = _extract_bracket_field(text, "注意事项", limit=220)

                docs.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "text": text,
                        "file": file_name,
                        "line_no": line_no,
                        "indication": indication,
                        "contraindications": contraindications,
                        "cautions": cautions,
                    }
                )
    except Exception as exc:
        log_error("tcm._patent_medicine_docs", exc)
        return []

    return docs


@lru_cache(maxsize=1)
def _patent_vector_store() -> SKLearnVectorStore | Milvus | None:
    """加载或构建中成药向量库。"""

    embeddings = _tcm_embeddings()
    if embeddings is None:
        return None

    docs = _patent_medicine_docs()
    if not docs:
        return None

    build_limit = _patent_vector_build_limit(len(docs))
    build_docs = docs[:build_limit]
    texts = [
        _embedding_safe_text(
            "；".join(
                [
                    str(d.get("title", "")),
                    f"功能与主治: {d.get('indication', '')}",
                    f"禁忌: {d.get('contraindications', '')}",
                    f"注意事项: {d.get('cautions', '')}",
                    str(d.get("text", "")),
                ]
            ),
            max_chars=1800,
        )
        for d in build_docs
    ]
    metadatas = [
        {
            "doc_id": str(d.get("doc_id", "")),
            "title": str(d.get("title", "")),
            "file": str(d.get("file", "")),
            "line_no": int(d.get("line_no", 0) or 0),
        }
        for d in build_docs
    ]

    store_path = _patent_vector_store_path()
    meta_path = _patent_vector_meta_path()
    current_meta = _patent_meta_signature(total_docs=len(docs), build_limit=build_limit)
    saved_meta = _read_json_file(meta_path)

    backend = _vector_backend()
    if backend == "milvus":
        try:
            collection_name = _milvus_patent_collection_name()
            connection_args = _milvus_connection_args()
            index_params = _milvus_index_params()
            search_params = _milvus_search_params()

            if _is_same_patent_meta(current_meta, saved_meta):
                return Milvus(
                    embedding_function=embeddings,
                    collection_name=collection_name,
                    connection_args=connection_args,
                    index_params=index_params,
                    search_params=search_params,
                    auto_id=True,
                    consistency_level="Session",
                    drop_old=False,
                )

            vector_store = Milvus.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                collection_name=collection_name,
                connection_args=connection_args,
                index_params=index_params,
                search_params=search_params,
                drop_old=True,
            )
            _write_json_file(meta_path, current_meta)
            return vector_store
        except Exception as exc:
            # Milvus 不可用时回退 SKLearn，避免线上不可用。
            log_error("tcm._patent_vector_store.milvus", exc)

    try:
        if store_path.exists() and _is_same_patent_meta(current_meta, saved_meta):
            return SKLearnVectorStore(
                embedding=embeddings,
                persist_path=str(store_path),
                serializer="json",
                metric="cosine",
            )

        vector_store = SKLearnVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_path=str(store_path),
            serializer="json",
            metric="cosine",
        )
        vector_store.persist()
        _write_json_file(meta_path, current_meta)
        return vector_store
    except Exception as exc:
        log_error("tcm._patent_vector_store", exc)
        return None


def _keyword_patent_hits(query: str, symptoms: list[str], limit: int) -> list[dict[str, Any]]:
    """中成药关键词检索。"""

    docs = _patent_medicine_docs()
    q_tokens = _tokenize(query)
    symptom_terms = [s for s in symptoms if s]

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs:
        title = str(doc.get("title", ""))
        text = str(doc.get("text", ""))
        haystack = f"{title}\n{text}"

        score = 0.0
        for symptom in symptom_terms:
            if symptom in haystack:
                score += 2.8
        for tok in q_tokens:
            if tok and tok in haystack:
                score += 0.3
        if score <= 0:
            continue
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for rank, (score, doc) in enumerate(scored[:limit], start=1):
        out.append(
            {
                "doc_id": str(doc.get("doc_id", "")),
                "title": str(doc.get("title", "")),
                "text": str(doc.get("text", "")),
                "file": str(doc.get("file", "")),
                "line_no": int(doc.get("line_no", 0) or 0),
                "indication": str(doc.get("indication", "")),
                "contraindications": str(doc.get("contraindications", "")),
                "cautions": str(doc.get("cautions", "")),
                "keyword_score": float(score),
                "keyword_rank": rank,
            }
        )
    return out


def _vector_patent_hits(query: str, limit: int) -> list[dict[str, Any]]:
    """中成药向量检索。"""

    vector_store = _patent_vector_store()
    if vector_store is None:
        return []

    docs_by_id = {str(d.get("doc_id", "")): d for d in _patent_medicine_docs()}
    out: list[dict[str, Any]] = []
    rows = _safe_similarity_rows(vector_store, query=query, limit=limit, trace_name="tcm._vector_patent_hits")
    if not rows:
        return out

    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, tuple) or len(row) < 2:
            continue
        doc = row[0]
        raw_score = row[1]
        try:
            score = max(0.0, min(1.0, float(raw_score)))
        except (TypeError, ValueError):
            score = 0.0

        metadata = getattr(doc, "metadata", {}) or {}
        doc_id = str(metadata.get("doc_id", "")).strip()
        if not doc_id:
            continue

        raw_doc = docs_by_id.get(doc_id, {})
        out.append(
            {
                "doc_id": doc_id,
                "title": str(raw_doc.get("title", metadata.get("title", ""))),
                "text": str(raw_doc.get("text", getattr(doc, "page_content", ""))),
                "file": str(raw_doc.get("file", metadata.get("file", ""))),
                "line_no": int(raw_doc.get("line_no", metadata.get("line_no", 0)) or 0),
                "indication": str(raw_doc.get("indication", "")),
                "contraindications": str(raw_doc.get("contraindications", "")),
                "cautions": str(raw_doc.get("cautions", "")),
                "vector_score": score,
                "vector_rank": rank,
            }
        )
    return out


def _vector_metric_type(vector_store: Any) -> str:
    """尽量从向量库对象提取 metric_type。"""

    candidates: list[Any] = []
    try:
        candidates.append(getattr(vector_store, "search_params", None))
    except Exception:
        pass
    try:
        candidates.append(getattr(vector_store, "index_params", None))
    except Exception:
        pass

    for item in candidates:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric_type", "")).strip().upper()
        if metric:
            return metric

    return "COSINE"


def _normalize_vector_score(raw_score: Any, metric_type: str) -> float:
    """把不同向量库/度量返回的原始分值归一化到 [0,1]。"""

    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0

    metric = str(metric_type or "").strip().upper()

    # Milvus: COSINE/IP 通常“越大越相似”；L2 通常“越小越相似”。
    if metric in {"COSINE", "IP", "INNER_PRODUCT"}:
        if 0.0 <= score <= 1.0:
            return score
        if -1.0 <= score < 0.0:
            return (score + 1.0) / 2.0
        if score > 1.0:
            return score / (1.0 + score)
        return 0.0

    if metric in {"L2", "EUCLIDEAN"}:
        dist = abs(score)
        return 1.0 / (1.0 + dist)

    return max(0.0, min(1.0, score))


def _safe_similarity_rows(
    vector_store: Any,
    *,
    query: str,
    limit: int,
    trace_name: str,
) -> list[tuple[Any, float]]:
    """兼容不同向量库实现：优先 relevance，失败自动回退 score。"""

    try:
        rows = vector_store.similarity_search_with_relevance_scores(query, k=limit)
        return rows if isinstance(rows, list) else []
    except NotImplementedError:
        # 某些向量库（如 langchain_community 的 Milvus）未实现 relevance 映射。
        pass
    except Exception as exc:
        log_error(trace_name, exc)
        return []

    metric = _vector_metric_type(vector_store)
    try:
        rows = vector_store.similarity_search_with_score(query, k=limit)
    except Exception as exc:
        log_error(trace_name, exc)
        return []

    out: list[tuple[Any, float]] = []
    if not isinstance(rows, list):
        return out

    for row in rows:
        if not isinstance(row, tuple) or len(row) < 2:
            continue
        doc = row[0]
        score = _normalize_vector_score(row[1], metric)
        out.append((doc, score))
    return out


def _hybrid_patent_fusion(
    *,
    keyword_hits: list[dict[str, Any]],
    vector_hits: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """中成药混合召回融合（RRF + 分数补偿）。"""

    if not keyword_hits and not vector_hits:
        return []

    kw_weight = max(0.0, min(1.0, _to_float_env("TCM_PATENT_KEYWORD_WEIGHT", 0.65)))
    vec_weight = max(0.0, min(1.0, _to_float_env("TCM_PATENT_VECTOR_WEIGHT", 0.35)))
    if kw_weight + vec_weight == 0:
        kw_weight, vec_weight = 0.65, 0.35
    rrf_k = int(max(1, _to_float_env("TCM_PATENT_RRF_K", 60)))

    merged: dict[str, dict[str, Any]] = {}
    for item in keyword_hits:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        merged[doc_id] = {
            "doc_id": doc_id,
            "title": item.get("title", ""),
            "text": item.get("text", ""),
            "file": item.get("file", ""),
            "line_no": int(item.get("line_no", 0) or 0),
            "indication": item.get("indication", ""),
            "contraindications": item.get("contraindications", ""),
            "cautions": item.get("cautions", ""),
            "keyword_score": float(item.get("keyword_score", 0.0)),
            "keyword_rank": int(item.get("keyword_rank", 10**6)),
            "vector_score": 0.0,
            "vector_rank": 10**6,
        }

    for item in vector_hits:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        target = merged.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "file": item.get("file", ""),
                "line_no": int(item.get("line_no", 0) or 0),
                "indication": item.get("indication", ""),
                "contraindications": item.get("contraindications", ""),
                "cautions": item.get("cautions", ""),
                "keyword_score": 0.0,
                "keyword_rank": 10**6,
                "vector_score": 0.0,
                "vector_rank": 10**6,
            },
        )
        target["vector_score"] = max(target.get("vector_score", 0.0), float(item.get("vector_score", 0.0)))
        target["vector_rank"] = min(target.get("vector_rank", 10**6), int(item.get("vector_rank", 10**6)))

    fused: list[dict[str, Any]] = []
    for item in merged.values():
        kw_rank = int(item.get("keyword_rank", 10**6))
        vec_rank = int(item.get("vector_rank", 10**6))
        kw_score = float(item.get("keyword_score", 0.0))
        vec_score = float(item.get("vector_score", 0.0))

        keyword_rrf = (1.0 / (rrf_k + kw_rank)) if kw_rank < 10**6 else 0.0
        vector_rrf = (1.0 / (rrf_k + vec_rank)) if vec_rank < 10**6 else 0.0

        hybrid = kw_weight * keyword_rrf + vec_weight * vector_rrf
        hybrid += 0.08 * vec_score
        hybrid += 0.02 * min(1.0, kw_score / 8.0)

        fused.append(
            {
                "doc_id": str(item.get("doc_id", "")),
                "title": str(item.get("title", "")),
                "text": str(item.get("text", "")),
                "file": str(item.get("file", "")),
                "line_no": int(item.get("line_no", 0) or 0),
                "indication": str(item.get("indication", "")),
                "contraindications": str(item.get("contraindications", "")),
                "cautions": str(item.get("cautions", "")),
                "score": round(hybrid, 6),
                "keyword_score": round(kw_score, 4),
                "vector_score": round(vec_score, 4),
                "retriever": "hybrid",
            }
        )

    fused.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return fused[:top_k]


def search_tcm_patent_medicines(query: str, symptoms: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    """中成药检索入口（按单药分段 + 混合召回）。"""

    top_k = max(3, min(int(top_k or 12), 30))
    pool_size = max(top_k * 5, 30)

    keyword_hits = _keyword_patent_hits(query=query, symptoms=symptoms, limit=pool_size)
    vector_hits = _vector_patent_hits(query=query, limit=pool_size)
    fused = _hybrid_patent_fusion(keyword_hits=keyword_hits, vector_hits=vector_hits, top_k=top_k)
    if fused:
        return fused

    out: list[dict[str, Any]] = []
    for item in keyword_hits[:top_k]:
        out.append(
            {
                "doc_id": str(item.get("doc_id", "")),
                "title": str(item.get("title", "")),
                "text": str(item.get("text", "")),
                "file": str(item.get("file", "")),
                "line_no": int(item.get("line_no", 0) or 0),
                "indication": str(item.get("indication", "")),
                "contraindications": str(item.get("contraindications", "")),
                "cautions": str(item.get("cautions", "")),
                "score": round(float(item.get("keyword_score", 0.0)), 4),
                "keyword_score": round(float(item.get("keyword_score", 0.0)), 4),
                "vector_score": 0.0,
                "retriever": "keyword",
            }
        )
    return out


def _normalize_medicine_name(name: str) -> str:
    """药名归一化，便于匹配检索结果。"""

    text = re.sub(r"[\s（）()、，,./\\-]+", "", str(name or "").strip())
    return text


def recommend_tcm_patent_medicines(
    *,
    final_syndrome: str,
    second_choices: list[str],
    symptom_profile: dict[str, Any],
    analysis: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """基于阶段性辨证结果，推荐 1~3 个中成药建议。"""

    syndrome = str(final_syndrome or "").strip()
    if not syndrome or syndrome == "待进一步辨证":
        return []

    symptoms = symptom_profile.get("symptoms", []) if isinstance(symptom_profile, dict) else []
    symptoms = _normalize_str_list(list(symptoms), limit=10)
    choices = _normalize_str_list(list(second_choices or []), limit=3)
    query = "；".join([syndrome, " ".join(choices), " ".join(symptoms), str(analysis or "")[:180]])

    refs = search_tcm_patent_medicines(query=query, symptoms=symptoms, top_k=12)
    if not refs:
        return []

    ref_text = "\n\n".join(
        [
            (
                f"[{item.get('doc_id')}] {item.get('title')} ({item.get('file')}:{item.get('line_no')})\n"
                f"{str(item.get('text', ''))[:1200]}"
            )
            for item in refs[:10]
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是中成药推荐助手。"
                "请基于辨证结果与给定原文，给出1-3个“中成药建议”。"
                "注意：不能给出确诊、不能替代线下面诊。"
                "只输出JSON数组，每项字段必须包含: name, fit_for, cautions, why, evidence_doc_ids。"
                "evidence_doc_ids 必须来自提供的中括号ID。",
            ),
            (
                "human",
                "阶段性最可能证候: {final_syndrome}\n"
                "备选证候: {second_choices}\n"
                "症状: {symptoms}\n"
                "辨证分析: {analysis}\n"
                "中成药原文候选:\n{ref_text}\n"
                "请输出JSON数组。",
            ),
        ]
    )
    chain = prompt | _llm()
    payload = {
        "final_syndrome": syndrome,
        "second_choices": json.dumps(choices, ensure_ascii=False),
        "symptoms": json.dumps(symptoms, ensure_ascii=False),
        "analysis": str(analysis or ""),
        "ref_text": ref_text,
    }
    log_prompt("tcm.recommend_tcm_patent_medicines", prompt, payload)

    try:
        parsed = _invoke_json(
            chain,
            payload,
            mode="array",
            retries=2,
            trace_name="tcm.recommend_tcm_patent_medicines",
        )
    except Exception as exc:
        log_error("tcm.recommend_tcm_patent_medicines", exc)
        parsed = []

    refs_by_id = {str(r.get("doc_id", "")): r for r in refs}
    suggestions: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        norm_name = _normalize_medicine_name(name)
        if not norm_name or norm_name in seen_names:
            continue
        seen_names.add(norm_name)

        evidence_ids = [str(x).strip() for x in _ensure_list(item.get("evidence_doc_ids", []))]
        matched_refs: list[dict[str, Any]] = []
        for eid in evidence_ids:
            ref = refs_by_id.get(eid)
            if ref:
                matched_refs.append(ref)
        if not matched_refs:
            for ref in refs:
                title_norm = _normalize_medicine_name(str(ref.get("title", "")))
                if norm_name and (norm_name in title_norm or title_norm in norm_name):
                    matched_refs.append(ref)
                    break
        if not matched_refs:
            continue

        main_ref = matched_refs[0]
        fit_for = str(item.get("fit_for", "")).strip() or str(main_ref.get("indication", "")).strip() or "请结合辨证结果使用"
        cautions = str(item.get("cautions", "")).strip()
        if not cautions:
            cautions = str(main_ref.get("contraindications", "")).strip() or str(main_ref.get("cautions", "")).strip() or "孕妇、儿童、慢病患者请先咨询医生"
        why = str(item.get("why", "")).strip() or "与当前辨证结果和症状特征匹配"

        suggestions.append(
            {
                "name": name,
                "fit_for": fit_for[:180],
                "cautions": cautions[:220],
                "why": why[:220],
                "evidence": {
                    "doc_id": str(main_ref.get("doc_id", "")),
                    "title": str(main_ref.get("title", "")),
                    "file": str(main_ref.get("file", "")),
                    "line_no": int(main_ref.get("line_no", 0) or 0),
                    "excerpt": str(main_ref.get("text", ""))[:680],
                },
            }
        )
        if len(suggestions) >= max(1, min(int(top_k or 3), 3)):
            break
    return suggestions


def _keyword_case_hits(query: str, symptoms: list[str], limit: int) -> list[dict[str, Any]]:
    """关键词检索：按“症状命中 + token 命中”进行粗打分。"""

    records = _case_records()
    q_tokens = _tokenize(query)
    symptom_terms = [s for s in symptoms if s]

    scored: list[tuple[float, int, str, str, str, int, int]] = []
    for idx, rec in enumerate(records, start=1):
        line = str(rec.get("text", ""))
        source = str(rec.get("source", ""))
        file = str(rec.get("file", ""))
        paragraph_index = int(rec.get("paragraph_index", 0) or 0)
        chunk_index = int(rec.get("chunk_index", 0) or 0)
        score = 0.0
        for symptom in symptom_terms:
            if symptom in line:
                score += 4.0
        for tok in q_tokens:
            if tok and tok in line:
                score += 0.35
        if score > 0:
            scored.append((score, idx, line, source, file, paragraph_index, chunk_index))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]
    out: list[dict[str, Any]] = []
    for rank, (score, line_no, text, source, file, paragraph_index, chunk_index) in enumerate(top, start=1):
        out.append(
            {
                "line_no": line_no,
                "text": text,
                "source": source,
                "file": file,
                "paragraph_index": paragraph_index,
                "chunk_index": chunk_index,
                "keyword_score": float(score),
                "keyword_rank": rank,
            }
        )
    return out


def _vector_case_hits(query: str, limit: int) -> list[dict[str, Any]]:
    """向量检索：返回 line_no + 相似度。"""

    vector_store = _tcm_vector_store()
    if vector_store is None:
        return []

    docs = _safe_similarity_rows(vector_store, query=query, limit=limit, trace_name="tcm._vector_case_hits")
    if not docs:
        return []

    out: list[dict[str, Any]] = []
    for rank, item in enumerate(docs, start=1):
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        doc = item[0]
        raw_score = item[1]
        try:
            score = max(0.0, min(1.0, float(raw_score)))
        except (TypeError, ValueError):
            score = 0.0

        line_no = 0
        source = ""
        file = ""
        paragraph_index = 0
        chunk_index = 0
        if getattr(doc, "metadata", None):
            try:
                line_no = int(doc.metadata.get("line_no", 0))
            except (TypeError, ValueError):
                line_no = 0
            source = str(doc.metadata.get("source", "")).strip()
            file = str(doc.metadata.get("file", "")).strip()
            try:
                paragraph_index = int(doc.metadata.get("paragraph_index", 0) or 0)
            except (TypeError, ValueError):
                paragraph_index = 0
            try:
                chunk_index = int(doc.metadata.get("chunk_index", 0) or 0)
            except (TypeError, ValueError):
                chunk_index = 0

        if line_no <= 0:
            continue

        out.append(
            {
                "line_no": line_no,
                "text": str(getattr(doc, "page_content", "") or ""),
                "source": source,
                "file": file,
                "paragraph_index": paragraph_index,
                "chunk_index": chunk_index,
                "vector_score": score,
                "vector_rank": rank,
            }
        )
    return out


def _hybrid_case_fusion(
    *,
    keyword_hits: list[dict[str, Any]],
    vector_hits: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """混合召回融合：RRF + 分数补偿。"""

    if not keyword_hits and not vector_hits:
        return []

    # 两路权重可通过环境变量微调，默认偏向关键词（更可解释）。
    kw_weight = max(0.0, min(1.0, _to_float_env("TCM_HYBRID_KEYWORD_WEIGHT", 0.65)))
    vec_weight = max(0.0, min(1.0, _to_float_env("TCM_HYBRID_VECTOR_WEIGHT", 0.35)))
    if kw_weight + vec_weight == 0:
        kw_weight = 0.65
        vec_weight = 0.35

    # RRF 的平滑参数，值越大越“保守”。
    rrf_k = int(max(1, _to_float_env("TCM_HYBRID_RRF_K", 60)))

    merged: dict[int, dict[str, Any]] = {}
    for item in keyword_hits:
        line_no = int(item.get("line_no", 0))
        if line_no <= 0:
            continue
        merged[line_no] = {
            "line_no": line_no,
            "text": item.get("text", ""),
            "source": item.get("source", ""),
            "file": item.get("file", ""),
            "paragraph_index": int(item.get("paragraph_index", 0) or 0),
            "chunk_index": int(item.get("chunk_index", 0) or 0),
            "keyword_score": float(item.get("keyword_score", 0.0)),
            "keyword_rank": int(item.get("keyword_rank", 10**6)),
            "vector_score": 0.0,
            "vector_rank": 10**6,
        }

    for item in vector_hits:
        line_no = int(item.get("line_no", 0))
        if line_no <= 0:
            continue
        target = merged.setdefault(
            line_no,
            {
                "line_no": line_no,
                "text": item.get("text", ""),
                "source": item.get("source", ""),
                "file": item.get("file", ""),
                "paragraph_index": int(item.get("paragraph_index", 0) or 0),
                "chunk_index": int(item.get("chunk_index", 0) or 0),
                "keyword_score": 0.0,
                "keyword_rank": 10**6,
                "vector_score": 0.0,
                "vector_rank": 10**6,
            },
        )
        if not target.get("source"):
            target["source"] = item.get("source", "")
        if not target.get("file"):
            target["file"] = item.get("file", "")
        if int(target.get("paragraph_index", 0) or 0) <= 0:
            target["paragraph_index"] = int(item.get("paragraph_index", 0) or 0)
        if int(target.get("chunk_index", 0) or 0) <= 0:
            target["chunk_index"] = int(item.get("chunk_index", 0) or 0)
        target["vector_score"] = max(target["vector_score"], float(item.get("vector_score", 0.0)))
        target["vector_rank"] = min(target["vector_rank"], int(item.get("vector_rank", 10**6)))

    fused: list[dict[str, Any]] = []
    for item in merged.values():
        kw_rank = int(item.get("keyword_rank", 10**6))
        vec_rank = int(item.get("vector_rank", 10**6))
        kw_score = float(item.get("keyword_score", 0.0))
        vec_score = float(item.get("vector_score", 0.0))
        source = str(item.get("source", ""))
        src_weight = _source_weight(source)

        keyword_rrf = (1.0 / (rrf_k + kw_rank)) if kw_rank < 10**6 else 0.0
        vector_rrf = (1.0 / (rrf_k + vec_rank)) if vec_rank < 10**6 else 0.0

        # 主体使用 RRF，附加少量原始分数增强区分度。
        hybrid = kw_weight * keyword_rrf + vec_weight * vector_rrf
        hybrid += 0.08 * vec_score
        hybrid += 0.02 * min(1.0, kw_score / 8.0)
        hybrid *= src_weight

        fused.append(
            {
                "line_no": item["line_no"],
                "text": item.get("text", ""),
                "source": source,
                "file": item.get("file", ""),
                "paragraph_index": int(item.get("paragraph_index", 0) or 0),
                "chunk_index": int(item.get("chunk_index", 0) or 0),
                "score": round(hybrid, 6),
                "keyword_score": round(kw_score, 4),
                "vector_score": round(vec_score, 4),
                "source_weight": round(src_weight, 3),
                "retriever": "hybrid",
            }
        )

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_k]

def _repair_json_with_llm(raw_text: str, mode: str, *, repair_model: str = "main") -> dict[str, Any] | list[Any]:
    """当模型输出不是合法 JSON 时，用低温模型做格式修复。"""

    hint = "JSON对象" if mode == "object" else "JSON数组"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是JSON修复器。把输入修复为合法JSON。只输出JSON，不要解释。",
            ),
            (
                "human",
                "目标类型: {hint}\n原文本:\n{raw_text}\n请输出修复后的JSON。",
            ),
        ]
    )
    payload = {"raw_text": raw_text, "hint": hint}
    log_prompt("tcm._repair_json_with_llm", prompt, payload)

    chain = prompt | (_extractor_llm() if repair_model == "extractor" else _llm())
    repaired = _extract_text(chain.invoke(payload))
    log_response("tcm._repair_json_with_llm", repaired)
    if mode == "object":
        return _extract_json(repaired)
    return _extract_json_array(repaired)

def _invoke_json(
    chain: Any,
    payload: dict[str, Any],
    mode: str,
    retries: int = 2,
    trace_name: str = "tcm._invoke_json",
    repair_model: str = "main",
) -> dict[str, Any] | list[Any]:
    """统一 JSON 调用器：调用 -> 解析 -> 重试/修复。"""

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        # 每次都记录原始返回，方便定位“模型没按 JSON 输出”的问题。
        raw = chain.invoke(payload)
        text = _extract_text(raw)
        log_response(f"{trace_name}.attempt_{attempt + 1}", text)
        try:
            if mode == "object":
                return _extract_json(text)
            return _extract_json_array(text)
        except Exception as exc:
            last_error = exc
            try:
                # 先尝试自动修复 JSON，修复失败再重试下一轮。
                return _repair_json_with_llm(text, mode, repair_model=repair_model)
            except Exception as repair_exc:
                last_error = repair_exc
                continue

    if last_error is not None:
        log_error(trace_name, last_error)
        raise last_error
    err = ValueError("llm_json_parse_failed")
    log_error(trace_name, err)
    raise err

def search_tcm_cases(query: str, symptoms: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    """中医医案检索总入口（优先混合检索，失败退化关键词检索）。"""

    top_k = max(3, min(int(top_k or 12), 30))
    pool_size = max(top_k * 5, 30)

    keyword_hits = _keyword_case_hits(query=query, symptoms=symptoms, limit=pool_size)
    vector_hits = _vector_case_hits(query=query, limit=pool_size)

    # 优先使用混合召回；若向量不可用则降级为关键词检索。
    fused = _hybrid_case_fusion(keyword_hits=keyword_hits, vector_hits=vector_hits, top_k=pool_size)
    fused = _apply_source_quota(fused, top_k=top_k)
    if fused:
        return fused

    out: list[dict[str, Any]] = []
    for item in keyword_hits[:pool_size]:
        src = _normalize_source(str(item.get("source", "")))
        out.append(
            {
                "line_no": int(item.get("line_no", 0)),
                "score": round(float(item.get("keyword_score", 0.0)), 4),
                "text": str(item.get("text", "")),
                "source": src,
                "file": str(item.get("file", "")),
                "paragraph_index": int(item.get("paragraph_index", 0) or 0),
                "chunk_index": int(item.get("chunk_index", 0) or 0),
                "keyword_score": round(float(item.get("keyword_score", 0.0)), 4),
                "vector_score": 0.0,
                "source_weight": round(_source_weight(src), 3),
                "retriever": "keyword",
            }
        )
    return _apply_source_quota(out, top_k=top_k)
@tool("search_tcm_cases")
def search_tcm_cases_tool(query: str, symptoms: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    """Search line-level TCM medical cases from local cleaned dataset."""
    return search_tcm_cases(query=query, symptoms=symptoms, top_k=top_k)


def detect_tcm_red_flags(text: str) -> list[str]:
    """检测红旗症状关键词。"""

    hits = [kw for kw in TCM_RED_FLAG_KEYWORDS if kw and kw in text]
    return _normalize_str_list(hits, limit=8)


def _friendly_extract_error(exc: Exception) -> str:
    """把底层异常转换为用户可读提示。"""

    text = str(exc or "").strip()
    low = text.lower()
    if "dashscope_api_key" in low or "api key" in low or "apikey" in low:
        if _tcm_llm_provider() == "volcengine":
            return "模型鉴权失败，请检查 VOLCENGINE_API_KEY / TCM_VOLCENGINE_API_KEY。"
        return "模型鉴权失败，请检查 DASHSCOPE_API_KEY。"
    if "failed to establish a new connection" in low or "winerror 10013" in low or "connection" in low:
        return "模型网络连接失败，请检查网络/代理/防火墙。"
    if "timed out" in low or "timeout" in low:
        return "模型请求超时，请稍后重试。"
    return f"模型调用失败: {text[:140]}"


def _extract_json_retries() -> int:
    raw = str(os.getenv("TCM_EXTRACT_JSON_RETRIES", "1")).strip()
    try:
        val = int(raw)
    except ValueError:
        val = 1
    return max(0, min(2, val))


def _extract_input_max_chars() -> int:
    raw = str(os.getenv("TCM_EXTRACT_INPUT_MAX_CHARS", "1200")).strip()
    try:
        val = int(raw)
    except ValueError:
        val = 1200
    return max(300, min(4000, val))


def extract_tcm_symptoms(user_text: str) -> dict[str, Any]:
    """症状抽取与标准化（低温 LLM）。"""

    if not _tcm_chat_enabled():
        provider = _tcm_llm_provider()
        key_name = "VOLCENGINE_API_KEY / TCM_VOLCENGINE_API_KEY" if provider == "volcengine" else "DASHSCOPE_API_KEY"
        # 没有模型配置时，返回结构化失败结果，交给上层节点决定下一步提示。
        return {
            "raw_symptoms": [],
            "symptoms": [],
            "summary": f"症状提取模型未配置，请设置 {key_name}。",
            "extraction_ok": False,
            "extraction_source": "llm",
            "extraction_error": f"missing_{provider}_api_key",
            "llm_invoked": False,
        }

    clipped_text = str(user_text or "").strip()[: _extract_input_max_chars()]

    alias_items = list(SYMPTOM_ALIAS_MAP.items())[:12]
    alias_hint = "；".join([f"{k}->{v}" for k, v in alias_items])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你负责把用户口语症状抽取并标准化。"
                "只抽取用户明确提到的信息，不做诊断推断。"
                "只输出JSON对象，字段必须是: raw_symptoms, normalized_symptoms, summary。"
                "raw_symptoms 保留原词；normalized_symptoms 统一为规范术语；"
                "如果没有明确症状则输出空数组。",
            ),
            (
                "human",
                "口语到标准术语映射提示: {alias_hint}\n"
                "用户描述: {user_text}\n"
                "请只输出JSON对象。",
            ),
        ]
    )
    chain = prompt | _extractor_llm()

    payload = {
        "user_text": clipped_text,
        "alias_hint": alias_hint,
    }
    log_prompt("tcm.extract_tcm_symptoms", prompt, payload)

    try:
        parsed = _invoke_json(
            chain,
            payload,
            mode="object",
            retries=_extract_json_retries(),
            trace_name="tcm.extract_tcm_symptoms",
            repair_model="extractor",
        )

        # raw_symptoms：保留用户原始表达中的症状词。
        raw_symptoms = _normalize_str_list(
            list(parsed.get("raw_symptoms", [])) + list(parsed.get("symptoms", [])),
            limit=24,
        )
        # normalized：模型转写后的标准术语。
        normalized = _normalize_str_list(list(parsed.get("normalized_symptoms", [])), limit=24)
        # symptoms：后续检索/辨证真正使用的统一症状集合。
        symptoms = _normalize_str_list(normalized + raw_symptoms, limit=24)
        summary = str(parsed.get("summary", "")).strip() or "；".join(symptoms[:6])

        return {
            "raw_symptoms": raw_symptoms,
            "symptoms": symptoms,
            "summary": summary,
            "extraction_ok": True,
            "extraction_source": "llm",
            "extraction_error": "",
            "llm_invoked": True,
        }
    except Exception as exc:
        log_error("tcm.extract_tcm_symptoms", exc)
        return {
            "raw_symptoms": [],
            "symptoms": [],
            "summary": "症状提取失败。",
            "extraction_ok": False,
            "extraction_source": "llm",
            "extraction_error": _friendly_extract_error(exc),
            "llm_invoked": True,
        }

def _fallback_candidates(symptoms: list[str]) -> list[dict[str, Any]]:
    """候选证候兜底规则（LLM 失败时保障流程可继续）。"""

    text = " ".join(symptoms)
    out: list[dict[str, Any]] = []
    if any(k in text for k in ["腹泻", "食欲差", "乏力", "大便偏稀"]):
        out.append({"name": "脾虚湿困", "score": 0.64, "reason": "腹泻、食欲差、乏力相关", "evidence_lines": [], "differentiators": ["食少便溏", "困倦乏力"]})
    if any(k in text for k in ["畏寒", "乏力", "腰酸"]):
        out.append({"name": "脾肾阳虚", "score": 0.61, "reason": "畏寒、乏力、阳虚倾向", "evidence_lines": [], "differentiators": ["畏寒肢冷", "晨泻"]})
    if any(k in text for k in ["口干", "失眠", "盗汗"]):
        out.append({"name": "阴虚火旺", "score": 0.58, "reason": "口干、失眠、阴虚内热倾向", "evidence_lines": [], "differentiators": ["五心烦热", "盗汗"]})
    if any(k in text for k in ["胸闷", "痰多", "咳嗽"]):
        out.append({"name": "痰湿阻滞", "score": 0.56, "reason": "痰湿阻滞相关症状", "evidence_lines": [], "differentiators": ["痰多", "胸闷"]})
    if not out:
        out = [
            {"name": "脾虚湿困", "score": 0.54, "reason": "症状信息有限，先作待鉴别证候", "evidence_lines": [], "differentiators": ["食少便溏"]},
            {"name": "肝郁气滞", "score": 0.51, "reason": "症状信息有限，需继续问诊", "evidence_lines": [], "differentiators": ["情志相关"]},
        ]
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:3]


def infer_tcm_syndrome_candidates(
    *,
    user_text: str,
    symptom_profile: dict[str, Any],
    case_refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """推断 1~3 个候选证候，并附带理由和区分点。"""

    symptoms = symptom_profile.get("symptoms", []) or []
    online_context = fetch_tavily_context(f"中医证候鉴别 {' '.join(symptoms)}", max_results=3, topic="general", intent_hint="tcm")

    short_cases = case_refs[:8]
    case_text = "\n".join(
        [
            f"line#{item['line_no']} score={item['score']} "
            f"[{item.get('source', 'case')}:{item.get('file', '')}]: {item['text']}"
            for item in short_cases
        ]
    ) or "(无检索结果)"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是中医辨证助手。请基于用户症状与医案证据给出1-3个候选证候。"
                "不能下确诊结论。"
                "只输出JSON数组。每项必须包含字段: name, score(0-1), reason, evidence_lines(行号数组), differentiators(区分症状数组)。",
            ),
            (
                "human",
                "用户描述: {user_text}\n"
                "结构化症状: {symptom_profile}\n"
                "医案RAG:\n{case_text}\n"
                "在线信息:\n{online_context}\n"
                "请输出JSON数组。",
            ),
        ]
    )
    chain = prompt | _llm()

    payload = {
        "user_text": user_text,
        "symptom_profile": json.dumps(symptom_profile, ensure_ascii=False),
        "case_text": case_text,
        "online_context": online_context,
    }
    log_prompt("tcm.infer_tcm_syndrome_candidates", prompt, payload)

    try:
        parsed = _invoke_json(
            chain,
            payload,
            mode="array",
            retries=2,
            trace_name="tcm.infer_tcm_syndrome_candidates",
        )

        out: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            try:
                score = float(item.get("score", 0.5))
            except (TypeError, ValueError):
                score = 0.5
            evidence_lines: list[int] = []
            for x in item.get("evidence_lines", []):
                try:
                    evidence_lines.append(int(x))
                except (TypeError, ValueError):
                    continue
            out.append(
                {
                    "name": name,
                    "score": max(0.0, min(1.0, score)),
                    "reason": str(item.get("reason", "")).strip(),
                    "evidence_lines": evidence_lines[:8],
                    "differentiators": _normalize_str_list(list(item.get("differentiators", [])), limit=6),
                    "source": "llm",
                }
            )
            if len(out) >= 3:
                break

        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out or _fallback_candidates(symptoms)
    except Exception as exc:
        log_error("tcm.infer_tcm_syndrome_candidates", exc)
        return _fallback_candidates(symptoms)

def _question_bank(candidates: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """内置问卷题库（兜底补题用）。"""

    return [
        {
            "question": "最近是否怕冷，或者手脚容易发凉？",
            "purpose": "区分寒热倾向",
            "discriminates": ["脾肾阳虚", "阴虚火旺"],
            "expected_gain": "高",
            "partial_keywords": ["怕冷", "手脚发凉"],
        },
        {
            "question": "是否口干、口渴，且更想喝冷饮？",
            "purpose": "判断津液与内热",
            "discriminates": ["阴虚火旺", "脾虚湿困"],
            "expected_gain": "高",
            "partial_keywords": ["口干", "口渴", "喜冷饮"],
        },
        {
            "question": "夜里是否出汗，或者手心脚心发热？",
            "purpose": "判断阴虚内热",
            "discriminates": ["阴虚火旺", "脾肾阳虚"],
            "expected_gain": "高",
            "partial_keywords": ["夜间出汗", "手足心热"],
        },
        {
            "question": "大便是否偏稀、不成形，早晨更明显吗？",
            "purpose": "判断脾肾阳虚与脾虚湿困",
            "discriminates": ["脾肾阳虚", "脾虚湿困"],
            "expected_gain": "高",
            "partial_keywords": ["大便偏稀", "大便不成形", "晨起明显"],
        },
        {
            "question": "最近是否食欲下降，吃完饭后容易腹胀？",
            "purpose": "判断脾胃运化",
            "discriminates": ["脾虚湿困", "肝郁气滞"],
            "expected_gain": "中",
            "partial_keywords": ["食欲下降", "饭后腹胀"],
        },
        {
            "question": "睡眠是否变浅，容易醒，或者多梦？",
            "purpose": "判断心肾失调",
            "discriminates": ["阴虚火旺", "肝郁气滞"],
            "expected_gain": "中",
            "partial_keywords": ["睡眠浅", "易醒", "多梦"],
        },
        {
            "question": "是否总觉得疲劳乏力，说话都懒得多说？",
            "purpose": "判断气虚程度",
            "discriminates": ["脾虚湿困", "脾肾阳虚"],
            "expected_gain": "中",
            "partial_keywords": ["疲劳", "乏力", "懒言"],
        },
        {
            "question": "小便是否偏清、量多，或者夜尿增多？",
            "purpose": "判断阳虚倾向",
            "discriminates": ["脾肾阳虚", "阴虚火旺"],
            "expected_gain": "中",
            "partial_keywords": ["小便清长", "小便量多", "夜尿增多"],
        },
        {
            "question": "是否有胸闷、痰多，舌苔看起来偏腻？",
            "purpose": "判断痰湿阻滞",
            "discriminates": ["痰湿阻滞", "脾虚湿困"],
            "expected_gain": "中",
            "partial_keywords": ["胸闷", "痰多", "舌苔腻"],
        },
        {
            "question": "情绪起伏后，症状会不会更明显？",
            "purpose": "判断肝郁因素",
            "discriminates": ["肝郁气滞", "脾虚湿困"],
            "expected_gain": "中",
            "partial_keywords": ["情绪波动", "症状加重"],
        },
        {
            "question": "最近是否容易腹痛，按压后会感觉舒服一些？",
            "purpose": "判断虚寒与气滞",
            "discriminates": ["脾肾阳虚", "肝郁气滞"],
            "expected_gain": "中",
            "partial_keywords": ["腹痛", "喜按"],
        },
        {
            "question": "白天是否容易困倦，头身感觉沉重？",
            "purpose": "判断湿困程度",
            "discriminates": ["脾虚湿困", "阴虚火旺"],
            "expected_gain": "中",
            "partial_keywords": ["困倦", "头身困重"],
        },
        {
            "question": "近期是否心烦急躁，或者胸胁胀满不舒？",
            "purpose": "判断肝郁化热倾向",
            "discriminates": ["肝郁气滞", "阴虚火旺"],
            "expected_gain": "中",
            "partial_keywords": ["心烦", "急躁", "胸胁胀满"],
        },
        {
            "question": "饭后是否反酸、嗳气，或胃里不舒服？",
            "purpose": "判断肝胃不和线索",
            "discriminates": ["肝郁气滞", "脾虚湿困"],
            "expected_gain": "中",
            "partial_keywords": ["反酸", "嗳气", "胃部不适"],
        },
    ]


def _fallback_questionnaire(
    *,
    asked_question_keys: list[str] | None,
    target_count: int,
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """从内置题库补齐问卷，确保达到目标题量。"""

    asked = set(asked_question_keys or [])
    bank = _question_bank(candidates)
    out: list[dict[str, Any]] = []

    for idx, item in enumerate(bank, start=1):
        key = build_question_key(item["question"])
        if key in asked:
            continue
        out.append(
            {
                "id": f"q{idx}",
                "question": item["question"],
                "purpose": item["purpose"],
                "discriminates": item.get("discriminates", []),
                "expected_gain": item.get("expected_gain", "中"),
                "partial_keywords": _normalize_partial_keywords(
                    _ensure_list(item.get("partial_keywords", [])),
                    item["question"],
                    limit=6,
                ),
                "options": list(QUESTION_ANSWER_OPTIONS),
            }
        )
        if len(out) >= target_count:
            break

    return out


def build_tcm_questionnaire(
    *,
    user_text: str,
    symptom_profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    case_refs: list[dict[str, Any]],
    asked_question_keys: list[str] | None = None,
    target_count: int = 6,
) -> list[dict[str, Any]]:
    """生成区分问卷（支持“部分是”及关键词子项）。"""
    # 在线信息可作为辅助背景（若关闭联网会返回占位文本）。
    online_context = fetch_tavily_context(
        f"中医证候 区分症状 {' '.join([c.get('name', '') for c in candidates])}",
        max_results=3,
        topic="general", intent_hint="tcm",
    )

    case_text = "\n".join(
        [
            f"line#{item['line_no']} score={item['score']} "
            f"[{item.get('source', 'case')}:{item.get('file', '')}]: {item['text']}"
            for item in case_refs[:8]
        ]
    ) or "(无检索结果)"
    # 题量限制：太少区分度不足，太多会增加用户负担。
    target_count = max(6, min(target_count, 12))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是中医辨证问卷生成助手。"
                "目标: 为了确定更加准确的证候，区分相似证候的关键区别，生成高区分度的不同证候的关键性问题，尽快减少轮次。"
                "要求:"
                "1) 每题必须可回答 是/部分是/否/不确定；"
                "2) 用通俗易懂、口语化的中文提问，避免晦涩术语；"
                "3) 问题应优先区分前两位候选证候；"
                "4) 不得与历史问题重复；"
                "5) 若题干涉及多个症状，partial_keywords 必须按症状逐个拆分；若只涉及一个症状，也要给出1-3个可勾选关键词；"
                "6) 输出题数尽量接近目标题数。"
                "只输出JSON数组，每项字段: id, question, purpose, discriminates, expected_gain, partial_keywords。",
            ),
            (
                "human",
                "目标题数: {target_count}\n"
                "用户描述: {user_text}\n"
                "症状结构化: {symptom_profile}\n"
                "候选证候: {candidates}\n"
                "已问问题key: {asked_question_keys}\n"
                "医案RAG:\n{case_text}\n"
                "在线信息:\n{online_context}\n"
                "请输出JSON数组，不要输出额外解释。",
            ),
        ]
    )

    chain = prompt | _llm()
    payload = {
        "user_text": user_text,
        "symptom_profile": json.dumps(symptom_profile, ensure_ascii=False),
        "candidates": json.dumps(candidates, ensure_ascii=False),
        "asked_question_keys": json.dumps(asked_question_keys or [], ensure_ascii=False),
        "case_text": case_text,
        "online_context": online_context,
        "target_count": target_count,
    }
    log_prompt("tcm.build_tcm_questionnaire", prompt, payload)

    try:
        parsed = _invoke_json(
            chain,
            payload,
            mode="array",
            retries=2,
            trace_name="tcm.build_tcm_questionnaire",
        )
        questions = _normalize_questionnaire(parsed, asked_question_keys=asked_question_keys)
    except Exception as exc:
        log_error("tcm.build_tcm_questionnaire", exc)
        questions = []

    if len(questions) < target_count:
        existed_keys = set(collect_question_keys(questions))
        supplement = _fallback_questionnaire(
            asked_question_keys=list((asked_question_keys or []) + list(existed_keys)),
            target_count=target_count - len(questions),
            candidates=candidates,
        )
        questions.extend(supplement)

    return questions[:target_count]

def summarize_questionnaire_answers(questionnaire: list[dict[str, Any]], answers: dict[str, Any]) -> str:
    """将结构化问卷答案压缩成人类可读摘要。"""

    q_map: dict[str, str] = {}
    for q in questionnaire:
        qid = str(q.get("id", "")).strip()
        question = str(q.get("question", "")).strip()
        if qid and question:
            q_map[qid] = question

    parts: list[str] = []
    for qid, answer in answers.items():
        q_text = q_map.get(str(qid), str(qid))
        normalized = normalize_tcm_answer(answer)
        val = normalized.get("value", "不确定")
        if val != "部分是":
            parts.append(f"{q_text}:{val}")
            continue

        details: list[str] = []
        details.extend(normalized.get("selected_keywords", []))
        other_text = str(normalized.get("other_text", "")).strip()
        if other_text:
            details.append(f"其他:{other_text}")
        if details:
            parts.append(f"{q_text}:部分是({ '、'.join(details) })")
        else:
            parts.append(f"{q_text}:部分是")
    return "；".join(parts)


def compute_tcm_confidence(
    *,
    candidates: list[dict[str, Any]],
    answers: dict[str, Any],
    questionnaire: list[dict[str, Any]],
) -> float:
    """计算辨证置信度（启发式组合评分）。

    评分组成：
    - top_score：第一候选本身分值；
    - margin：第一与第二候选差距；
    - answer_ratio：问卷完成度；
    - evidence_ratio：证据行覆盖；
    - partial_detail_ratio：“部分是”细节完整度。
    """

    if not candidates:
        return 0.35

    sorted_candidates = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    top_score = max(0.0, min(1.0, float(sorted_candidates[0].get("score", 0.0))))
    second_score = 0.0
    if len(sorted_candidates) > 1:
        second_score = max(0.0, min(1.0, float(sorted_candidates[1].get("score", 0.0))))
    margin = max(0.0, top_score - second_score)

    total_questions = max(1, len(questionnaire))
    normalized_answers = [normalize_tcm_answer(v) for v in answers.values()]
    answered = len([a for a in normalized_answers if str(a.get("value", "")).strip() in QUESTION_ANSWER_OPTIONS])
    answer_ratio = min(1.0, answered / total_questions)

    partial_answers = [a for a in normalized_answers if a.get("value") == "部分是"]
    partial_details = 0
    for item in partial_answers:
        partial_details += len(item.get("selected_keywords", []))
        if str(item.get("other_text", "")).strip():
            partial_details += 1
    partial_detail_ratio = 0.0
    if partial_answers:
        partial_detail_ratio = min(1.0, partial_details / max(1, len(partial_answers) * 3))

    evidence_lines: set[int] = set()
    for cand in sorted_candidates[:2]:
        for line in cand.get("evidence_lines", []):
            try:
                evidence_lines.add(int(line))
            except (TypeError, ValueError):
                continue
    evidence_ratio = min(1.0, len(evidence_lines) / 5.0)

    confidence = 0.5 * top_score + 0.3 * margin + 0.12 * answer_ratio + 0.05 * evidence_ratio + 0.03 * partial_detail_ratio
    return round(max(0.05, min(0.95, confidence)), 4)


def finalize_tcm_assessment(
    *,
    user_text: str,
    symptom_profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    questionnaire: list[dict[str, Any]],
    answers: dict[str, Any],
    case_refs: list[dict[str, Any]],
    confidence: float = 0.0,
    round_no: int = 1,
    answers_history: list[str] | None = None,
    red_flags: list[str] | None = None,
) -> dict[str, Any]:
    """生成阶段性辨证结果（明确非确诊）。

    输出只作为“问诊阶段参考”，不会给出临床确诊或处方建议。
    """

    online_context = fetch_tavily_context("中医辨证 建议与诊断边界", max_results=3, topic="general", intent_hint="tcm")
    case_text = "\n".join(
        [
            f"line#{item['line_no']} score={item['score']} "
            f"[{item.get('source', 'case')}:{item.get('file', '')}]: {item['text']}"
            for item in case_refs[:10]
        ]
    ) or "(无检索结果)"

    fallback = {
        "final_syndrome": candidates[0]["name"] if candidates else "待进一步辨证",
        "second_choices": [str(c.get("name", "")) for c in candidates[1:3]],
        "analysis": "当前信息仍有限，建议继续补充症状并线下就医面诊确认。",
        "advice": "本结果仅作中医辨证参考，不能替代医生面诊与检查。",
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是中医辨证总结助手。请根据候选证候、问卷回答与医案证据生成阶段性结果。"
                "不能输出确诊。"
                "只输出JSON对象，字段: final_syndrome, second_choices, analysis, advice。",
            ),
            (
                "human",
                "用户描述: {user_text}\n"
                "症状结构化: {symptom_profile}\n"
                "候选证候: {candidates}\n"
                "当前轮问卷: {questionnaire}\n"
                "当前轮回答: {answers}\n"
                "历史问答摘要: {answers_history}\n"
                "红旗症状: {red_flags}\n"
                "当前置信度: {confidence}\n"
                "当前轮次: {round_no}\n"
                "医案RAG:\n{case_text}\n"
                "在线信息:\n{online_context}\n"
                "请输出JSON对象。",
            ),
        ]
    )
    chain = prompt | _llm()

    payload = {
        "user_text": user_text,
        "symptom_profile": json.dumps(symptom_profile, ensure_ascii=False),
        "candidates": json.dumps(candidates, ensure_ascii=False),
        "questionnaire": json.dumps(questionnaire, ensure_ascii=False),
        "answers": json.dumps(answers, ensure_ascii=False),
        "answers_history": json.dumps(answers_history or [], ensure_ascii=False),
        "red_flags": json.dumps(red_flags or [], ensure_ascii=False),
        "confidence": confidence,
        "round_no": round_no,
        "case_text": case_text,
        "online_context": online_context,
    }
    log_prompt("tcm.finalize_tcm_assessment", prompt, payload)

    try:
        parsed = _invoke_json(
            chain,
            payload,
            mode="object",
            retries=2,
            trace_name="tcm.finalize_tcm_assessment",
        )
        result_data = {
            "final_syndrome": str(parsed.get("final_syndrome", fallback["final_syndrome"])).strip(),
            "second_choices": _normalize_str_list(list(parsed.get("second_choices", [])), limit=3),
            "analysis": str(parsed.get("analysis", fallback["analysis"])).strip(),
            "advice": str(parsed.get("advice", fallback["advice"])).strip(),
        }
    except Exception as exc:
        log_error("tcm.finalize_tcm_assessment", exc)
        result_data = dict(fallback)

    try:
        suggestions = recommend_tcm_patent_medicines(
            final_syndrome=str(result_data.get("final_syndrome", "")),
            second_choices=list(result_data.get("second_choices", [])),
            symptom_profile=symptom_profile,
            analysis=str(result_data.get("analysis", "")),
            top_k=3,
        )
        result_data["patent_medicine_suggestions"] = suggestions
    except Exception as exc:
        log_error("tcm.finalize_tcm_assessment.patent_medicine", exc)
        result_data["patent_medicine_suggestions"] = []

    return result_data
