from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .guardrails import classify_intent
from .llm_chains import generate_followups_with_llm
from .state import AgentState
from .workflow import (
    emergency_node,
    intent_node,
    normalize_node,
    response_node,
    retrieve_node,
    risk_node,
    risk_route,
    tools_node,
)

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DB_PATH = BASE_DIR / "data" / "pipeline_cache.sqlite3"
CACHE_NAMESPACE_VERSION = str(os.getenv("PIPELINE_CACHE_NAMESPACE", "v1")).strip() or "v1"


def _to_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return int(default)


def _to_bool_env(name: str, default: bool) -> bool:
    value = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


class ThreeLevelCache:
    """三级缓存：L1(进程内) + L2(SQLite TTL) + L3(SQLite 冷缓存)。"""

    def __init__(
        self,
        db_path: Path,
        *,
        l1_ttl_sec: int = 60,
        l2_ttl_sec: int = 900,
        l3_promote_ttl_sec: int = 300,
        enable_l3: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.l1_ttl_sec = max(1, int(l1_ttl_sec))
        self.l2_ttl_sec = max(5, int(l2_ttl_sec))
        self.l3_promote_ttl_sec = max(5, int(l3_promote_ttl_sec))
        self.enable_l3 = bool(enable_l3)
        self._lock = threading.RLock()
        self._l1: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_l2 (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_l3 (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )
            conn.commit()

    def _l1_get(self, namespace: str, cache_key: str) -> tuple[dict[str, Any] | None, str]:
        now = time.time()
        with self._lock:
            record = self._l1.get((namespace, cache_key))
            if not record:
                return None, "miss"
            expires_at, payload = record
            if expires_at < now:
                self._l1.pop((namespace, cache_key), None)
                return None, "miss"
            return copy.deepcopy(payload), "l1"

    def _l1_set(self, namespace: str, cache_key: str, payload: dict[str, Any], ttl_sec: int) -> None:
        expires_at = time.time() + max(1, int(ttl_sec))
        with self._lock:
            self._l1[(namespace, cache_key)] = (expires_at, copy.deepcopy(payload))

    def _load_l2(self, namespace: str, cache_key: str) -> dict[str, Any] | None:
        now = time.time()
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT payload FROM cache_l2 WHERE namespace=? AND cache_key=? AND expires_at>?",
                (namespace, cache_key, now),
            ).fetchone()
            if not row:
                return None
            try:
                data = json.loads(str(row[0]))
            except Exception:
                return None
            if isinstance(data, dict):
                return data
        return None

    def _save_l2(self, namespace: str, cache_key: str, payload: dict[str, Any], ttl_sec: int) -> None:
        now = time.time()
        expires_at = now + max(5, int(ttl_sec))
        dump = json.dumps(payload, ensure_ascii=False)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO cache_l2(namespace, cache_key, payload, expires_at, updated_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    payload=excluded.payload,
                    expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at
                """,
                (namespace, cache_key, dump, expires_at, now),
            )
            conn.commit()

    def _load_l3(self, namespace: str, cache_key: str) -> dict[str, Any] | None:
        if not self.enable_l3:
            return None
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT payload FROM cache_l3 WHERE namespace=? AND cache_key=?",
                (namespace, cache_key),
            ).fetchone()
            if not row:
                return None
            try:
                data = json.loads(str(row[0]))
            except Exception:
                return None
            if isinstance(data, dict):
                return data
        return None

    def _save_l3(self, namespace: str, cache_key: str, payload: dict[str, Any]) -> None:
        if not self.enable_l3:
            return
        now = time.time()
        dump = json.dumps(payload, ensure_ascii=False)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO cache_l3(namespace, cache_key, payload, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (namespace, cache_key, dump, now),
            )
            conn.commit()

    def get(self, namespace: str, cache_key: str) -> tuple[dict[str, Any] | None, str]:
        namespaced = f"{CACHE_NAMESPACE_VERSION}:{namespace}"
        value, level = self._l1_get(namespaced, cache_key)
        if value is not None:
            return value, level

        value = self._load_l2(namespaced, cache_key)
        if value is not None:
            self._l1_set(namespaced, cache_key, value, self.l1_ttl_sec)
            return copy.deepcopy(value), "l2"

        value = self._load_l3(namespaced, cache_key)
        if value is not None:
            self._save_l2(namespaced, cache_key, value, self.l3_promote_ttl_sec)
            self._l1_set(namespaced, cache_key, value, self.l1_ttl_sec)
            return copy.deepcopy(value), "l3"
        return None, "miss"

    def set(self, namespace: str, cache_key: str, payload: dict[str, Any], ttl_sec: int | None = None) -> None:
        namespaced = f"{CACHE_NAMESPACE_VERSION}:{namespace}"
        ttl = max(5, int(ttl_sec if ttl_sec is not None else self.l2_ttl_sec))
        self._l1_set(namespaced, cache_key, payload, self.l1_ttl_sec)
        self._save_l2(namespaced, cache_key, payload, ttl)
        self._save_l3(namespaced, cache_key, payload)


PIPELINE_CACHE = ThreeLevelCache(
    CACHE_DB_PATH,
    l1_ttl_sec=_to_int_env("PIPELINE_CACHE_L1_TTL_SEC", 90),
    l2_ttl_sec=_to_int_env("PIPELINE_CACHE_L2_TTL_SEC", 900),
    l3_promote_ttl_sec=_to_int_env("PIPELINE_CACHE_L3_PROMOTE_TTL_SEC", 300),
    enable_l3=_to_bool_env("PIPELINE_CACHE_ENABLE_L3", True),
)


def _build_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _intent_route_signature(
    *,
    llm_provider: str,
    llm_model: str,
    llm_thinking: bool,
) -> dict[str, Any]:
    """意图路由签名：模型相关配置变化时应触发 intent 缓存失效。"""

    return {
        "provider": str(llm_provider or "default"),
        "runtime_model": str(llm_model or ""),
        "runtime_thinking": bool(llm_thinking),
        "volcengine_intent_model": str(os.getenv("VOLCENGINE_INTENT_MODEL", "doubao-seed-2-0-mini-260215")).strip(),
        "volcengine_chat_model": str(
            os.getenv("VOLCENGINE_CHAT_MODEL") or os.getenv("VOLCENGINE_MODEL", "deepseek-v3-2-251201")
        ).strip(),
        "tongyi_router_model": str(os.getenv("TONGYI_ROUTER_MODEL", "")).strip(),
        "tongyi_model": str(os.getenv("TONGYI_MODEL", "")).strip(),
        "intent_cache_version": "v2",
    }


def _trace(trace: list[dict[str, Any]] | None, stage: str, status: str, **meta: Any) -> None:
    if trace is None:
        return
    item: dict[str, Any] = {"stage": stage, "status": status}
    if meta:
        item.update(meta)
    trace.append(item)


async def _run_blocking(fn: Any, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(fn, *args, **kwargs)


def _merge_state(base: AgentState, patch: dict[str, Any]) -> AgentState:
    merged = dict(base)
    merged.update(patch)
    return merged


async def run_agent_async_pipeline(
    user_input: str,
    *,
    llm_provider: str = "default",
    llm_model: str = "",
    llm_thinking: bool = False,
    conversation_history_text: str = "",
    disable_llm_response: bool = False,
    disable_llm_followups: bool = False,
    enable_online_search: bool = False,
    use_cache: bool = True,
    trace: list[dict[str, Any]] | None = None,
) -> AgentState:
    """异步链路：risk 后并发意图+检索，并接入三级缓存。"""

    state: AgentState = {
        "user_input": user_input,
        "llm_provider": str(llm_provider or "default"),
        "llm_model": str(llm_model or ""),
        "llm_thinking": bool(llm_thinking),
        "conversation_history_text": str(conversation_history_text or ""),
        "disable_llm_response": disable_llm_response,
        "disable_llm_followups": disable_llm_followups,
        "enable_online_search": enable_online_search,
    }

    final_key = _build_key(
        {
            "query": user_input,
            "disable_llm_response": disable_llm_response,
            "disable_llm_followups": disable_llm_followups,
            "enable_online_search": enable_online_search,
            "conversation_history_text": str(conversation_history_text or "")[:1200],
            "llm_provider": str(llm_provider or "default"),
            "llm_model": str(llm_model or ""),
            "llm_thinking": bool(llm_thinking),
            "risk_guard_version": "v2",
        }
    )

    if use_cache:
        final_cached, level = PIPELINE_CACHE.get("final_response", final_key)
        if final_cached is not None:
            _trace(trace, "final_response", "cache_hit", cache_level=level)
            return final_cached
        _trace(trace, "final_response", "cache_miss")

    _trace(trace, "normalize", "start")
    normalized = normalize_node(state)
    state = _merge_state(state, normalized)
    _trace(trace, "normalize", "done")

    _trace(trace, "risk", "start")
    risk = risk_node(state)
    state = _merge_state(state, risk)
    _trace(trace, "risk", "done", risk_level=state.get("risk_level", "low"))

    if risk_route(state) == "emergency":
        _trace(trace, "emergency", "start")
        state = _merge_state(state, emergency_node(state))
        _trace(trace, "emergency", "done")
        if use_cache:
            PIPELINE_CACHE.set("final_response", final_key, dict(state), ttl_sec=120)
        return state

    normalized_input = str(state.get("normalized_input", ""))
    intent_key = _build_key(
        {
            "normalized_input": normalized_input,
            "intent_route": _intent_route_signature(
                llm_provider=state.get("llm_provider", "default"),
                llm_model=state.get("llm_model", ""),
                llm_thinking=bool(state.get("llm_thinking", False)),
            ),
        }
    )
    retrieve_key = _build_key({"normalized_input": normalized_input, "k": 3})

    intent_result: dict[str, Any] | None = None
    retrieve_result: dict[str, Any] | None = None
    intent_task: asyncio.Task[Any] | None = None
    retrieve_task: asyncio.Task[Any] | None = None
    followup_prefetch_task: asyncio.Task[Any] | None = None

    if not bool(state.get("disable_llm_followups", False)):
        pre_intent = classify_intent(normalized_input)
        _trace(trace, "followups_prefetch", "start", intent_hint=pre_intent)
        followup_prefetch_task = asyncio.create_task(
            _run_blocking(
                generate_followups_with_llm,
                query=state.get("user_input", ""),
                intent=pre_intent,
                risk_level=state.get("risk_level", "low"),
                conversation_history_text=state.get("conversation_history_text", ""),
                llm_runtime={
                    "provider": state.get("llm_provider", "default"),
                    "model": state.get("llm_model", ""),
                    "thinking": bool(state.get("llm_thinking", False)),
                },
            )
        )

    if use_cache:
        intent_cached, level = PIPELINE_CACHE.get("intent_stage", intent_key)
        if intent_cached is not None:
            intent_result = intent_cached
            _trace(trace, "intent", "cache_hit", cache_level=level)
        else:
            _trace(trace, "intent", "cache_miss")

        retrieve_cached, level = PIPELINE_CACHE.get("retrieve_stage", retrieve_key)
        if retrieve_cached is not None:
            retrieve_result = retrieve_cached
            _trace(trace, "retrieve", "cache_hit", cache_level=level)
        else:
            _trace(trace, "retrieve", "cache_miss")

    if intent_result is None:
        _trace(trace, "intent", "start")
        intent_task = asyncio.create_task(_run_blocking(intent_node, state))
    if retrieve_result is None:
        _trace(trace, "retrieve", "start")
        retrieve_task = asyncio.create_task(_run_blocking(retrieve_node, state))

    if intent_task is not None:
        intent_result = await intent_task
        _trace(trace, "intent", "done")
        if use_cache and isinstance(intent_result, dict):
            PIPELINE_CACHE.set("intent_stage", intent_key, intent_result, ttl_sec=3600)
    if retrieve_task is not None:
        retrieve_result = await retrieve_task
        _trace(trace, "retrieve", "done")
        if use_cache and isinstance(retrieve_result, dict):
            PIPELINE_CACHE.set("retrieve_stage", retrieve_key, retrieve_result, ttl_sec=1800)

    if intent_result:
        state = _merge_state(state, intent_result)
    if retrieve_result:
        state = _merge_state(state, retrieve_result)

    tools_key = _build_key(
        {
            "intent": state.get("intent", "other"),
            "normalized_input": normalized_input,
            "user_input": state.get("user_input", ""),
            "context_docs": state.get("context_docs", []),
        }
    )
    tools_result: dict[str, Any] | None = None

    if use_cache:
        tools_cached, level = PIPELINE_CACHE.get("tools_stage", tools_key)
        if tools_cached is not None:
            tools_result = tools_cached
            _trace(trace, "tools", "cache_hit", cache_level=level)
        else:
            _trace(trace, "tools", "cache_miss")

    if tools_result is None:
        _trace(trace, "tools", "start")
        tools_result = await _run_blocking(tools_node, state)
        _trace(trace, "tools", "done")
        if use_cache and isinstance(tools_result, dict):
            PIPELINE_CACHE.set("tools_stage", tools_key, tools_result, ttl_sec=900)

    if tools_result:
        state = _merge_state(state, tools_result)

    if followup_prefetch_task is not None:
        try:
            prefetched = await asyncio.wait_for(followup_prefetch_task, timeout=0.85)
            if isinstance(prefetched, list) and prefetched:
                state = _merge_state(state, {"prefetched_followups": prefetched})
                _trace(trace, "followups_prefetch", "done", count=len(prefetched))
            else:
                _trace(trace, "followups_prefetch", "empty")
        except asyncio.TimeoutError:
            _trace(trace, "followups_prefetch", "timeout")
        except Exception as exc:
            _trace(trace, "followups_prefetch", "error", error=str(exc))

    _trace(trace, "respond", "start")
    response = await _run_blocking(response_node, state)
    _trace(trace, "respond", "done")
    if response:
        state = _merge_state(state, response)

    if use_cache:
        final_ttl = _to_int_env("PIPELINE_FINAL_CACHE_TTL_SEC", 420)
        PIPELINE_CACHE.set("final_response", final_key, dict(state), ttl_sec=final_ttl)

    return state


def run_agent_async_pipeline_sync(
    user_input: str,
    *,
    llm_provider: str = "default",
    llm_model: str = "",
    llm_thinking: bool = False,
    conversation_history_text: str = "",
    disable_llm_response: bool = False,
    disable_llm_followups: bool = False,
    enable_online_search: bool = False,
    use_cache: bool = True,
    trace: list[dict[str, Any]] | None = None,
) -> AgentState:
    """同步环境入口（Flask 路由里调用）。"""

    return asyncio.run(
        run_agent_async_pipeline(
            user_input,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_thinking=llm_thinking,
            conversation_history_text=conversation_history_text,
            disable_llm_response=disable_llm_response,
            disable_llm_followups=disable_llm_followups,
            enable_online_search=enable_online_search,
            use_cache=use_cache,
            trace=trace,
        )
    )
