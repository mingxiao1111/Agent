from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from locust import HttpUser, between, task


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "tests" / "datasets"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


INTENT_CASES = _load_jsonl(DATASET_DIR / "intent_complex_cases.jsonl")
INJECTION_CASES = _load_jsonl(DATASET_DIR / "prompt_injection_cases.jsonl")

BLOCK_CASES = [x for x in INJECTION_CASES if bool(x.get("expect_block", False))]
SAFE_CASES = [x for x in INJECTION_CASES if not bool(x.get("expect_block", False))]


class MedicalAssistantUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        self.session_id = ""

    @task(6)
    def functional_chat(self) -> None:
        case = random.choice(INTENT_CASES)
        payload = {
            "query": str(case.get("query", "")),
            "session_id": self.session_id,
            "enable_cache": False,
            "enable_online_search": False,
        }

        with self.client.post("/api/chat", json=payload, name="/api/chat:functional", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")
                return
            try:
                body = resp.json()
            except ValueError:
                resp.failure("invalid_json")
                return
            if not isinstance(body, dict) or not body.get("answer"):
                resp.failure("empty_answer")
                return
            sid = str(body.get("session_id", "")).strip()
            if sid:
                self.session_id = sid
            resp.success()

    @task(2)
    def security_guardrail(self) -> None:
        case = random.choice(BLOCK_CASES)
        payload = {
            "query": str(case.get("query", "")),
            "session_id": self.session_id,
            "enable_cache": False,
            "enable_online_search": False,
        }

        with self.client.post("/api/chat", json=payload, name="/api/chat:security", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")
                return
            try:
                body = resp.json()
            except ValueError:
                resp.failure("invalid_json")
                return
            risk_level = str(body.get("risk_level", ""))
            handoff = bool(body.get("handoff", False))
            if risk_level != "high" or not handoff:
                resp.failure(f"guardrail_miss:risk={risk_level},handoff={handoff}")
                return
            resp.success()

    @task(2)
    def stream_chat(self) -> None:
        picked = random.choice(SAFE_CASES or INTENT_CASES)
        payload = {
            "query": str(picked.get("query", "")),
            "session_id": self.session_id,
            "enable_cache": False,
            "enable_online_search": False,
        }

        with self.client.post("/api/chat/stream", json=payload, name="/api/chat/stream", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")
                return
            text = resp.text or ""
            if "event: done" not in text:
                resp.failure("missing_done_event")
                return
            if "event: meta" not in text:
                resp.failure("missing_meta_event")
                return
            resp.success()
