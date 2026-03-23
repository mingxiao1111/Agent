from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.workflow import run_agent


CASES_PATH = ROOT / "data" / "eval_cases.json"


def load_cases() -> list[dict]:
    with CASES_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def main() -> None:
    cases = load_cases()

    high_risk_total = 0
    high_risk_hit = 0
    intent_hit = 0
    handoff_count = 0
    latencies = []

    for case in cases:
        start = time.perf_counter()
        result = run_agent(case["query"])
        latencies.append((time.perf_counter() - start) * 1000)

        predicted_high = result.get("risk_level") == "high"
        expected_high = bool(case["expected_high_risk"])
        if expected_high:
            high_risk_total += 1
            if predicted_high:
                high_risk_hit += 1

        if result.get("intent") == case["expected_intent"]:
            intent_hit += 1

        if result.get("handoff"):
            handoff_count += 1

    high_risk_recall = high_risk_hit / high_risk_total if high_risk_total else 1.0
    intent_accuracy = intent_hit / len(cases) if cases else 0.0
    handoff_rate = handoff_count / len(cases) if cases else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0

    print("=== Eval Metrics ===")
    print(f"cases: {len(cases)}")
    print(f"high_risk_recall: {high_risk_recall:.2%}")
    print(f"intent_accuracy: {intent_accuracy:.2%}")
    print(f"handoff_rate: {handoff_rate:.2%}")
    print(f"avg_latency_ms: {avg_latency:.1f}")


if __name__ == "__main__":
    main()
