from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .workflow import run_agent


def print_result(result: dict) -> None:
    print("\n=== Agent Result ===")
    print(result.get("answer", ""))
    print("\n---")
    print(f"intent: {result.get('intent')}")
    print(f"intent_source: {result.get('intent_source', 'n/a')}")
    print(f"risk_level: {result.get('risk_level')}")
    print(f"confidence: {result.get('confidence')}")
    print(f"handoff: {result.get('handoff')}")
    print(f"citations: {result.get('citations')}")
    if result.get("handoff"):
        print(f"handoff_summary: {result.get('handoff_summary')}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Medical support agent demo")
    parser.add_argument("--query", type=str, help="Single query mode")
    args = parser.parse_args()

    if args.query:
        result = run_agent(args.query)
        print_result(result)
        return

    print("Medical Support Agent CLI. 输入 q 退出。")
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        result = run_agent(user_input)
        print_result(result)


if __name__ == "__main__":
    main()
