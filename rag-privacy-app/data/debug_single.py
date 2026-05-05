#!/usr/bin/env python3
from __future__ import annotations

# MOD: 快速单条调试脚本，复现 static-split / nexus-dynamic 并打印完整日志
import json
import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.evaluator import calculate_rouge_l
from data.nexus_system import NexusSystem


def load_first_question(data_path: Path) -> tuple[str, str]:
    with data_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or not payload:
        raise ValueError("hotpotqa_dev.json 为空或格式错误")

    first = payload[0]
    if not isinstance(first, dict):
        raise ValueError("第一条数据格式错误")

    question = (first.get("question") or "").strip()
    answer = (first.get("answer") or "").strip()

    if not question:
        raise ValueError("第一条问题为空")

    return question, answer


def run_mode(nexus: NexusSystem, question: str, gold_answer: str, mode: str) -> None:
    print("\n" + "=" * 90)
    print(f"[debug_single] mode={mode}")
    print("=" * 90)

    start = time.perf_counter()
    response, retrieved_docs = nexus.ask(question, mode)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000.0
    rouge_l = calculate_rouge_l(response, gold_answer)

    response_preview = (response or "").replace("\n", " ")[:200]
    print(f"[debug_single] retrieved_docs={len(retrieved_docs) if retrieved_docs else 0}")
    print(f"[debug_single] response_preview={response_preview}...")
    print(f"[debug_single] rouge_l={rouge_l:.6f} latency_ms={latency_ms:.4f}")


def main() -> None:
    data_path = (ROOT_DIR / "data" / "hotpotqa_dev.json").resolve()
    if not data_path.exists():
        raise SystemExit(f"找不到数据文件: {data_path}")

    question, gold_answer = load_first_question(data_path)

    print("=" * 90)
    print("[debug_single] 使用第一条 HotpotQA 问题")
    print(f"[debug_single] question={question}")
    print(f"[debug_single] gold_answer={gold_answer}")
    print("=" * 90)

    nexus = NexusSystem()

    run_mode(nexus, question, gold_answer, "static-split")
    run_mode(nexus, question, gold_answer, "nexus-dynamic")


if __name__ == "__main__":
    main()
