#!/usr/bin/env python3
from __future__ import annotations

# MOD: 5条同题对照测试脚本（static-split vs nexus-dynamic）
import argparse
import json
import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.evaluator import calculate_rouge_l
from data.nexus_system import NexusSystem


def load_questions(data_path: Path, limit: int) -> list[dict]:
    with data_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"数据格式错误: {data_path}")

    selected = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if question:
            selected.append(
                {
                    "question": question,
                    "answer": answer,
                }
            )
        if len(selected) >= limit:
            break

    if not selected:
        raise ValueError("未从数据集中读取到有效问题")

    return selected


def run_single(nexus: NexusSystem, question: str, gold_answer: str, mode: str) -> dict:
    start = time.perf_counter()
    response, retrieved_docs = nexus.ask(question, mode)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000.0
    rouge_l = calculate_rouge_l(response, gold_answer)
    retrieval_count = len(retrieved_docs) if retrieved_docs else 0

    return {
        "question": question,
        "mode": mode,
        "retrieval_count": retrieval_count,
        "rouge_l": round(float(rouge_l), 6),
        "latency_ms": round(float(latency_ms), 4),
    }


def print_table(rows: list[dict]) -> None:
    headers = ["question", "mode", "retrieval_count", "rouge_l", "latency_ms"]

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                (row["question"][:56] + "...") if len(row["question"]) > 59 else row["question"],
                row["mode"],
                str(row["retrieval_count"]),
                f"{row['rouge_l']:.6f}",
                f"{row['latency_ms']:.4f}",
            ]
        )

    col_widths = [len(h) for h in headers]
    for tr in table_rows:
        for i, cell in enumerate(tr):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_line(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(col_widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt_line(headers))
    print(sep)
    for tr in table_rows:
        print(fmt_line(tr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare static-split and nexus-dynamic on same questions")
    parser.add_argument("--data", type=str, default="./data/hotpotqa_dev.json", help="HotpotQA JSON path")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"找不到数据文件: {data_path}")

    questions = load_questions(data_path, limit=max(1, int(args.limit)))
    nexus = NexusSystem()

    modes = ["static-split", "nexus-dynamic"]
    rows: list[dict] = []

    for item in questions:
        question = item["question"]
        gold_answer = item["answer"]

        for mode in modes:
            try:
                rows.append(run_single(nexus, question, gold_answer, mode))
            except Exception as exc:
                rows.append(
                    {
                        "question": question,
                        "mode": mode,
                        "retrieval_count": -1,
                        "rouge_l": -1.0,
                        "latency_ms": -1.0,
                    }
                )
                print(f"[ERROR] mode={mode} question={question[:40]}... err={exc}")

    print_table(rows)


if __name__ == "__main__":
    main()
