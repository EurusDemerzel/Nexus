#!/usr/bin/env python3
import argparse
import csv
import json
import os
import traceback
from pathlib import Path

from tqdm import tqdm

from evaluator import evaluate_single_query
from nexus_system import NexusSystem


def load_qa_questions(json_path: str, limit: int | None = None) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            data = payload["data"]
        elif isinstance(payload.get("examples"), list):
            data = payload["examples"]
        else:
            data = []
    elif isinstance(payload, list):
        data = payload
    else:
        data = []

    normalized: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        question = item.get("question", "")
        if isinstance(question, dict):
            question = question.get("text", "")

        answer = item.get("answer", "")
        if isinstance(answer, list):
            answer = " ".join(str(x) for x in answer if x is not None)
        elif isinstance(answer, dict):
            answer = answer.get("text", "")

        supporting_facts = item.get("supporting_facts", [])
        if not isinstance(supporting_facts, list):
            supporting_facts = []

        gold_answers = item.get("gold_answers", [])
        normalized_gold_answers: list[str] = []
        if isinstance(gold_answers, list):
            normalized_gold_answers = [str(x).strip() for x in gold_answers if str(x).strip()]

        normalized.append(
            {
                "question": str(question or "").strip(),
                "answer": str(answer or "").strip(),
                "gold_answers": normalized_gold_answers,
                "supporting_facts": supporting_facts,
            }
        )

    data = normalized

    if limit is not None:
        data = data[:limit]
    return data


def ensure_output_parent(output_path: str) -> None:
    parent = Path(output_path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus EMNLP week-1 experiment runner")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["cloud-only", "static-split", "nexus-dynamic"],
        help="运行模式",
    )
    parser.add_argument("--limit", type=int, default=200, help="测试问题数量")
    parser.add_argument("--output", type=str, default="./data/experiment_results.csv", help="输出 CSV 路径")
    parser.add_argument("--data", type=str, default="./data/triviaqa_dev.json", help="问答 JSON 路径")
    parser.add_argument(
        "--retriever",
        type=str,
        default=os.getenv("NEXUS_RETRIEVER", "notes"),
        choices=["notes", "faiss"],
        help="本地检索器类型",
    )
    args = parser.parse_args()

    os.environ["NEXUS_RETRIEVER"] = args.retriever

    print(f"📂 加载数据: {args.data}")
    dataset = load_qa_questions(args.data, limit=args.limit)
    print(f"📊 共 {len(dataset)} 个问题，将运行模式: {args.mode}\n")
    print(f"🔎 本地检索器: {args.retriever}")

    if not dataset:
        raise SystemExit("数据集为空，请先执行 import_hotpotqa.py 生成 hotpotqa_dev.json")

    ensure_output_parent(args.output)
    nexus = NexusSystem()

    fieldnames = [
        "question",
        "gold_answer",
        "model_answer",
        "mode",
        "latency_ms",
        "cpu_time_sec",
        "mem_bytes",
        "rouge_l",
        "exact_match",
        "token_f1",
        "bleu_1",
        "bleu_4",
        "retrieval_precision",
        "privacy_mode",
        "privacy_overhead_ms",
    ]

    success_count = 0
    error_count = 0
    
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item_idx, item in enumerate(tqdm(dataset, desc=f"Running {args.mode}"), start=1):
            question = item.get("question", "")
            gold_answers = item.get("gold_answers", [])
            if isinstance(gold_answers, list) and gold_answers:
                gold_answer = gold_answers
            else:
                gold_answer = item.get("answer", "")
            gold_facts = item.get("supporting_facts", [])

            try:
                metrics = evaluate_single_query(
                    question=question,
                    gold_answer=gold_answer,
                    gold_supporting_facts=gold_facts,
                    mode=args.mode,
                    nexus_system=nexus,
                )
                writer.writerow(metrics)
                csvfile.flush()
                success_count += 1
            except Exception as exc:
                error_count += 1
                error_msg = f"{type(exc).__name__}: {str(exc)}"
                print(f"\n❌ 错误 [#{item_idx}]: {question[:50]}...")
                print(f"   详情: {error_msg}")
                print(f"   堆栈:\n{traceback.format_exc()}\n")
                
                writer.writerow(
                    {
                        "question": question,
                        "gold_answer": " | ".join(gold_answer) if isinstance(gold_answer, list) else gold_answer,
                        "model_answer": f"ERROR: {error_msg}",
                        "mode": args.mode,
                        "latency_ms": -1,
                        "cpu_time_sec": -1,
                        "mem_bytes": -1,
                        "rouge_l": -1,
                        "exact_match": -1,
                        "token_f1": -1,
                        "bleu_1": -1,
                        "bleu_4": -1,
                        "retrieval_precision": -1,
                        "privacy_mode": "error",
                        "privacy_overhead_ms": -1,
                    }
                )
                csvfile.flush()

    print(f"\n{'='*60}")
    print(f"✅ 实验完成!")
    print(f"   成功: {success_count}/{len(dataset)}")
    print(f"   失败: {error_count}/{len(dataset)}")
    print(f"   结果保存至: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
