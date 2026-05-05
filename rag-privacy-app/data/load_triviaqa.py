#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_dataset_with_retry(split: str, retries: int = 5):
    try:
        from datasets import DownloadConfig, load_dataset  # type: ignore
    except Exception as exc:
        raise SystemExit("缺少依赖 datasets，请先执行: pip install datasets") from exc

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(
                "trivia_qa",
                "rc",
                split=split,
                streaming=True,
                download_config=DownloadConfig(max_retries=5),
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait_s = min(30.0, 2 ** (attempt - 1))
            print(f"load_dataset 第 {attempt}/{retries} 次失败: {type(exc).__name__}: {exc}")
            print(f"{wait_s:.1f}s 后重试...")
            time.sleep(wait_s)

    raise SystemExit(f"加载 TriviaQA 失败（已重试 {retries} 次）: {last_error}")


def load_and_convert_triviaqa(split: str = "validation", sample_size: int | None = None) -> list[dict]:
    dataset = _load_dataset_with_retry(split=split)

    processed_data: list[dict] = []
    max_n = sample_size if sample_size is not None and sample_size > 0 else None

    for i, item in enumerate(dataset):
        if max_n is not None and i >= max_n:
            break

        question = str(item.get("question", "")).strip()
        if not question:
            continue

        answer_obj = item.get("answer", {}) if isinstance(item.get("answer"), dict) else {}
        aliases = answer_obj.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        gold_answers = [str(a).strip() for a in aliases if str(a).strip()]
        if not gold_answers:
            value = str(answer_obj.get("value", "")).strip()
            if value:
                gold_answers = [value]

        if not gold_answers:
            continue

        processed_data.append(
            {
                "question": question,
                "answer": gold_answers[0],
                "gold_answers": gold_answers,
                "supporting_facts": [],
            }
        )

    return processed_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Load TriviaQA and convert to Nexus-compatible JSON")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split, e.g. validation/train")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum number of samples")
    parser.add_argument("--output", type=str, default="./data/triviaqa_dev.json", help="Output JSON path")
    args = parser.parse_args()

    sample_size = max(1, int(args.limit))
    data = load_and_convert_triviaqa(split=args.split, sample_size=sample_size)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(data)} samples to {output_path}")


if __name__ == "__main__":
    main()
