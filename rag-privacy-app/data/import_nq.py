#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_question(question_field: Any) -> str:
    if isinstance(question_field, str):
        return question_field.strip()
    if isinstance(question_field, dict):
        text = question_field.get("text", "")
        return str(text).strip()
    return ""


def _extract_short_answer_text(sample: dict[str, Any]) -> str:
    short_answer = sample.get("short_answer", {})
    if isinstance(short_answer, dict):
        text = short_answer.get("text", "")
        if isinstance(text, list):
            merged = " ".join(str(x).strip() for x in text if str(x).strip())
            if merged:
                return merged
        elif isinstance(text, str) and text.strip():
            return text.strip()

    short_answers = sample.get("short_answers")
    if isinstance(short_answers, list):
        parts: list[str] = []
        for item in short_answers:
            if isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, list):
                    parts.extend(str(x).strip() for x in text if str(x).strip())
                elif isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        merged = " ".join(parts).strip()
        if merged:
            return merged

    return ""


def _extract_long_answer_text(sample: dict[str, Any]) -> str:
    long_answer_text = sample.get("long_answer_text", "")
    if isinstance(long_answer_text, str) and long_answer_text.strip():
        return long_answer_text.strip()

    long_answer = sample.get("long_answer", {})
    if isinstance(long_answer, dict):
        text = long_answer.get("text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()

    annotations = sample.get("annotations")
    document = sample.get("document")
    if isinstance(annotations, list) and annotations and isinstance(document, dict):
        tokens = document.get("tokens")
        token_values: list[str] = []

        if isinstance(tokens, dict):
            token_field = tokens.get("token")
            if isinstance(token_field, list):
                token_values = [str(t) for t in token_field]
        elif isinstance(tokens, list):
            token_values = [str(t) for t in tokens]

        if token_values:
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                la = ann.get("long_answer")
                if not isinstance(la, dict):
                    continue

                start = la.get("start_token", -1)
                end = la.get("end_token", -1)
                if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(token_values):
                    span = " ".join(token_values[start:end]).strip()
                    if span:
                        return span

    return ""


def build_nq_records(limit: int) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing dependency: datasets. Install with: pip install datasets") from exc

    ds = load_dataset("natural_questions", split="validation")

    records: list[dict[str, str]] = []
    for sample in ds:
        question = _normalize_question(sample.get("question"))
        if not question:
            continue

        answer = _extract_long_answer_text(sample)
        if not answer:
            answer = _extract_short_answer_text(sample)
        if not answer:
            continue

        records.append({"question": question, "answer": answer})
        if len(records) >= limit:
            break

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Natural Questions validation split to Nexus JSON format")
    parser.add_argument("--limit", type=int, default=2000, help="Max number of NQ samples to export")
    parser.add_argument("--output", type=str, default="./data/nq_dev.json", help="Output JSON path")
    args = parser.parse_args()

    limit = max(1, int(args.limit))
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_nq_records(limit=limit)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(records)} samples to: {output_path}")


if __name__ == "__main__":
    main()
