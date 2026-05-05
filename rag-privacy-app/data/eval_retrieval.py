#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.local_retrieval import retrieve


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_aliases_from_hf(target_questions: set[str], max_scan: int = 50000) -> dict[str, list[str]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return {}

    out: dict[str, list[str]] = {}
    ds = load_dataset("trivia_qa", "rc", split="validation", streaming=True)

    scanned = 0
    for item in ds:
        scanned += 1
        q = _normalize(str(item.get("question", "")))
        if q in target_questions:
            aliases: list[str] = []
            ans = item.get("answer", {}) if isinstance(item.get("answer"), dict) else {}
            raw_aliases = ans.get("aliases", []) if isinstance(ans.get("aliases", []), list) else []
            aliases.extend([str(a).strip() for a in raw_aliases if str(a).strip()])
            val = str(ans.get("value", "")).strip()
            if val:
                aliases.append(val)
            if aliases:
                out[q] = sorted(set(aliases))

        if len(out) >= len(target_questions) or scanned >= max_scan:
            break

    return out


def _load_questions(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        if not q:
            continue

        answer = str(item.get("answer", "")).strip()
        gold_answers = item.get("gold_answers", [])
        aliases: list[str] = []
        if isinstance(gold_answers, list):
            aliases.extend([str(x).strip() for x in gold_answers if str(x).strip()])
        if answer:
            aliases.append(answer)

        rows.append(
            {
                "question": q,
                "norm_question": _normalize(q),
                "aliases": sorted(set(aliases)),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall@k on TriviaQA questions")
    parser.add_argument("--data", type=str, default="./data/triviaqa_dev.json")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--backend", type=str, default="triviaqa-hybrid")
    parser.add_argument("--k", type=str, default="1,5,10")
    parser.add_argument("--use-hf-aliases", action="store_true")
    args = parser.parse_args()

    os.environ["RETRIEVAL_BACKEND"] = args.backend

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"data file not found: {data_path}")

    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    ks = sorted(set([k for k in ks if k > 0]))
    if not ks:
        raise SystemExit("invalid k list")

    rows = _load_questions(data_path, limit=max(1, int(args.limit)))
    if not rows:
        raise SystemExit("no questions loaded")

    if args.use_hf_aliases:
        qset = {r["norm_question"] for r in rows}
        hf_alias_map = _extract_aliases_from_hf(qset)
        for r in rows:
            merged = set(r["aliases"])
            merged.update(hf_alias_map.get(r["norm_question"], []))
            r["aliases"] = sorted(a for a in merged if a)

    hit_counts = {k: 0 for k in ks}
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        q = row["question"]
        aliases = [_normalize(a) for a in row["aliases"] if a]
        if not aliases:
            continue

        docs = retrieve(q, top_k=max(ks))
        merged_docs = []
        for d in docs:
            title = str((d.get("metadata") or {}).get("title", ""))
            content = str(d.get("content", ""))
            merged_docs.append(_normalize(title + " " + content))

        for k in ks:
            top_docs = merged_docs[:k]
            hit = any(any(alias and alias in doc for alias in aliases) for doc in top_docs)
            if hit:
                hit_counts[k] += 1

        if i % 10 == 0:
            print(f"processed {i}/{total}")

    print("=== Retrieval Recall@k ===")
    for k in ks:
        recall = hit_counts[k] / total if total else 0.0
        print(f"Recall@{k}: {recall:.4f} ({hit_counts[k]}/{total})")


if __name__ == "__main__":
    main()
