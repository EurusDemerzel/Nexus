#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return " ".join(parts).strip()
    return str(value).strip()


def _extract_entity_docs(sample: dict[str, Any]) -> list[tuple[str, str]]:
    entity_pages = sample.get("entity_pages")
    docs: list[tuple[str, str]] = []

    if isinstance(entity_pages, dict):
        titles = entity_pages.get("title") or entity_pages.get("titles") or []
        texts = (
            entity_pages.get("text")
            or entity_pages.get("wiki_context")
            or entity_pages.get("wiki_contexts")
            or entity_pages.get("content")
            or []
        )

        if isinstance(titles, list) and isinstance(texts, list):
            n = max(len(titles), len(texts))
            for i in range(n):
                title = str(titles[i]).strip() if i < len(titles) else ""
                text = _flatten_text(texts[i]) if i < len(texts) else ""
                if text:
                    docs.append((title or f"entity_{i}", text))
        elif isinstance(texts, list):
            for i, t in enumerate(texts):
                text = _flatten_text(t)
                if text:
                    docs.append((f"entity_{i}", text))
        else:
            text = _flatten_text(texts)
            if text:
                title = str(entity_pages.get("title", "")).strip() or "entity"
                docs.append((title, text))

        pages = entity_pages.get("pages")
        if isinstance(pages, list):
            for i, p in enumerate(pages):
                if not isinstance(p, dict):
                    continue
                title = str(p.get("title", "")).strip() or f"entity_page_{i}"
                text = _flatten_text(p.get("text") or p.get("content") or p.get("wiki_context"))
                if text:
                    docs.append((title, text))

    elif isinstance(entity_pages, list):
        for i, p in enumerate(entity_pages):
            if isinstance(p, dict):
                title = str(p.get("title", "")).strip() or f"entity_{i}"
                text = _flatten_text(p.get("text") or p.get("content") or p.get("wiki_context"))
            else:
                title = f"entity_{i}"
                text = _flatten_text(p)
            if text:
                docs.append((title, text))

    return docs


def build_triviaqa_kb(max_docs: int, output_dir: Path, batch_size: int = 64) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print("[1/5] Loading TriviaQA unfiltered validation split...")
    ds = load_dataset("trivia_qa", "unfiltered", split="validation", streaming=True)

    print("[2/5] Collecting entity pages...")
    docs: list[str] = []
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sample_idx, sample in enumerate(ds):
        for title, text in _extract_entity_docs(sample):
            key = hashlib.sha256(f"{title}::{text}".encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)

            docs.append(text)
            metadata.append({"title": title, "source": "trivia_qa_unfiltered", "sample_idx": int(sample_idx)})

            if len(docs) >= max_docs:
                break
        if len(docs) >= max_docs:
            break

    if not docs:
        raise SystemExit("No entity page documents extracted from TriviaQA unfiltered validation split.")

    print(f"Collected documents: {len(docs)}")

    print("[3/5] Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("[4/5] Encoding documents...")
    embeddings = model.encode(
        docs,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("[5/5] Building and saving FAISS index...")
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = output_dir / "index.faiss"
    docs_path = output_dir / "documents.json"
    meta_path = output_dir / "metadata.json"
    config_path = output_dir / "config.json"

    faiss.write_index(index, str(index_path))
    docs_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "dataset": "trivia_qa/unfiltered/validation",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "metric": "inner_product",
                "dimension": dim,
                "doc_count": len(docs),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Index: {index_path}")
    print(f"Documents: {docs_path}")
    print(f"Metadata: {meta_path}")
    print(f"Config: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TriviaQA FAISS KB from unfiltered entity_pages")
    parser.add_argument("--max-docs", type=int, default=5000, help="Maximum number of documents")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output KB directory. 默认 = 项目根/triviaqa_kb")
    args = parser.parse_args()

    max_docs = max(1, int(args.max_docs))
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (Path(__file__).resolve().parents[1] / "triviaqa_kb")
    build_triviaqa_kb(max_docs=max_docs, output_dir=output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
