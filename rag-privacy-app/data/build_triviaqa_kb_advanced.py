#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np
from datasets import DownloadConfig, load_dataset
from sentence_transformers import SentenceTransformer


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "\n".join(parts).strip()
    return str(value).strip()


def _iter_entity_pages(sample: dict[str, Any]) -> list[tuple[str, str]]:
    entity_pages = sample.get("entity_pages")
    out: list[tuple[str, str]] = []

    if isinstance(entity_pages, dict):
        titles = entity_pages.get("title") or entity_pages.get("titles") or []
        texts = entity_pages.get("text") or entity_pages.get("wiki_context") or entity_pages.get("content") or []

        if isinstance(titles, list) and isinstance(texts, list):
            n = max(len(titles), len(texts))
            for i in range(n):
                title = str(titles[i]).strip() if i < len(titles) else f"entity_{i}"
                text = _flatten_text(texts[i]) if i < len(texts) else ""
                if text:
                    out.append((title, text))
        elif isinstance(texts, list):
            for i, t in enumerate(texts):
                text = _flatten_text(t)
                if text:
                    out.append((f"entity_{i}", text))
        else:
            text = _flatten_text(texts)
            if text:
                out.append((str(entity_pages.get("title", "entity")).strip() or "entity", text))

    elif isinstance(entity_pages, list):
        for i, page in enumerate(entity_pages):
            if isinstance(page, dict):
                title = str(page.get("title", "")).strip() or f"entity_{i}"
                text = _flatten_text(page.get("text") or page.get("wiki_context") or page.get("content"))
            else:
                title = f"entity_{i}"
                text = _flatten_text(page)
            if text:
                out.append((title, text))

    return out


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[tuple[str, int, int]]:
    value = (text or "").strip()
    if not value:
        return []

    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= chunk_size:
        return [(value, 0, len(value))]

    chunks: list[tuple[str, int, int]] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(value):
        end = min(len(value), start + chunk_size)
        chunk = value[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        if end >= len(value):
            break
        start += step
    return chunks


def _load_triviaqa_stream_with_retry(
    retries: int = 5,
    backoff_seconds: float = 2.0,
):
    configs = ["unfiltered", "rc"]
    last_error: Exception | None = None

    for config_name in configs:
        print(f"  trying config: {config_name}")
        for attempt in range(1, retries + 1):
            try:
                return load_dataset(
                    "trivia_qa",
                    config_name,
                    split="validation",
                    streaming=True,
                    download_config=DownloadConfig(max_retries=5),
                )
            except Exception as exc:  # pragma: no cover - network failures are environment-dependent
                last_error = exc
                if attempt >= retries:
                    break

                wait_s = min(30.0, backoff_seconds * (2 ** (attempt - 1)))
                print(
                    f"  load_dataset({config_name}) attempt {attempt}/{retries} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                print(f"  retrying in {wait_s:.1f}s ...")
                time.sleep(wait_s)

        print(f"  config {config_name} exhausted, trying next fallback...")

    raise SystemExit(
        "Failed to load trivia_qa (tried configs: unfiltered -> rc) "
        f"after retries: {last_error}"
    )


def build_kb(
    output_dir: Path,
    max_docs: int,
    chunk_size: int,
    overlap: int,
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading trivia_qa (unfiltered/validation) ...")
    ds = _load_triviaqa_stream_with_retry()

    print("[2/5] Extracting and chunking entity_pages ...")
    docs: list[str] = []
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sample_idx, sample in enumerate(ds):
        pages = _iter_entity_pages(sample)
        for page_idx, (title, text) in enumerate(pages):
            for chunk_idx, (chunk, start, end) in enumerate(_chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
                dedup_key = hashlib.sha256(f"{title}::{chunk}".encode("utf-8")).hexdigest()
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                docs.append(chunk)
                metadata.append(
                    {
                        "title": title,
                        "source": "trivia_qa_unfiltered",
                        "sample_idx": int(sample_idx),
                        "page_idx": int(page_idx),
                        "chunk_idx": int(chunk_idx),
                        "start_char": int(start),
                        "end_char": int(end),
                    }
                )

                if len(docs) % 500 == 0:
                    print(f"  collected chunks: {len(docs)}")

                if len(docs) >= max_docs:
                    break
            if len(docs) >= max_docs:
                break
        if len(docs) >= max_docs:
            break

    if not docs:
        raise SystemExit("No chunks collected from TriviaQA entity_pages")

    print(f"Collected chunks: {len(docs)}")
    print("[3/5] Loading embedding model sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("[4/5] Encoding chunks ...")
    emb = model.encode(
        docs,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("[5/5] Building FAISS (IndexFlatIP) and saving ...")
    dim = int(emb.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    (output_dir / "documents.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": "trivia_qa/unfiltered/validation",
                "chunk_size": int(chunk_size),
                "overlap": int(overlap),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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
    print(f"Output dir: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build advanced TriviaQA KB with chunking + FAISS")
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录. 默认 = 项目根/triviaqa_kb",
    )
    args = parser.parse_args()

    # 默认输出到项目根目录下的 triviaqa_kb/
    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "triviaqa_kb"
    else:
        output_dir = Path(args.output_dir).resolve()

    build_kb(
        output_dir=output_dir,
        max_docs=max(1, int(args.max_docs)),
        chunk_size=max(64, int(args.chunk_size)),
        overlap=max(0, int(args.overlap)),
        batch_size=max(1, int(args.batch_size)),
    )


if __name__ == "__main__":
    main()
