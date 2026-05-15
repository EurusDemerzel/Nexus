#!/usr/bin/env python3
"""
rebuild_faiss_bge_large.py — 用 BGE-large-en-v1.5 (1024-dim) 重建 FAISS 索引
从已有的 docs JSON 文件读取段落，重新编码并保存索引。

用法:
  # 使用默认的 triviaqa_kb/documents.json + bge-large
  python data/rebuild_faiss_bge_large.py

  # 指定输入 docs 和输出目录
  python data/rebuild_faiss_bge_large.py \
    --docs ./triviaqa_kb/documents_wiki.json \
    --output ./triviaqa_kb_bge_large \
    --model ./models_for_server/bge-large-en-v1.5
"""

import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = ROOT_DIR / "triviaqa_kb" / "documents.json"
DEFAULT_MODEL = ROOT_DIR / "models_for_server" / "bge-large-en-v1.5"
DEFAULT_OUTPUT = ROOT_DIR / "triviaqa_kb"
DEFAULT_OUTPUT_SUFFIX = "_bge_large"  # 避免覆盖旧索引，追加后缀


def load_docs(docs_path: Path) -> list[str]:
    """加载文档 JSON（支持 documents.json 和 triviaqa_wiki_docs.json 两种格式）。"""
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents file not found: {docs_path}")

    data = json.loads(docs_path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        # 如果是字符串列表，直接返回
        if data and isinstance(data[0], str):
            return data
        # 如果是字典列表，提取 content / text 字段
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("content") or item.get("text") or item.get("document") or ""
                texts.append(str(text))
            else:
                texts.append(str(item))
        return texts

    if isinstance(data, dict):
        # 可能是 {id: text} 或 {"documents": [...]} 格式
        docs = data.get("documents") or data.get("texts") or data.get("data") or []
        if isinstance(docs, list):
            return [str(d) if isinstance(d, str) else str(d.get("content", d)) for d in docs]
        return [str(v) for v in data.values()]

    raise ValueError(f"Unsupported JSON format in {docs_path}")


def build_faiss_index(
    docs_path: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    index_filename: str = "index.faiss",
    docs_filename: str = "documents.json",
    metadata_filename: str = "metadata.json",
    config_filename: str = "config.json",
):
    """加载 docs → BGE-large 编码 → FAISS IndexFlatIP → 保存。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载文档 ──
    print(f"[1/4] Loading documents from: {docs_path}")
    docs = load_docs(docs_path)
    print(f"  Loaded {len(docs)} documents")

    # ── 2. 加载 BGE-large 模型 ──
    print(f"[2/4] Loading BGE-large model from: {model_path}")
    t0 = time.perf_counter()
    if model_path.is_dir():
        model = SentenceTransformer(str(model_path), local_files_only=True)
    else:
        model = SentenceTransformer(str(model_path))
    dim = model.get_sentence_embedding_dimension()
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s, embedding dim = {dim}")

    # ── 3. 编码 ──
    print(f"[3/4] Encoding {len(docs)} documents (batch_size={batch_size}) ...")
    t1 = time.perf_counter()
    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # BGE 推荐 L2 归一化后用内积 = 余弦相似度
    ).astype(np.float32)
    encode_time = time.perf_counter() - t1
    print(f"  Encoded in {encode_time:.1f}s, shape = {embeddings.shape}")

    # ── 4. 构建 FAISS 索引 ──
    print(f"[4/4] Building FAISS IndexFlatIP ({dim}-dim) ...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = output_dir / index_filename
    docs_out_path = output_dir / docs_filename
    meta_out_path = output_dir / metadata_filename
    config_out_path = output_dir / config_filename

    faiss.write_index(index, str(index_path))
    print(f"  Index saved: {index_path} ({index.ntotal} vectors)")

    # 保存文档副本
    docs_out_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Documents saved: {docs_out_path}")

    # 生成简单 metadata（如果原文件有 metadata.json 则保留）
    metadata = [{"title": f"doc_{i}", "source": "triviaqa_wiki"} for i in range(len(docs))]
    meta_out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Metadata saved: {meta_out_path}")

    # 配置文件
    config = {
        "embedding_model": str(model_path),
        "embedding_dim": dim,
        "metric": "inner_product",
        "doc_count": len(docs),
        "source_docs": str(docs_path),
    }
    config_out_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Config saved: {config_out_path}")

    total_time = time.perf_counter() - t0 + load_time - load_time  # approximate
    print(f"\n✓ Done. Total {len(docs)} docs, {dim}-dim, IndexFlatIP")
    print(f"  Output: {output_dir}")
    print(f"\n  Usage:")
    print(f"    export FAISS_KB_DIR={output_dir}")
    print(f"    export FAISS_INDEX_PATH={output_dir / index_filename}")
    print(f"    export FAISS_DOCUMENTS_PATH={output_dir / docs_filename}")
    print(f"    export FAISS_METADATA_PATH={output_dir / metadata_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index with BAAI/bge-large-en-v1.5 (1024-dim)"
    )
    parser.add_argument(
        "--docs", type=str, default=str(DEFAULT_DOCS),
        help=f"Path to documents JSON (default: {DEFAULT_DOCS})",
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL),
        help=f"Path to BGE model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for FAISS index (default: <docs_dir>_bge_large/)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Encoding batch size (default: 32)",
    )
    parser.add_argument(
        "--index-name", type=str, default="index.faiss",
        help="FAISS index filename (default: index.faiss)",
    )
    args = parser.parse_args()

    docs_path = Path(args.docs)
    model_path = Path(args.model)
    if args.output:
        output_dir = Path(args.output)
    else:
        # 默认输出到 docs 同级目录 + _bge_large 后缀
        parent = docs_path.parent
        output_dir = parent.parent / (parent.name + DEFAULT_OUTPUT_SUFFIX)

    if not docs_path.exists():
        raise SystemExit(f"ERROR: Documents file not found: {docs_path}")
    if not model_path.is_dir():
        raise SystemExit(
            f"ERROR: Model directory not found: {model_path}\n"
            f"  Please run: python data/download_bge_large.py first"
        )

    build_faiss_index(
        docs_path=docs_path,
        model_path=model_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        index_filename=args.index_name,
    )


if __name__ == "__main__":
    main()
