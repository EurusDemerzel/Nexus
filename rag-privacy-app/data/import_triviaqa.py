#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _load_validation_dataset_only():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise SystemExit("缺少依赖 datasets，请先执行: pip install datasets") from exc

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise SystemExit("缺少依赖 huggingface_hub，请先执行: pip install huggingface_hub") from exc

    # 优先仅拉取 validation parquet，避免下载 train 分片。
    snapshot_dir = snapshot_download(
        repo_id="trivia_qa",
        repo_type="dataset",
        allow_patterns=["rc/validation-*.parquet", "rc/dataset_info.json", "README.md", "**/README.md"],
        ignore_patterns=["**/train-*.parquet", "**/test-*.parquet"],
    )

    snap = Path(snapshot_dir)
    val_files = sorted(snap.glob("rc/validation-*.parquet"))
    if not val_files:
        val_files = sorted(snap.glob("**/validation-*.parquet"))
    if not val_files:
        raise SystemExit("未找到 TriviaQA validation parquet 文件")

    ds = load_dataset(
        "parquet",
        data_files={"validation": [str(p) for p in val_files]},
        split="validation",
    )
    return ds


def build_triviaqa_dev(limit: int) -> list[dict[str, str]]:
    # 可选：如需镜像加速，可在运行前设置环境变量
    # export HF_ENDPOINT=https://hf-mirror.com
    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    hf_cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "trivia_qa"
    if hf_cache_root.exists():
        for bad_dir in hf_cache_root.rglob("*.incomplete"):
            shutil.rmtree(bad_dir, ignore_errors=True)

    ds = _load_validation_dataset_only()

    rows: list[dict[str, str]] = []
    for sample in ds:
        question = str(sample.get("question", "")).strip()
        if not question:
            continue

        answer_obj = sample.get("answer", {})
        aliases = answer_obj.get("aliases", []) if isinstance(answer_obj, dict) else []
        if not isinstance(aliases, list) or not aliases:
            continue

        answer = str(aliases[0]).strip()
        if not answer:
            continue

        rows.append({"question": question, "answer": answer})
        if len(rows) >= limit:
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import TriviaQA (rc/validation) to Nexus JSON format")
    parser.add_argument("--limit", type=int, default=2000, help="导出样本上限")
    parser.add_argument("--output", type=str, default="./data/triviaqa_dev.json", help="输出 JSON 路径")
    args = parser.parse_args()

    limit = max(1, int(args.limit))
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_triviaqa_dev(limit=limit)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已导出 {len(data)} 条到: {output_path}")


if __name__ == "__main__":
    main()
