#!/usr/bin/env python3
from __future__ import annotations

# NEW: 人工评估抽样脚本（每种模式随机抽样 N 条，输出 judgement 留空）
import argparse
from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件: {path}")
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    # NEW: 统一字段，若缺失 model_answer/gold_answer 也保持可运行
    if "mode" not in df.columns:
        df["mode"] = mode_name
    if "gold_answer" not in df.columns:
        df["gold_answer"] = ""
    if "model_answer" not in df.columns:
        if "response" in df.columns:
            df["model_answer"] = df["response"].fillna("")
        else:
            df["model_answer"] = ""
    return df


def sample_mode(df: pd.DataFrame, sample_n: int, seed: int) -> pd.DataFrame:
    n = min(sample_n, len(df))
    if n <= 0:
        return df.iloc[0:0].copy()
    return df.sample(n=n, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample rows for human evaluation")
    parser.add_argument("--cloud", type=str, default="results_cloud_50.csv")
    parser.add_argument("--static", type=str, default="results_static_50.csv")
    parser.add_argument("--dynamic", type=str, default="results_dynamic_50.csv")
    parser.add_argument("--n-per-mode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="human_eval_template.csv")
    args = parser.parse_args()

    cloud_df = ensure_columns(load_csv(Path(args.cloud).resolve()), "cloud-only")
    static_df = ensure_columns(load_csv(Path(args.static).resolve()), "static-split")
    dynamic_df = ensure_columns(load_csv(Path(args.dynamic).resolve()), "nexus-dynamic")

    sampled = pd.concat(
        [
            sample_mode(cloud_df, args.n_per_mode, args.seed),
            sample_mode(static_df, args.n_per_mode, args.seed + 1),
            sample_mode(dynamic_df, args.n_per_mode, args.seed + 2),
        ],
        ignore_index=True,
    )

    keep_cols = ["question", "gold_answer", "model_answer", "mode"]
    for col in keep_cols:
        if col not in sampled.columns:
            sampled[col] = ""

    sampled = sampled[keep_cols].copy()
    # NEW: 人工标注列（留空）
    sampled["judgement"] = ""

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False, encoding="utf-8")
    print(f"已生成人工评估模板: {output_path}")


if __name__ == "__main__":
    main()
