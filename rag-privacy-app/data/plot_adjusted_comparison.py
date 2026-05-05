#!/usr/bin/env python3
from __future__ import annotations

# MOD: 调整后对比图脚本（Token-F1箱线图、延迟箱线图、延迟vsToken-F1散点图）
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_one(path: Path, default_mode: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    df = pd.read_csv(path)
    if "mode" not in df.columns:
        df["mode"] = default_mode
    return df


def load_pair(static_csv: Path, dynamic_csv: Path) -> pd.DataFrame:
    static_df = _load_one(static_csv, "static-split")
    dynamic_df = _load_one(dynamic_csv, "nexus-dynamic")
    df = pd.concat([static_df, dynamic_df], ignore_index=True)

    required_cols = ["mode", "latency_ms", "token_f1"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    return df


def plot_token_f1_box(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="mode", y="token_f1")
    plt.title("Token-F1 Comparison (Adjusted Setting)")
    plt.xlabel("Mode")
    plt.ylabel("Token-F1")
    plt.tight_layout()
    plt.savefig(out_dir / "adjusted_token_f1_box.png", dpi=200)
    plt.close()


def plot_latency_box(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="mode", y="latency_ms")
    plt.title("Latency Comparison (Adjusted Setting)")
    plt.xlabel("Mode")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(out_dir / "adjusted_latency_box.png", dpi=200)
    plt.close()


def plot_latency_vs_f1(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(9, 6))

    # MOD: 若存在负载分数列则用于标注，否则普通散点
    if "load_score" in df.columns:
        scatter = sns.scatterplot(
            data=df,
            x="latency_ms",
            y="token_f1",
            hue="mode",
            size="load_score",
            sizes=(40, 220),
            alpha=0.8,
        )
        scatter.set_title("Latency vs Token-F1 (marker size by load_score)")
    else:
        scatter = sns.scatterplot(
            data=df,
            x="latency_ms",
            y="token_f1",
            hue="mode",
            alpha=0.8,
        )
        scatter.set_title("Latency vs Token-F1")

    scatter.set_xlabel("Latency (ms)")
    scatter.set_ylabel("Token-F1")
    plt.tight_layout()
    plt.savefig(out_dir / "adjusted_latency_vs_token_f1_scatter.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot adjusted static vs dynamic comparison")
    parser.add_argument("--static", type=str, default="results_static_20_new.csv")
    parser.add_argument("--dynamic", type=str, default="results_dynamic_20_new.csv")
    parser.add_argument("--output-dir", type=str, default="./data/plots")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    static_csv = Path(args.static).resolve()
    dynamic_csv = Path(args.dynamic).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pair(static_csv, dynamic_csv)

    plot_token_f1_box(df, out_dir)
    plot_latency_box(df, out_dir)
    plot_latency_vs_f1(df, out_dir)

    print(f"图表已输出到: {out_dir}")


if __name__ == "__main__":
    main()
