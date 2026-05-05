#!/usr/bin/env python3
from __future__ import annotations

# NEW: 高级实验图表脚本（EM柱状图、Token-F1箱线图、延迟-Token-F1散点图、可选负载-切分折线图）
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"找不到结果文件: {p}")
        df = pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_exact_match_bar(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(x="mode", y="exact_match", data=df, estimator="mean", errorbar="sd")
    plt.title("Exact Match by Mode")
    plt.ylabel("Exact Match (mean ± sd)")
    plt.xlabel("Mode")
    plt.tight_layout()
    plt.savefig(output_dir / "exact_match_bar.png", dpi=200)
    plt.close()


def plot_token_f1_box(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="mode", y="token_f1", data=df)
    plt.title("Token-F1 Distribution by Mode")
    plt.ylabel("Token-F1")
    plt.xlabel("Mode")
    plt.tight_layout()
    plt.savefig(output_dir / "token_f1_boxplot.png", dpi=200)
    plt.close()


def plot_latency_vs_f1(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="latency_ms", y="token_f1", hue="mode", data=df, alpha=0.8)
    plt.title("Latency vs Token-F1")
    plt.ylabel("Token-F1")
    plt.xlabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_vs_token_f1_scatter.png", dpi=200)
    plt.close()


def plot_load_vs_localk_optional(df: pd.DataFrame, output_dir: Path) -> None:
    # NEW: 可选图，仅当结果中包含 load_score/local_k 列时绘制
    if "load_score" not in df.columns or "local_k" not in df.columns:
        print("[plot_advanced_results] 跳过 load_score vs local_k（缺少列）")
        return

    dynamic_df = df[df["mode"] == "nexus-dynamic"].copy()
    if dynamic_df.empty:
        print("[plot_advanced_results] 跳过 load_score vs local_k（无 nexus-dynamic 样本）")
        return

    dynamic_df = dynamic_df.sort_values(by="load_score")
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="load_score", y="local_k", data=dynamic_df, marker="o")
    plt.title("Dynamic Policy: Load Score vs local_k")
    plt.ylabel("local_k")
    plt.xlabel("load_score")
    plt.tight_layout()
    plt.savefig(output_dir / "loadscore_vs_localk_line.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot advanced EMNLP experiment figures")
    parser.add_argument("--cloud", type=str, default="results_cloud_50.csv", help="cloud-only results csv")
    parser.add_argument("--static", type=str, default="results_static_50.csv", help="static-split results csv")
    parser.add_argument("--dynamic", type=str, default="results_dynamic_50.csv", help="nexus-dynamic results csv")
    parser.add_argument("--output-dir", type=str, default="./data/plots", help="output directory")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    paths = [Path(args.cloud).resolve(), Path(args.static).resolve(), Path(args.dynamic).resolve()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(paths)

    plot_exact_match_bar(df, output_dir)
    plot_token_f1_box(df, output_dir)
    plot_latency_vs_f1(df, output_dir)
    plot_load_vs_localk_optional(df, output_dir)

    print(f"图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
