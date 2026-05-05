#!/usr/bin/env python3
from __future__ import annotations

# NEW: 人工评估统计脚本（正确率、部分正确率）
import argparse
from pathlib import Path

import pandas as pd


CORRECT_LABELS = {"正确", "correct", "1", "yes"}
PARTIAL_LABELS = {"部分正确", "partial", "partially_correct", "0.5"}
WRONG_LABELS = {"错误", "wrong", "0", "no"}


def normalize_label(x: str) -> str:
    s = str(x or "").strip().lower()
    if s in CORRECT_LABELS:
        return "correct"
    if s in PARTIAL_LABELS:
        return "partial"
    if s in WRONG_LABELS:
        return "wrong"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute human evaluation stats by mode")
    parser.add_argument("--input", type=str, default="human_eval_template.csv")
    parser.add_argument("--output", type=str, default="human_eval_stats.csv")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"找不到输入文件: {input_path}")

    df = pd.read_csv(input_path)
    if "mode" not in df.columns or "judgement" not in df.columns:
        raise SystemExit("输入 CSV 必须包含 mode 和 judgement 列")

    df["judgement_norm"] = df["judgement"].map(normalize_label)
    df_valid = df[df["judgement_norm"] != "unknown"].copy()

    rows = []
    for mode, g in df_valid.groupby("mode"):
        total = len(g)
        correct = int((g["judgement_norm"] == "correct").sum())
        partial = int((g["judgement_norm"] == "partial").sum())
        wrong = int((g["judgement_norm"] == "wrong").sum())

        correct_rate = correct / total if total else 0.0
        partial_rate = partial / total if total else 0.0
        wrong_rate = wrong / total if total else 0.0

        rows.append(
            {
                "mode": mode,
                "total": total,
                "correct": correct,
                "partial": partial,
                "wrong": wrong,
                "correct_rate": round(correct_rate, 4),
                "partial_rate": round(partial_rate, 4),
                "wrong_rate": round(wrong_rate, 4),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(by="mode")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

    print(out_df.to_string(index=False))
    print(f"统计结果已保存: {output_path}")


if __name__ == "__main__":
    main()
