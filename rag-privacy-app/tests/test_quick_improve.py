import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluator import token_f1
from nexus_system import NexusSystem


def _safe_float(value: str, default: float = -1.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_samples(csv_path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("question") and row.get("gold_answer"):
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick quality regression check with improved prompt/decoding")
    parser.add_argument("--csv", type=str, default="results_dynamic_50.csv")
    parser.add_argument("--mode", type=str, default="nexus-dynamic")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    csv_path = ROOT_DIR / args.csv
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    samples = load_samples(csv_path=csv_path, limit=max(1, args.limit))
    if not samples:
        raise SystemExit("No valid samples found in CSV.")

    nexus = NexusSystem()

    old_f1_all: list[float] = []
    new_f1_all: list[float] = []

    for i, row in enumerate(samples, start=1):
        question = row.get("question", "")
        gold_answer = row.get("gold_answer", "")
        old_answer = row.get("model_answer", "")

        old_f1_csv = _safe_float(row.get("token_f1", ""), default=-1.0)
        old_f1 = old_f1_csv if old_f1_csv >= 0 else token_f1(old_answer, gold_answer)

        try:
            new_answer, _ = nexus.ask(question, mode=args.mode)
        except Exception as exc:
            new_answer = f"ERROR: {type(exc).__name__}: {exc}"

        new_f1 = token_f1(new_answer, gold_answer) if not new_answer.startswith("ERROR:") else -1.0

        old_f1_all.append(old_f1)
        new_f1_all.append(new_f1)

        print(f"\n[{i}] Question: {question}")
        print(f"Old Answer: {old_answer}")
        print(f"New Answer: {new_answer}")
        print(f"Token-F1: old={old_f1:.4f} | new={new_f1:.4f} | delta={new_f1 - old_f1:+.4f}")

    valid_old = [x for x in old_f1_all if x >= 0]
    valid_new = [x for x in new_f1_all if x >= 0]

    if valid_old and valid_new:
        old_avg = sum(valid_old) / len(valid_old)
        new_avg = sum(valid_new) / len(valid_new)
        print("\n=== Summary ===")
        print(f"Avg old Token-F1: {old_avg:.4f}")
        print(f"Avg new Token-F1: {new_avg:.4f}")
        print(f"Avg delta: {new_avg - old_avg:+.4f}")


if __name__ == "__main__":
    main()
