import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.local_retrieval import add_document


def split_blocks(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n+", text.strip())
    return [block.strip() for block in blocks if block.strip()]


def main() -> None:
    kb_path = Path(__file__).resolve().parent / "knowledge_base.txt"
    if not kb_path.exists():
        print(f"[ERROR] 未找到知识库文件: {kb_path}")
        sys.exit(1)

    raw = kb_path.read_text(encoding="utf-8")
    blocks = split_blocks(raw)

    imported = 0
    skipped_short = 0
    skipped_dup = 0

    for block in blocks:
        if len(block) < 20:
            skipped_short += 1
            continue

        ok = add_document(block, metadata={"source": "manual", "topic": "computer_network"})
        if ok:
            imported += 1
        else:
            skipped_dup += 1

    print("=== 知识库导入完成 ===")
    print(f"文件: {kb_path}")
    print(f"原始块数: {len(blocks)}")
    print(f"成功导入: {imported}")
    print(f"跳过(长度<20): {skipped_short}")
    print(f"跳过(重复): {skipped_dup}")


if __name__ == "__main__":
    main()
