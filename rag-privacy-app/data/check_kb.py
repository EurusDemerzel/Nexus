import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.local_retrieval import retrieve


def main() -> None:
    kb_file = ROOT_DIR / "data" / "local_notes_store.json"
    if not kb_file.exists():
        raise SystemExit(f"未找到本地知识库文件: {kb_file}")

    payload = json.loads(kb_file.read_text(encoding="utf-8"))
    print(f"知识库文件: {kb_file}")
    print(f"文档总数: {len(payload) if isinstance(payload, list) else 0}")

    test_question = "Who directed The Shawshank Redemption?"
    print(f"\\n测试查询: {test_question}")

    results = retrieve(test_question, top_k=3)
    if not results:
        print("未检索到结果。")
        return

    for i, doc in enumerate(results, start=1):
        metadata = doc.get("metadata", {}) or {}
        title = metadata.get("title", "<no-title>")
        snippet = (doc.get("content", "") or "")[:120].replace("\\n", " ")
        print(f"[{i}] title={title} | {snippet}")


if __name__ == "__main__":
    main()
