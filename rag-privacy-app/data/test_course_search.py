import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.app import create_app
from app.services.retrieval_service import search_courses


def main() -> None:
    app = create_app()
    with app.app_context():
        for q in ["高等数学 期末考试", "数据结构 课程设计"]:
            hits = search_courses(q, top_k=3)
            print(f"\nQuery: {q}")
            print(json.dumps(hits, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
