import json

import requests


def check(question: str) -> dict:
    resp = requests.post(
        "http://127.0.0.1:5000/ask",
        json={"question": question, "user_id": 1, "debug": True, "source": "auto"},
        timeout=180,
    )
    data = resp.json()
    debug = data.get("debug", {})
    chunks = debug.get("retrieved_chunks") or []
    first_title = None
    if chunks and isinstance(chunks[0], dict):
        first_title = (chunks[0].get("metadata") or {}).get("title")
    return {
        "status": resp.status_code,
        "question": question,
        "source_used": debug.get("source_used"),
        "retrieved_count": len(chunks),
        "first_title": first_title,
    }


if __name__ == "__main__":
    print(json.dumps(check("高等数学 期末考试"), ensure_ascii=False))
    print(json.dumps(check("一种新的专利技术"), ensure_ascii=False))
