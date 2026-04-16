from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from app.services.embedding_service import embed_text


DEFAULT_LOCAL_NOTES = [
    {"id": 1, "content": "我的机器学习学习计划：每周完成两节课程并复盘实验结果。"},
    {"id": 2, "content": "健身记录：周一和周四晚上慢跑 5 公里。"},
    {"id": 3, "content": "阅读清单：本月阅读《深入理解计算机系统》和《Python Cookbook》。"},
    {"id": 4, "content": "工作安排：周三提交隐私计算 MVP 的阶段报告。"},
    {"id": 5, "content": "旅行计划：五一期间去杭州，重点看西湖和灵隐寺。"},
]

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
NOTES_FILE = DATA_DIR / "local_notes_store.json"

LOCAL_NOTES: list[dict] = []
_NOTE_EMBEDDINGS = None


def _load_notes() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if NOTES_FILE.exists():
        try:
            payload = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                if not payload:
                    return []
                valid = [x for x in payload if isinstance(x, dict) and x.get("content")]
                return valid
        except Exception:
            pass

    seed = [dict(item) for item in DEFAULT_LOCAL_NOTES]
    _save_notes(seed)
    return seed


def _save_notes(notes: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_notes_loaded() -> None:
    global LOCAL_NOTES
    if not LOCAL_NOTES:
        LOCAL_NOTES = _load_notes()


def _build_note_embeddings() -> np.ndarray:
    _ensure_notes_loaded()
    vectors = [embed_text(note["content"]) for note in LOCAL_NOTES]
    return np.asarray(vectors, dtype=float)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def add_document(text: str, metadata: dict | None = None) -> bool:
    global _NOTE_EMBEDDINGS

    content = (text or "").strip()
    if len(content) < 20:
        return False

    _ensure_notes_loaded()

    incoming_hash = _content_hash(content)

    # De-duplicate by content hash and exact text for robustness.
    for note in LOCAL_NOTES:
        existed_text = (note.get("content") or "").strip()
        existed_hash = note.get("content_hash") or _content_hash(existed_text)
        if existed_hash == incoming_hash or existed_text == content:
            return False

    max_id = max((int(note.get("id", 0)) for note in LOCAL_NOTES), default=0)
    LOCAL_NOTES.append(
        {
            "id": max_id + 1,
            "content": content,
            "content_hash": incoming_hash,
            "metadata": metadata or {},
        }
    )
    _save_notes(LOCAL_NOTES)
    _NOTE_EMBEDDINGS = None
    return True


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    global _NOTE_EMBEDDINGS

    _ensure_notes_loaded()
    top_k = max(1, int(top_k))
    if _NOTE_EMBEDDINGS is None:
        _NOTE_EMBEDDINGS = _build_note_embeddings()

    query_vec = np.asarray(embed_text(query), dtype=float)
    scored: list[dict] = []

    for idx, note in enumerate(LOCAL_NOTES):
        score = _cosine_similarity(query_vec, _NOTE_EMBEDDINGS[idx])

        # Add a small lexical prior so obvious intent matches are stable in demo.
        if "学习" in query and "机器学习" in note["content"]:
            score += 0.05

        scored.append(
            {
                "id": note["id"],
                "content": note["content"],
                "metadata": note.get("metadata") or {},
                "score": round(score, 6),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    results = scored[:top_k]

    preview = ""
    if results:
        preview = (results[0].get("content") or "")[:200].replace("\n", " ")
    print(f"[local_retrieval] hits={len(results)} preview={preview}")

    return results


def retrieve_local(query: str, top_k: int = 3) -> list[dict]:
    return retrieve(query, top_k=top_k)
