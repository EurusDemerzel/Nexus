from __future__ import annotations

import numpy as np

from app.services.embedding_service import embed_text


LOCAL_NOTES = [
    {"id": 1, "content": "我的机器学习学习计划：每周完成两节课程并复盘实验结果。"},
    {"id": 2, "content": "健身记录：周一和周四晚上慢跑 5 公里。"},
    {"id": 3, "content": "阅读清单：本月阅读《深入理解计算机系统》和《Python Cookbook》。"},
    {"id": 4, "content": "工作安排：周三提交隐私计算 MVP 的阶段报告。"},
    {"id": 5, "content": "旅行计划：五一期间去杭州，重点看西湖和灵隐寺。"},
]

_NOTE_EMBEDDINGS = None


def _build_note_embeddings() -> np.ndarray:
    vectors = [embed_text(note["content"]) for note in LOCAL_NOTES]
    return np.asarray(vectors, dtype=float)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def retrieve_local(query: str, top_k: int = 3) -> list[dict]:
    global _NOTE_EMBEDDINGS

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

        scored.append({"id": note["id"], "content": note["content"], "score": round(score, 6)})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
