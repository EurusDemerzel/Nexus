import json
import re
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from app.models import MaterialEmbedding, Patent
from app.services.embedding_service import embed_text


DEFAULT_RETRIEVAL_CONFIG: Dict[str, int] = {
    "candidate_limit": 50,
    "top_doc_limit": 10,
    "chunk_limit": 5,
    "min_chunk_score": 1,
}

COURSE_QUERY_KEYWORDS = {
    "课程",
    "教材",
    "考试",
    "期末",
    "期中",
    "作业",
    "实验",
    "课程设计",
    "高等数学",
    "数据结构",
}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Keep English words and individual Chinese characters for lightweight matching.
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())


def _score_document(query: str, title: str, content: str) -> int:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0

    t_tokens = _tokenize(title)
    c_tokens = _tokenize(content)
    t_set = set(t_tokens)
    c_set = set(c_tokens)

    overlap_title = sum(1 for tk in q_tokens if tk in t_set)
    overlap_content = sum(1 for tk in q_tokens if tk in c_set)
    phrase_bonus = 5 if query and query in (content or "") else 0

    # Hybrid score: title hit is stronger than content hit.
    return overlap_title * 3 + overlap_content + phrase_bonus


def _dynamic_split_text(text: str, query: str, max_chunk: Optional[int] = None) -> List[str]:
    """Dynamically split content into semantically smaller chunks.

    - If query is short, keep shorter chunks for precision.
    - If query is long, keep slightly larger chunks for context integrity.
    """
    if not text:
        return []

    query_len = len(query or "")
    if max_chunk is None:
        max_chunk = 220 if query_len <= 8 else 320 if query_len <= 20 else 420

    # Split by Chinese/English punctuation first.
    sentences = [s.strip() for s in re.split(r"[。！？!?；;\n]+", text) if s.strip()]

    chunks: List[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= max_chunk:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s

    if buf:
        chunks.append(buf)

    # Fallback for texts with no punctuation.
    if not chunks:
        return [text[i : i + max_chunk] for i in range(0, len(text), max_chunk)]

    return chunks

def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_config(retrieval_config: Optional[Dict[str, int]]) -> Dict[str, int]:
    if not retrieval_config:
        return dict(DEFAULT_RETRIEVAL_CONFIG)

    resolved = dict(DEFAULT_RETRIEVAL_CONFIG)
    for key, default in DEFAULT_RETRIEVAL_CONFIG.items():
        resolved[key] = _safe_int(retrieval_config.get(key), default)

    if "max_chunk" in retrieval_config:
        resolved["max_chunk"] = _safe_int(retrieval_config.get("max_chunk"), 320)

    return resolved


def _is_course_query(query: str) -> bool:
    text = (query or "").lower()
    return any(keyword in text for keyword in COURSE_QUERY_KEYWORDS)


def choose_source(query: str, source: str = "auto") -> str:
    source_value = (source or "auto").strip().lower()
    if source_value not in {"auto", "patents", "courses"}:
        source_value = "auto"

    if source_value == "auto":
        return "courses" if _is_course_query(query) else "patents"

    return source_value


def _is_valid_material_content(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    if text.startswith("[文件:"):
        return False
    return len(text) > 50


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def search_patents(query: str, retrieval_config: Optional[Dict[str, int]] = None) -> List[Dict[str, object]]:
    if not query:
        return []

    config = _resolve_config(retrieval_config)

    # Broad candidate set, then rank in Python with a deterministic score.
    search_query = f"%{query}%"
    candidates = Patent.query.filter(
        (Patent.content.like(search_query)) | (Patent.title.like(search_query))
    ).limit(config["candidate_limit"]).all()

    ranked_docs = sorted(
        candidates,
        key=lambda p: _score_document(query, p.title or "", p.content or ""),
        reverse=True,
    )

    top_docs = ranked_docs[: config["top_doc_limit"]]

    chunk_candidates: List[Dict[str, object]] = []
    for p in top_docs:
        chunks = _dynamic_split_text(p.content or "", query, config.get("max_chunk"))
        for idx, chunk in enumerate(chunks):
            score = _score_document(query, p.title or "", chunk)
            if score >= config["min_chunk_score"]:
                chunk_candidates.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "title": p.title,
                            "chunk_index": idx,
                            "patent_id": p.id,
                        },
                        "score": float(score),
                        "source": "patents",
                    }
                )

    chunk_candidates.sort(key=lambda x: float(x["score"]), reverse=True)
    if chunk_candidates:
        return chunk_candidates[: config["chunk_limit"]]

    # Fallback: return coarse patent chunks to keep patent source usable even on sparse lexical matches.
    fallback_docs = top_docs
    if not fallback_docs:
        fallback_docs = Patent.query.order_by(Patent.id.desc()).limit(config["top_doc_limit"]).all()

    fallback_chunks: List[Dict[str, object]] = []
    for p in fallback_docs[: config["chunk_limit"]]:
        text = (p.content or "").strip()
        if not text:
            continue
        fallback_chunks.append(
            {
                "content": text[:320],
                "metadata": {
                    "title": p.title,
                    "chunk_index": 0,
                    "patent_id": p.id,
                },
                "score": 0.0,
                "source": "patents",
            }
        )

    return fallback_chunks


def search_courses(query: str, top_k: int = 5) -> List[Dict[str, object]]:
    if not query:
        return []

    query_vec = np.asarray(embed_text(query), dtype=float)
    if query_vec.size == 0:
        return []

    try:
        rows = MaterialEmbedding.query.all()
    except SQLAlchemyError:
        return []

    scored_rows: List[Dict[str, object]] = []

    for row in rows:
        material = row.material
        if material is None:
            continue
        if not _is_valid_material_content(material.content or ""):
            continue

        try:
            material_vec = np.asarray(json.loads(row.embedding_json or "[]"), dtype=float)
        except Exception:
            continue

        if material_vec.size == 0:
            continue

        similarity = _cosine_similarity(query_vec, material_vec)
        lexical_score = _score_document(
            query,
            f"{material.title or ''} {(material.course.name if material.course else '')}",
            material.content or "",
        )
        score = similarity + lexical_score * 0.01
        if score <= 0.0:
            continue

        course_name = material.course.name if material.course else ""
        scored_rows.append(
            {
                "content": material.content,
                "score": round(score, 6),
                "metadata": {
                    "title": material.title,
                    "material_id": material.id,
                    "course_id": material.course_id,
                    "course_name": course_name,
                    "source_repo": material.source_repo,
                    "file_type": material.file_type,
                    "similarity": round(similarity, 6),
                },
                "source": "courses",
            }
        )

    scored_rows.sort(key=lambda x: float(x["score"]), reverse=True)
    return scored_rows[: max(1, int(top_k))]


def retrieve_content(query, retrieval_config: Optional[Dict[str, int]] = None, source: str = "auto"):
    if not query:
        return []

    config = _resolve_config(retrieval_config)

    source_value = choose_source(query, source)

    if source_value == "courses":
        course_hits = search_courses(query, top_k=config["chunk_limit"])
        if course_hits:
            return course_hits

    return search_patents(query, retrieval_config=config)

class RetrievalService:
    def __init__(self):
        # We use SQLAlchemy model queries directly.
        pass

    def retrieve_content(self, query, retrieval_config: Optional[Dict[str, int]] = None, source: str = "auto"):
        return retrieve_content(query, retrieval_config, source=source)

    def search_patents(self, query: str, retrieval_config: Optional[Dict[str, int]] = None):
        return search_patents(query, retrieval_config=retrieval_config)

    def search_courses(self, query: str, top_k: int = 5):
        return search_courses(query, top_k=top_k)

    def close_connection(self):
        pass