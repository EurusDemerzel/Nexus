import re
from typing import Dict, List, Optional

from app.models import Patent


DEFAULT_RETRIEVAL_CONFIG: Dict[str, int] = {
    "candidate_limit": 50,
    "top_doc_limit": 10,
    "chunk_limit": 5,
    "min_chunk_score": 1,
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


def retrieve_content(query, retrieval_config: Optional[Dict[str, int]] = None):
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

    # Dynamic split + chunk-level rerank
    chunk_candidates = []
    for p in top_docs:
        chunks = _dynamic_split_text(p.content or "", query, config.get("max_chunk"))
        for idx, chunk in enumerate(chunks):
            score = _score_document(query, p.title or "", chunk)
            if score >= config["min_chunk_score"]:
                chunk_candidates.append(
                    {
                        "title": p.title,
                        "content": chunk,
                        "chunk_index": idx,
                        "score": score,
                    }
                )

    chunk_candidates.sort(key=lambda x: x["score"], reverse=True)
    return chunk_candidates[: config["chunk_limit"]]

class RetrievalService:
    def __init__(self):
        # We use SQLAlchemy model queries directly.
        pass

    def retrieve_content(self, query, retrieval_config: Optional[Dict[str, int]] = None):
        return retrieve_content(query, retrieval_config)

    def close_connection(self):
        pass