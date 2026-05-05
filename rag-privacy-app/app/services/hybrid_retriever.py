from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None

from app.services.embedding_service import embed_text


class HybridRetriever:
    def __init__(
        self,
        kb_dir: str | Path,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.kb_dir = Path(kb_dir)
        self.vector_weight = float(vector_weight)
        self.bm25_weight = float(bm25_weight)

        self.index = None
        self.documents: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self._bm25 = None
        self._tokenized_docs: list[list[str]] = []

        self._load()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", (text or "").lower())

    @staticmethod
    def _minmax(values: dict[int, float]) -> dict[int, float]:
        if not values:
            return {}
        arr = np.asarray(list(values.values()), dtype=np.float32)
        v_min = float(arr.min())
        v_max = float(arr.max())
        if v_max <= v_min:
            return {k: 0.0 for k in values}
        return {k: (float(v) - v_min) / (v_max - v_min) for k, v in values.items()}

    def _load(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss-cpu is not available")

        index_path = self.kb_dir / "index.faiss"
        docs_path = self.kb_dir / "documents.json"
        meta_path = self.kb_dir / "metadata.json"

        if not index_path.exists() or not docs_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"KB files missing under {self.kb_dir}. Expected index.faiss/documents.json/metadata.json"
            )

        self.index = faiss.read_index(str(index_path))
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if not isinstance(docs, list) or not isinstance(meta, list) or len(docs) != len(meta):
            raise ValueError("Invalid KB payload: documents/metadata mismatch")

        self.documents = [str(d) for d in docs]
        self.metadata = [m if isinstance(m, dict) else {"title": str(m)} for m in meta]

        self._tokenized_docs = [self._tokenize(d) for d in self.documents]
        if BM25Okapi is not None:
            self._bm25 = BM25Okapi(self._tokenized_docs)

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if self.index is None or not self.documents:
            return []

        top_k = max(1, int(top_k))
        candidate_k = min(len(self.documents), max(50, top_k * 8))

        q_vec = np.asarray(embed_text(query), dtype=np.float32).reshape(1, -1)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0.0:
            q_vec = q_vec / q_norm

        vec_scores, vec_indices = self.index.search(q_vec, candidate_k)
        vec_map: dict[int, float] = {
            int(i): float(s)
            for i, s in zip(vec_indices[0].tolist(), vec_scores[0].tolist())
            if int(i) >= 0
        }

        bm25_map: dict[int, float] = {}
        if self._bm25 is not None:
            q_tokens = self._tokenize(query)
            bm_scores = self._bm25.get_scores(q_tokens)
            if len(bm_scores) > 0:
                top_idx = np.argpartition(bm_scores, -candidate_k)[-candidate_k:]
                for i in top_idx.tolist():
                    bm25_map[int(i)] = float(bm_scores[int(i)])
        else:
            q_tokens = set(self._tokenize(query))
            for i, doc_tokens in enumerate(self._tokenized_docs):
                overlap = len(q_tokens & set(doc_tokens))
                if overlap > 0:
                    bm25_map[i] = float(overlap)

        candidates = set(vec_map.keys()) | set(bm25_map.keys())
        if not candidates:
            return []

        vec_norm = self._minmax({i: vec_map.get(i, 0.0) for i in candidates})
        bm_norm = self._minmax({i: bm25_map.get(i, 0.0) for i in candidates})

        fused: list[dict[str, Any]] = []
        for i in candidates:
            score = self.vector_weight * vec_norm.get(i, 0.0) + self.bm25_weight * bm_norm.get(i, 0.0)
            fused.append(
                {
                    "id": int(i),
                    "content": self.documents[i],
                    "metadata": self.metadata[i],
                    "score": float(score),
                    "vector_score": float(vec_map.get(i, 0.0)),
                    "bm25_score": float(bm25_map.get(i, 0.0)),
                }
            )

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]
