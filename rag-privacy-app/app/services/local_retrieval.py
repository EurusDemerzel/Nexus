from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from app.services.hybrid_retriever import HybridRetriever
except Exception:
    HybridRetriever = None

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
TRIVIAQA_KB_DIR = ROOT_DIR / "triviaqa_kb"
TRIVIAQA_INDEX_FILE = TRIVIAQA_KB_DIR / "index.faiss"
TRIVIAQA_DOCS_FILE = TRIVIAQA_KB_DIR / "documents.json"
TRIVIAQA_META_FILE = TRIVIAQA_KB_DIR / "metadata.json"

LOCAL_NOTES: list[dict] = []
_NOTE_EMBEDDINGS = None
_FAISS_INDEX = None
_FAISS_DOCS: list[str] = []
_FAISS_METADATA: list[dict] = []
_HYBRID_RETRIEVER = None


def _load_faiss_kb() -> bool:
    global _FAISS_INDEX, _FAISS_DOCS, _FAISS_METADATA

    if _FAISS_INDEX is not None and _FAISS_DOCS and _FAISS_METADATA:
        return True

    if faiss is None:
        print("[local_retrieval.faiss] faiss-cpu 未安装，无法使用 FAISS 检索")
        return False

    if not (TRIVIAQA_INDEX_FILE.exists() and TRIVIAQA_DOCS_FILE.exists() and TRIVIAQA_META_FILE.exists()):
        print(
            "[local_retrieval.faiss] 未找到 TriviaQA 知识库文件: "
            f"{TRIVIAQA_INDEX_FILE}, {TRIVIAQA_DOCS_FILE}, {TRIVIAQA_META_FILE}"
        )
        return False

    try:
        _FAISS_INDEX = faiss.read_index(str(TRIVIAQA_INDEX_FILE))
        docs = json.loads(TRIVIAQA_DOCS_FILE.read_text(encoding="utf-8"))
        meta = json.loads(TRIVIAQA_META_FILE.read_text(encoding="utf-8"))

        if not isinstance(docs, list) or not isinstance(meta, list):
            raise ValueError("documents/metadata 格式错误")
        if len(docs) != len(meta):
            raise ValueError("documents 与 metadata 数量不一致")

        _FAISS_DOCS = [str(x) for x in docs]
        _FAISS_METADATA = [m if isinstance(m, dict) else {"title": str(m)} for m in meta]
        return True
    except Exception as exc:
        print(f"[local_retrieval.faiss][ERROR] 加载失败: {type(exc).__name__}: {exc}")
        _FAISS_INDEX = None
        _FAISS_DOCS = []
        _FAISS_METADATA = []
        return False


def retrieve_from_faiss(query: str, top_k: int = 8) -> list[dict]:
    top_k = max(1, int(top_k))
    query_preview = (query or "").replace("\n", " ")[:50]
    print(f"[local_retrieval.retrieve_from_faiss] query='{query_preview}' top_k={top_k}")

    if not _load_faiss_kb():
        return []

    if not _FAISS_DOCS:
        return []

    query_vec = np.asarray(embed_text(query), dtype=np.float32).reshape(1, -1)
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm > 0.0:
        query_vec = query_vec / q_norm

    k = min(top_k, len(_FAISS_DOCS))
    scores, indices = _FAISS_INDEX.search(query_vec, k)

    results: list[dict] = []
    for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
        if idx < 0 or idx >= len(_FAISS_DOCS):
            continue
        results.append(
            {
                "id": int(idx),
                "content": _FAISS_DOCS[idx],
                "metadata": _FAISS_METADATA[idx],
                "score": float(score),
            }
        )

    print(f"[local_retrieval.retrieve_from_faiss] hits={len(results)}")
    return results


def _get_hybrid_retriever():
    global _HYBRID_RETRIEVER
    if _HYBRID_RETRIEVER is not None:
        return _HYBRID_RETRIEVER

    if HybridRetriever is None:
        print("[local_retrieval.hybrid] HybridRetriever 不可用")
        return None

    kb_dir = os.getenv("TRIVIAQA_KB_DIR", str(TRIVIAQA_KB_DIR))
    vector_weight = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.7"))
    bm25_weight = float(os.getenv("HYBRID_BM25_WEIGHT", "0.3"))

    try:
        _HYBRID_RETRIEVER = HybridRetriever(
            kb_dir=kb_dir,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
    except Exception as exc:
        print(f"[local_retrieval.hybrid][ERROR] 初始化失败: {type(exc).__name__}: {exc}")
        _HYBRID_RETRIEVER = None
    return _HYBRID_RETRIEVER


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


# MOD: 简单实体+词汇重排序，提升多实体问题召回（如 Scott Derrickson + Ed Wood）
def _extract_entities(query: str) -> list[str]:
    # 捕获英文专有名词短语：例如 "Scott Derrickson"、"Ed Wood"
    entities = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", query or "")
    lead_stopwords = {
        "were",
        "was",
        "is",
        "are",
        "do",
        "does",
        "did",
        "what",
        "who",
        "where",
        "when",
        "which",
        "why",
        "how",
    }
    # 过滤过短噪声，并统一去重
    dedup: list[str] = []
    seen = set()
    for ent in entities:
        parts = ent.strip().split()
        # 移除问句开头动词，避免 "Were Scott Derrickson" 这类误抽取
        while parts and parts[0].lower() in lead_stopwords:
            parts = parts[1:]
        if len(parts) < 2:
            continue

        key = " ".join(parts).lower()
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        dedup.append(key)
    return dedup


def _lexical_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    query_text = (query or "").lower()
    query_tokens = set(re.findall(r"[a-z0-9]+", query_text))
    entities = _extract_entities(query or "")

    rescored = []
    for item in candidates:
        metadata = item.get("metadata") or {}
        title = str(metadata.get("title", ""))
        content = str(item.get("content", ""))
        doc_text = (title + " " + content).lower()
        doc_tokens = set(re.findall(r"[a-z0-9]+", doc_text))

        # 实体命中分（优先确保问题关键实体被召回）
        entity_hits = sum(1 for ent in entities if ent in doc_text)
        title_entity_hits = sum(1 for ent in entities if ent in title.lower())

        # 词汇重合分（轻量 tf-idf/bm25 替代）
        token_overlap = len(query_tokens & doc_tokens)

        base_score = float(item.get("score", 0.0))
        rerank_bonus = (entity_hits * 1.5) + (title_entity_hits * 2.0) + (token_overlap * 0.01)
        final_score = base_score + rerank_bonus

        new_item = dict(item)
        new_item["dense_score"] = round(base_score, 6)
        new_item["entity_hits"] = int(entity_hits)
        new_item["title_entity_hits"] = int(title_entity_hits)
        new_item["rerank_score"] = round(final_score, 6)
        rescored.append(new_item)

    rescored.sort(
        key=lambda x: (
            x.get("title_entity_hits", 0),
            x.get("entity_hits", 0),
            x.get("rerank_score", 0.0),
            x.get("dense_score", 0.0),
        ),
        reverse=True,
    )

    # 覆盖式重排序：每个关键实体至少保留一个最相关文档，再补齐剩余名额
    selected: list[dict] = []
    selected_ids: set[int] = set()

    if entities:
        for ent in entities:
            for item in rescored:
                doc_id = int(item.get("id", -1))
                title = str((item.get("metadata") or {}).get("title", "")).lower()
                content = str(item.get("content", "")).lower()
                if doc_id in selected_ids:
                    continue
                if ent in title or ent in content:
                    selected.append(item)
                    selected_ids.add(doc_id)
                    break

    for item in rescored:
        if len(selected) >= top_k:
            break
        doc_id = int(item.get("id", -1))
        if doc_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(doc_id)

    return selected[:top_k]


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


def retrieve(query: str, top_k: int = 8) -> list[dict]:
    global _NOTE_EMBEDDINGS

    top_k = max(1, int(top_k))

    retriever_mode = os.getenv("RETRIEVAL_BACKEND", "").strip().lower()
    if not retriever_mode:
        retriever_mode = os.getenv("NEXUS_RETRIEVER", "notes").strip().lower()

    if retriever_mode in {"triviaqa-hybrid", "hybrid", "triviaqa"}:
        hybrid = _get_hybrid_retriever()
        if hybrid is not None:
            hybrid_hits = hybrid.search(query, top_k=top_k)
            if hybrid_hits:
                print(f"[local_retrieval.retrieve] backend=hybrid hits={len(hybrid_hits)}")
                return hybrid_hits
        print("[local_retrieval.retrieve] Hybrid 检索为空，回退到 FAISS/notes")

    if retriever_mode in {"faiss", "triviaqa-faiss", "triviaqa"}:
        faiss_hits = retrieve_from_faiss(query, top_k=top_k)
        if faiss_hits:
            print(f"[local_retrieval.retrieve] backend=faiss hits={len(faiss_hits)}")
            return faiss_hits
        print("[local_retrieval.retrieve] FAISS 检索为空，回退到 local_notes_store.json")

    _ensure_notes_loaded()

    # MOD: 调试日志 - 打印查询前 50 字符与 top_k
    query_preview = (query or "").replace("\n", " ")[:50]
    print(f"[local_retrieval.retrieve] query='{query_preview}' top_k={top_k}")

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

    candidate_k = max(30, top_k)
    dense_candidates = scored[:candidate_k]
    # MOD: 先做稠密召回 candidate_k=30，再做实体/词汇重排序
    results = _lexical_rerank(query=query, candidates=dense_candidates, top_k=top_k)

    # MOD: 调试日志 - 逐条打印检索结果详情
    print(f"[local_retrieval.retrieve] hits={len(results)}")
    for i, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        title = metadata.get("title", f"doc_{i}")
        score = item.get("score", 0.0)
        dense_score = item.get("dense_score", score)
        rerank_score = item.get("rerank_score", score)
        entity_hits = item.get("entity_hits", 0)
        title_entity_hits = item.get("title_entity_hits", 0)
        content_preview = (item.get("content") or "").replace("\n", " ")[:100]
        print(
            f"  [hit#{i}] title={title} score={score} dense={dense_score} rerank={rerank_score} "
            f"entity_hits={entity_hits} title_entity_hits={title_entity_hits} preview={content_preview}"
        )

    return results


def retrieve_local(query: str, top_k: int = 3) -> list[dict]:
    return retrieve(query, top_k=top_k)
