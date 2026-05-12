import hashlib
import os
from pathlib import Path

import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


_model = None
_FALLBACK_DIM = 384
_ROOT_DIR = Path(__file__).resolve().parents[2]
_LOCAL_MODEL_DIRS = [
    _ROOT_DIR / "models_for_server" / "BAAI" / "bge-base-en-v1.5",
    _ROOT_DIR / "models_for_server" / "all-MiniLM-L6-v2",
    _ROOT_DIR / "models" / "all-MiniLM-L6-v2",
]
_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
_FORCE_FALLBACK = os.getenv("NEXUS_EMBED_FALLBACK_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def _pick_best_model_dir() -> str:
    """根据 FAISS 索引维度或可用的本地模型，自动选择嵌入模型。"""
    # 优先使用显式环境变量
    explicit = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    if explicit:
        return explicit

    # 按优先级尝试：BGE > MiniLM
    for _dir in _LOCAL_MODEL_DIRS:
        if _dir.exists():
            return str(_dir)

    return "sentence-transformers/all-MiniLM-L6-v2"


_MODEL_NAME = _pick_best_model_dir()


def _load_sentence_transformer():
    if SentenceTransformer is None:
        return False

    try:
        return SentenceTransformer(
            _MODEL_NAME,
            cache_folder=str(_HF_CACHE_DIR),
            local_files_only=os.getenv("HF_DATASETS_OFFLINE", "") == "1" or None,
        )
    except Exception:
        return False
    except Exception:
        return False


def get_model():
    global _model
    if _FORCE_FALLBACK:
        return False
    if _model is None:
        _model = _load_sentence_transformer()
    return _model


def embed_text(text: str):
    model = get_model()
    if model:
        try:
            return model.encode(text)
        except KeyboardInterrupt:
            # Graceful degrade: avoid crashing long-running experiments when encoding is interrupted.
            print("[embedding_service] interrupted during model.encode, fallback to hash embedding")
            return _fallback_embed(text)
        except Exception:
            pass
    return _fallback_embed(text)


def _fallback_embed(text: str) -> np.ndarray:
    vec = np.zeros(_FALLBACK_DIM, dtype=np.float32)
    for token in (text or ""):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="big") % _FALLBACK_DIM
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec

# Backward-compatible aliases.
def get_embedding_model():
    return get_model()

def embed_query(text: str):
    return embed_text(text)