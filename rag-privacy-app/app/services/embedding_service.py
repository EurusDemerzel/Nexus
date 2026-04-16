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
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_ROOT_DIR = Path(__file__).resolve().parents[2]
_LOCAL_MODEL_DIR = _ROOT_DIR / "models" / "all-MiniLM-L6-v2"
_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def _load_sentence_transformer():
    if SentenceTransformer is None:
        return False

    # 1) Prefer local project model folder.
    if _LOCAL_MODEL_DIR.exists():
        try:
            return SentenceTransformer(str(_LOCAL_MODEL_DIR))
        except Exception:
            pass

    # 2) Prefer local Hugging Face cache only (no network).
    try:
        return SentenceTransformer(
            _MODEL_NAME,
            cache_folder=str(_HF_CACHE_DIR),
            local_files_only=True,
        )
    except Exception:
        pass

    # 3) Fallback to mirror-based download when local assets are unavailable.
    try:
        return SentenceTransformer(
            _MODEL_NAME,
            cache_folder=str(_HF_CACHE_DIR),
        )
    except Exception:
        return False


def get_model():
    global _model
    if _model is None:
        _model = _load_sentence_transformer()
    return _model


def embed_text(text: str):
    model = get_model()
    if model:
        try:
            return model.encode(text)
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