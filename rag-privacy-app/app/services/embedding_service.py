import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

_model = None
_FALLBACK_DIM = 384

def get_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', local_files_only=True)
        except Exception:
            _model = False
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