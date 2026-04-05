from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict


_POLICY_VERSION = 1
_POLICY_UPDATED_AT = datetime.now(timezone.utc).isoformat()

_POLICY_MAP: Dict[str, Dict[str, int]] = {
    "precision_first": {
        "candidate_limit": 35,
        "top_doc_limit": 8,
        "chunk_limit": 4,
        "min_chunk_score": 2,
        "max_chunk": 220,
    },
    "balanced": {
        "candidate_limit": 50,
        "top_doc_limit": 10,
        "chunk_limit": 5,
        "min_chunk_score": 1,
        "max_chunk": 320,
    },
    "recall_first": {
        "candidate_limit": 70,
        "top_doc_limit": 14,
        "chunk_limit": 6,
        "min_chunk_score": 1,
        "max_chunk": 420,
    },
}


def get_latest_policy() -> Dict[str, Any]:
    return {
        "version": _POLICY_VERSION,
        "updated_at": _POLICY_UPDATED_AT,
        "map": deepcopy(_POLICY_MAP),
    }


def reload_policy(new_map: Dict[str, Dict[str, int]] | None = None) -> Dict[str, Any]:
    global _POLICY_VERSION
    global _POLICY_UPDATED_AT

    if new_map:
        for key in ["precision_first", "balanced", "recall_first"]:
            if key in new_map and isinstance(new_map[key], dict):
                _POLICY_MAP[key].update(new_map[key])

    _POLICY_VERSION += 1
    _POLICY_UPDATED_AT = datetime.now(timezone.utc).isoformat()
    return get_latest_policy()


def choose_policy(predicted_label: str, predicted_score: float) -> Dict[str, Any]:
    label = predicted_label if predicted_label in _POLICY_MAP else "balanced"
    config = deepcopy(_POLICY_MAP[label])

    # Slightly tighten strategy when uncertainty is very high.
    if predicted_score >= 0.85:
        config["chunk_limit"] = max(3, int(config["chunk_limit"] - 1))
        config["min_chunk_score"] = max(1, int(config["min_chunk_score"] + 1))

    return {
        "strategy_label": label,
        "policy_version": _POLICY_VERSION,
        "retrieval_config": config,
    }
