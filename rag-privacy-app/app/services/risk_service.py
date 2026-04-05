from typing import Dict


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _norm(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return _clamp(value / max_value)


def evaluate_risk(features: Dict[str, float]) -> Dict[str, float]:
    """Rule-based risk scoring for online routing.

    Higher risk means we should prefer conservative strategy.
    """
    query_length = float(features.get("query_length", 0.0))
    retrieved_docs = float(features.get("retrieved_docs", 0.0))
    avg_doc_score = float(features.get("avg_doc_score", 0.0))
    latency_ms = float(features.get("latency_ms", 0.0))

    # Long query and low retrieval quality generally imply higher uncertainty.
    complexity = _norm(query_length, 80.0)
    sparsity = 1.0 - _norm(retrieved_docs, 8.0)
    low_relevance = 1.0 - _norm(avg_doc_score, 12.0)
    latency_penalty = _norm(latency_ms, 4000.0)

    score = _clamp(0.35 * complexity + 0.3 * sparsity + 0.25 * low_relevance + 0.1 * latency_penalty)

    if score >= 0.67:
        level = "P_high"
    elif score >= 0.4:
        level = "P_medium"
    else:
        level = "P_low"

    return {
        "level": level,
        "score": round(score, 4),
        "complexity": round(complexity, 4),
        "sparsity": round(sparsity, 4),
        "low_relevance": round(low_relevance, 4),
    }
