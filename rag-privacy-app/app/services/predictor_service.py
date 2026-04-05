from typing import Dict, List


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def predict_state_and_label(feature_window: List[Dict[str, float]], current_score: float) -> Dict[str, float | str | int]:
    """Rule-based mock predictor for sprint-1/2.

    Returns both continuous score and discrete strategy label.
    """
    history_scores = [float(x.get("risk_score", 0.0)) for x in feature_window][-5:]
    trend = _avg(history_scores)

    predicted_score = 0.65 * float(current_score) + 0.35 * trend
    predicted_score = max(0.0, min(1.0, predicted_score))

    if predicted_score >= 0.75:
        label = "precision_first"
    elif predicted_score >= 0.45:
        label = "balanced"
    else:
        label = "recall_first"

    confidence = 0.55 + min(0.4, len(history_scores) * 0.06)

    return {
        "predicted_score": round(predicted_score, 4),
        "predicted_label": label,
        "confidence": round(min(confidence, 0.95), 4),
        "window_size": len(history_scores),
    }
