from typing import Dict


DEFAULT_WEIGHTS: Dict[str, float] = {
    "Wc": 0.6,
    "Wm": 0.4,
}


RISK_BASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "P_high": {"Wc": 0.7, "Wm": 0.3},
    "P_medium": {"Wc": 0.5, "Wm": 0.5},
    "P_low": {"Wc": 0.4, "Wm": 0.6},
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(wc: float, wm: float) -> Dict[str, float]:
    s = wc + wm
    if s <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {"Wc": round(wc / s, 4), "Wm": round(wm / s, 4)}


def update_weights(
    risk_level: str,
    context_signal: float,
    memory_signal: float,
    prev_weights: Dict[str, float] | None = None,
    threshold_mem: float = 0.65,
    delta: float = 0.05,
) -> Dict[str, float]:
    base = RISK_BASE_WEIGHTS.get(risk_level, DEFAULT_WEIGHTS)

    wc = float(base["Wc"])
    wm = float(base["Wm"])

    if prev_weights:
        wc = 0.6 * wc + 0.4 * float(prev_weights.get("Wc", wc))
        wm = 0.6 * wm + 0.4 * float(prev_weights.get("Wm", wm))

    # If memory signal is high, increase memory weight.
    if memory_signal > threshold_mem:
        wm += delta
        wc -= delta

    # Context signal strong -> slightly raise context weight.
    if context_signal > 0.7:
        wc += delta / 2
        wm -= delta / 2

    wc = _clamp(wc, 0.1, 0.9)
    wm = _clamp(wm, 0.1, 0.9)

    return _normalize(wc, wm)
