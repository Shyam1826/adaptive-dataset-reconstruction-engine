"""Central confidence scoring utilities for adaptive inference."""

from typing import Dict, List, Optional


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_linear(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low))


def combine_weighted(signals: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted sum normalized to 0–1 when weights sum to 1."""
    if not signals:
        return 0.0
    total_w = sum(weights.get(k, 0.0) for k in signals)
    if total_w <= 0:
        return sum(signals.values()) / len(signals) if signals else 0.0
    score = sum(signals.get(k, 0.0) * weights.get(k, 0.0) for k in signals)
    return clamp(score / total_w if total_w > 1.0 else score)


def combine_product(signals: Dict[str, float], floor: float = 0.01) -> float:
    """Multiplicative combination — all signals must be reasonably strong."""
    if not signals:
        return 0.0
    product = 1.0
    for v in signals.values():
        product *= clamp(v, floor, 1.0)
    return product


def margin_over_alternatives(best: float, others: List[float]) -> float:
    """Confidence boost when best hypothesis clearly wins."""
    if not others:
        return 1.0
    second = max(others) if others else 0.0
    return clamp(0.5 + (best - second) * 2.0)


def score_to_percent(score_0_1: float) -> float:
    return round(clamp(score_0_1) * 100.0, 2)


class ConfidenceReport:
    """Aggregated confidence from multiple pipeline layers."""

    def __init__(self):
        self.layers: Dict[str, float] = {}
        self.signals: Dict[str, Dict[str, float]] = {}

    def add_layer(self, layer: str, score: float, signals: Optional[Dict[str, float]] = None):
        self.layers[layer] = clamp(score)
        if signals:
            self.signals[layer] = signals

    @property
    def global_confidence(self) -> float:
        if not self.layers:
            return 0.0
        weights = {
            "profiling": 0.08,
            "header": 0.18,
            "blueprint": 0.12,
            "reconstruction": 0.32,
            "semantic": 0.18,
            "boundary": 0.12,
        }
        available = {k: v for k, v in self.layers.items() if k in weights}
        if not available:
            return sum(self.layers.values()) / len(self.layers)
        w_sum = sum(weights[k] for k in available)
        return sum(available[k] * weights[k] for k in available) / w_sum

    def to_dict(self) -> Dict:
        return {
            "global_confidence": self.global_confidence,
            "layers": self.layers,
            "signals": self.signals,
        }
