"""Hypothesis containers and selection for confidence-driven inference."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

from .confidence import clamp, margin_over_alternatives

T = TypeVar("T")


@dataclass
class ScoredHypothesis(Generic[T]):
    name: str
    value: T
    confidence: float
    signals: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = clamp(self.confidence)


class HypothesisSelector:
    """Select highest-confidence hypothesis with optional margin requirement."""

    @staticmethod
    def select_best(
        hypotheses: List[ScoredHypothesis],
        min_confidence: float = 0.0,
    ) -> Optional[ScoredHypothesis]:
        if not hypotheses:
            return None
        ranked = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        best = ranked[0]
        if best.confidence < min_confidence:
            return None
        others = [h.confidence for h in ranked[1:]]
        margin = margin_over_alternatives(best.confidence, others)
        best.signals["selection_margin"] = margin
        return best

    @staticmethod
    def rank(hypotheses: List[ScoredHypothesis]) -> List[ScoredHypothesis]:
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
