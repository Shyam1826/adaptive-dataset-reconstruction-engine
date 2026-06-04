import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .models import DatasetProfile
from .row_signature import build_row_signature, classify_cell, detect_repeating_patterns

# Reuse anchor patterns from boundary_detector without circular import at module level
ANCHOR_VALUE_PATTERNS = [
    (re.compile(r"^TXN[_-][A-Z0-9]+", re.I), "txn_prefix"),
    (re.compile(r"^ORD[_-][A-Z0-9]+", re.I), "order_prefix"),
    (re.compile(r"^INV[_-][A-Z0-9]+", re.I), "invoice_prefix"),
    (re.compile(r"^CUST[_-][A-Z0-9]+", re.I), "customer_prefix"),
    (re.compile(r"^STU[_-][A-Z0-9]+", re.I), "student_prefix"),
    (
        re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.I,
        ),
        "uuid",
    ),
]

HEADER_LABEL_KEYWORDS = {
    "id",
    "date",
    "amount",
    "score",
    "total",
    "quantity",
    "age",
    "name",
    "category",
    "status",
    "description",
    "price",
    "value",
    "type",
    "index",
    "guid",
    "sum",
    "customer",
    "transaction",
    "order",
    "invoice",
    "payment",
    "location",
    "discount",
    "item",
}

DATA_START_THRESHOLD = 42.0


@dataclass
class HeaderBoundaryResult:
    data_start_row: int
    header_rows: List[int]
    boundary_confidence: float
    row_scores: List[float] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)


class HeaderBoundaryDetector:
    """Detects the transition from header region to repeating data region."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        profile: DatasetProfile,
        header_confidences: Optional[List[float]] = None,
    ):
        self.raw_df = raw_df
        self.profile = profile
        self.header_confidences = header_confidences or []

    def detect(self) -> HeaderBoundaryResult:
        n = self.profile.total_rows
        if n == 0:
            return HeaderBoundaryResult(0, [], 0.0)

        row_scores: List[float] = []
        row_signals: List[Dict[str, float]] = []

        for i in range(n):
            score, signals = self._data_start_score(i)
            row_scores.append(score)
            row_signals.append(signals)

        data_start = self._resolve_data_start(row_scores, row_signals)
        header_rows = self._finalize_header_rows(data_start)

        boundary_confidence = (
            row_scores[data_start] if data_start < len(row_scores) else 0.0
        )

        return HeaderBoundaryResult(
            data_start_row=data_start,
            header_rows=header_rows,
            boundary_confidence=boundary_confidence,
            row_scores=row_scores,
            signals={
                "data_start_row": data_start,
                "periodicity_start": self._find_periodicity_start(),
                "anchor_start": self._find_first_anchor_row(),
            },
        )

    def _resolve_data_start(
        self,
        row_scores: List[float],
        row_signals: List[Dict[str, float]],
    ) -> int:
        n = len(row_scores)
        if n == 0:
            return 0

        periodicity_start = self._find_periodicity_start()
        anchor_start = self._find_first_anchor_row()

        candidates: List[Tuple[int, float]] = []

        for i in range(n):
            row = self.raw_df.iloc[i]
            if self._row_looks_like_label_header(row):
                continue
            if row_scores[i] >= DATA_START_THRESHOLD:
                candidates.append((i, row_scores[i]))

        if anchor_start is not None:
            candidates.append((anchor_start, row_scores[anchor_start] + 10))

        if periodicity_start is not None:
            candidates.append(
                (periodicity_start, row_scores[periodicity_start] + 8)
            )

        if candidates:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            data_start = self._pick_best_candidate(candidates, row_scores, row_signals)
        else:
            data_start = self._fallback_data_start(row_scores)

        return self._ensure_label_rows_not_data_start(data_start, row_scores)

    def _ensure_label_rows_not_data_start(
        self, data_start: int, row_scores: List[float]
    ) -> int:
        n = self.profile.total_rows
        if n == 0:
            return 0

        while data_start < n and self._row_looks_like_label_header(
            self.raw_df.iloc[data_start]
        ):
            data_start += 1

        if data_start == 0 and n > 1:
            row0 = self.raw_df.iloc[0]
            if self._row_looks_like_label_header(row0):
                for i in range(1, n):
                    if not self._row_looks_like_label_header(self.raw_df.iloc[i]):
                        if row_scores[i] >= 25 or self._row_looks_like_data_value(
                            self.raw_df.iloc[i]
                        ):
                            return i
                return 1

        return min(data_start, n)

    def _pick_best_candidate(
        self,
        candidates: List[Tuple[int, float]],
        row_scores: List[float],
        row_signals: List[Dict[str, float]],
    ) -> int:
        """Prefer earliest row with strong structural data signals."""
        for idx, _ in sorted(candidates, key=lambda x: x[0]):
            sig = row_signals[idx] if idx < len(row_signals) else {}
            anchor = sig.get("anchor_values", 0)
            periodic = sig.get("periodicity", 0)
            label_penalty = sig.get("label_likelihood", 0)

            if anchor >= 25 or periodic >= 20:
                if label_penalty < 15:
                    return idx
            if row_scores[idx] >= DATA_START_THRESHOLD + 15:
                row = self.raw_df.iloc[idx]
                if not self._row_looks_like_label_header(row):
                    return idx

        for idx, _ in sorted(candidates, key=lambda x: x[0]):
            if not self._row_looks_like_label_header(self.raw_df.iloc[idx]):
                return idx
        return candidates[0][0]

    def _fallback_data_start(self, row_scores: List[float]) -> int:
        n = len(row_scores)
        if n <= 1:
            return 0 if n == 0 else 1

        best_idx = 0
        best_margin = -999.0
        for i in range(1, n):
            header_conf = (
                self.header_confidences[i - 1]
                if i - 1 < len(self.header_confidences)
                else 50.0
            )
            margin = row_scores[i] - header_conf * 0.4
            if margin > best_margin:
                best_margin = margin
                best_idx = i

        if row_scores[best_idx] < 25 and best_idx == 0:
            return 1 if n > 1 else 0
        return best_idx

    def _finalize_header_rows(self, data_start: int) -> List[int]:
        header_rows = []
        for i in range(data_start):
            row = self.raw_df.iloc[i]
            if row.isna().all():
                continue
            conf = (
                self.header_confidences[i]
                if i < len(self.header_confidences)
                else 0.0
            )
            if conf >= 25.0 or self._row_looks_like_label_header(row):
                header_rows.append(i)
            elif not header_rows:
                header_rows.append(i)
        return header_rows

    def _data_start_score(self, row_idx: int) -> Tuple[float, Dict[str, float]]:
        row = self.raw_df.iloc[row_idx]
        signals: Dict[str, float] = {}

        if row.isna().all():
            return 0.0, signals

        anchor_density = self._anchor_value_density(row)
        label_likelihood = self._label_header_likelihood(row)
        signals["anchor_values"] = anchor_density * 100
        signals["label_likelihood"] = label_likelihood * 100

        rp = self.profile.row_profiles[row_idx]
        if rp.numeric_ratio > 0.35:
            signals["numeric_density"] = rp.numeric_ratio * 40
        if "D" in rp.signature or "N" in rp.signature:
            signals["datatype_data"] = 20.0
        if label_likelihood < 0.3 and anchor_density > 0.2:
            signals["anchor_not_label"] = 30.0

        if label_likelihood >= 0.45:
            signals["strong_label_penalty"] = -label_likelihood * 90
        elif label_likelihood >= 0.25:
            signals["label_penalty"] = -label_likelihood * 50

        periodic = self._periodicity_strength_from(row_idx)
        signals["periodicity"] = periodic

        if row_idx > 0:
            prev_sig = self.profile.row_profiles[row_idx - 1].signature
            sim = _signature_similarity(prev_sig, rp.signature)
            if sim > 0.7 and anchor_density > 0.15:
                signals["signature_continuation"] = 15.0
            if sim < 0.35 and anchor_density > 0.1:
                signals["signature_reset"] = 12.0

        if row_idx + 1 < self.profile.total_rows:
            next_rp = self.profile.row_profiles[row_idx + 1]
            if _signature_similarity(rp.signature, next_rp.signature) > 0.65:
                signals["forward_repeat"] = 18.0

        header_conf = (
            self.header_confidences[row_idx]
            if row_idx < len(self.header_confidences)
            else 0.0
        )
        signals["header_confidence_penalty"] = -header_conf * 0.25

        if self._row_looks_like_data_value(row):
            signals["data_value_shape"] = 25.0

        total = sum(v for v in signals.values() if v > 0) + signals.get(
            "header_confidence_penalty", 0
        )
        return max(0.0, total), signals

    def _anchor_value_density(self, row: pd.Series) -> float:
        values = [row.iloc[i] for i in range(len(row)) if pd.notna(row.iloc[i])]
        if not values:
            return 0.0
        hits = sum(1 for v in values if _matches_anchor_value(v))
        return hits / len(values)

    def _label_header_likelihood(self, row: pd.Series) -> float:
        values = [
            str(row.iloc[i]).strip()
            for i in range(len(row))
            if pd.notna(row.iloc[i]) and str(row.iloc[i]).strip()
        ]
        if not values:
            return 0.0

        label_hits = 0
        for val in values:
            low = val.lower().replace(" ", "_")
            if _matches_anchor_value(val):
                continue
            if any(kw in low for kw in HEADER_LABEL_KEYWORDS):
                label_hits += 1
                continue
            if len(val) < 35 and re.match(r"^[a-zA-Z_][a-zA-Z0-9_\s-]*$", val):
                label_hits += 0.5

        return label_hits / len(values)

    def _row_looks_like_label_header(self, row: pd.Series) -> bool:
        if self._anchor_value_density(row) >= 0.25:
            return False
        return self._label_header_likelihood(row) >= 0.4

    def _row_looks_like_data_value(self, row: pd.Series) -> bool:
        anchor = self._anchor_value_density(row)
        if anchor >= 0.2:
            return True
        classes = [classify_cell(row.iloc[i]) for i in range(len(row)) if pd.notna(row.iloc[i])]
        if not classes:
            return False
        has_n = "N" in classes
        has_d = "D" in classes
        has_t = "T" in classes
        return (has_n or has_d) and has_t and self._label_header_likelihood(row) < 0.35

    def _periodicity_strength_from(self, start: int) -> float:
        signatures = [rp.signature for rp in self.profile.row_profiles[start:]]
        active = [s for s in signatures if set(s) != {"E"}]
        if len(active) < 3:
            return 0.0

        span, _ = detect_repeating_patterns(active, max_pattern_len=min(8, len(active)))
        if span <= 0:
            return 0.0

        blocks = [
            tuple(active[i : i + span])
            for i in range(0, len(active) - span + 1, span)
        ]
        if len(blocks) < 2:
            return 0.0

        from collections import Counter

        counts = Counter(blocks)
        _, count = counts.most_common(1)[0]
        ratio = count / len(blocks)
        return ratio * 35.0

    def _find_periodicity_start(self) -> Optional[int]:
        n = self.profile.total_rows
        best_start: Optional[int] = None
        best_strength = 0.0

        for start in range(n):
            strength = self._periodicity_strength_from(start)
            if strength > best_strength and strength >= 18.0:
                row = self.raw_df.iloc[start]
                if not self._row_looks_like_label_header(row):
                    best_strength = strength
                    best_start = start

        return best_start

    def _find_first_anchor_row(self) -> Optional[int]:
        for i in range(self.profile.total_rows):
            row = self.raw_df.iloc[i]
            if row.isna().all():
                continue
            if self._anchor_value_density(row) >= 0.25 and not self._row_looks_like_label_header(
                row
            ):
                return i
        return None


def _matches_anchor_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    for pattern, _ in ANCHOR_VALUE_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _signature_similarity(sig_a: str, sig_b: str) -> float:
    if not sig_a or not sig_b:
        return 0.0
    n = max(len(sig_a), len(sig_b))
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / n if n else 0.0


def trim_header_candidates(
    candidate_rows: List[int],
    boundary: HeaderBoundaryResult,
) -> List[int]:
    """Keep only header rows that fall strictly before the data region."""
    data_start = boundary.data_start_row
    trimmed = [r for r in candidate_rows if r < data_start]
    if trimmed:
        return sorted(trimmed)
    return boundary.header_rows
