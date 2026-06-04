import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .models import DatasetProfile, StructuralBlueprint
from .row_signature import build_row_signature, classify_cell
from .structure_blueprint import (
    BlueprintReconstructor,
    infer_header_semantic_type,
    infer_value_semantic_type,
    score_row_alignment,
)


ANCHOR_KEYWORDS = (
    "transaction_id",
    "transaction",
    "invoice",
    "order_id",
    "order",
    "customer_id",
    "customer",
    "student_id",
    "student",
    "record_id",
    "uuid",
    "guid",
    "id",
    "txn",
)

ANCHOR_PATTERNS = [
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

DEFAULT_BOUNDARY_THRESHOLD = 45.0


@dataclass
class AnchorColumn:
    column_index: int
    global_index: int
    header_label: str
    pattern_type: str
    confidence: float
    segment_offset: int = 0


@dataclass
class RowBoundaryScore:
    row_index: int
    score: float
    is_boundary: bool
    segment_offset: int
    signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class BoundaryDetectionResult:
    anchor_columns: List[AnchorColumn]
    row_scores: List[RowBoundaryScore]
    record_groups: List[List[int]]


class AnchorDetector:
    """Detect identifier-like columns that typically start a new logical record."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        blueprint: StructuralBlueprint,
        header_names: Optional[List[str]] = None,
        start_idx: int = 0,
    ):
        self.raw_df = raw_df
        self.blueprint = blueprint
        self.header_names = header_names or []
        self.start_idx = start_idx

    def detect(self) -> List[AnchorColumn]:
        anchors: List[AnchorColumn] = []
        seen_global: Set[int] = set()

        for seg in self.blueprint.segments:
            offset = seg.row_offset
            for local_i, col_idx in enumerate(seg.column_indices):
                global_idx = seg.global_start + local_i
                if global_idx in seen_global:
                    continue

                header_label = (
                    self.header_names[global_idx]
                    if global_idx < len(self.header_names)
                    else f"col_{col_idx}"
                )
                confidence = self._score_anchor_column(col_idx, header_label, offset)
                if confidence >= 0.45:
                    pattern = self._dominant_pattern(col_idx)
                    anchors.append(
                        AnchorColumn(
                            column_index=col_idx,
                            global_index=global_idx,
                            header_label=header_label,
                            pattern_type=pattern,
                            confidence=confidence,
                            segment_offset=offset,
                        )
                    )
                    seen_global.add(global_idx)

        anchors.sort(key=lambda a: (-a.confidence, a.segment_offset))
        return anchors

    def _score_anchor_column(
        self, col_idx: int, header_label: str, segment_offset: int
    ) -> float:
        score = 0.0
        name = str(header_label).lower().replace(" ", "_")

        if any(kw in name for kw in ANCHOR_KEYWORDS):
            score += 0.35
        if segment_offset == 0:
            score += 0.15

        values = self._column_values(col_idx)
        if not values:
            return 0.0

        uniqueness = len(set(str(v) for v in values)) / len(values)
        if uniqueness > 0.85:
            score += 0.25
        elif uniqueness > 0.6:
            score += 0.1

        pattern_hits = sum(1 for v in values if _match_any_anchor_pattern(v)[0])
        pattern_ratio = pattern_hits / len(values)
        if pattern_ratio > 0.5:
            score += 0.35
        elif pattern_ratio > 0.2:
            score += 0.15

        numeric_ids = sum(
            1 for v in values if classify_cell(v) == "N" and _looks_like_id(v)
        )
        if numeric_ids / len(values) > 0.7 and uniqueness > 0.7:
            score += 0.15

        return min(1.0, score)

    def _dominant_pattern(self, col_idx: int) -> str:
        values = self._column_values(col_idx)
        counts: Dict[str, int] = {}
        for v in values:
            matched, ptype = _match_any_anchor_pattern(v)
            if matched:
                counts[ptype] = counts.get(ptype, 0) + 1
        if counts:
            return max(counts, key=counts.get)
        return "generic_id"

    def _column_values(self, col_idx: int) -> List[Any]:
        values = []
        for idx in range(self.start_idx, len(self.raw_df)):
            row = self.raw_df.iloc[idx]
            if row.isna().all():
                continue
            if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                val = str(row.iloc[col_idx]).strip()
                if val:
                    values.append(row.iloc[col_idx])
        return values


class BoundaryScorer:
    """Score whether a row begins a new logical record."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        blueprint: StructuralBlueprint,
        anchors: List[AnchorColumn],
        profile: Optional[DatasetProfile] = None,
        segment_templates: Optional[Dict[int, str]] = None,
        threshold: float = DEFAULT_BOUNDARY_THRESHOLD,
    ):
        self.raw_df = raw_df
        self.blueprint = blueprint
        self.anchors = anchors
        self.profile = profile
        self.segment_templates = segment_templates or {}
        self.threshold = threshold

    def score_row(
        self,
        row_index: int,
        prev_row_index: Optional[int],
        position_in_group: int,
    ) -> RowBoundaryScore:
        row = self.raw_df.iloc[row_index]
        signals: Dict[str, float] = {}

        segment_offset = infer_segment_offset(
            row, self.blueprint, position_in_group=position_in_group
        )
        signals["segment_offset"] = float(segment_offset)

        if row_has_anchor(row, self.anchors, self.blueprint):
            signals["anchor_hit"] = 40.0

        if segment_offset == 0:
            signals["segment_restart"] = 30.0
            if position_in_group > 0:
                signals["segment_restart_in_group"] = 20.0

        sig = build_row_signature(row)
        if prev_row_index is not None:
            prev_sig = build_row_signature(self.raw_df.iloc[prev_row_index])
            similarity = _signature_similarity(sig, prev_sig)
            signals["signature_similarity"] = similarity * 100
            if similarity < 0.35 and segment_offset == 0:
                signals["signature_reset"] = 15.0
            if self.segment_templates.get(0) and _signature_similarity(
                sig, self.segment_templates[0]
            ) > 0.55:
                signals["matches_segment0_template"] = 20.0

        if self.profile and self.profile.repeating_signature_span > 1:
            cycle_pos = position_in_group % self.profile.repeating_signature_span
            if cycle_pos == 0 and position_in_group > 0:
                signals["cycle_restart"] = 15.0

        if _is_structural_reset(row, self.blueprint, segment_offset):
            signals["structural_reset"] = 10.0

        if prev_row_index is not None and _datatype_transition(
            self.raw_df.iloc[prev_row_index], row
        ):
            signals["datatype_transition"] = 10.0

        total = sum(v for k, v in signals.items() if k != "signature_similarity")
        is_boundary = total >= self.threshold or (
            segment_offset == 0 and position_in_group > 0 and total >= 25
        )

        return RowBoundaryScore(
            row_index=row_index,
            score=total,
            is_boundary=is_boundary,
            segment_offset=segment_offset,
            signals=signals,
        )


class DynamicRecordGrouper:
    """Group raw rows into logical records using dynamic boundary detection."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        start_idx: int,
        blueprint: StructuralBlueprint,
        profile: Optional[DatasetProfile] = None,
        header_names: Optional[List[str]] = None,
    ):
        self.raw_df = raw_df
        self.start_idx = start_idx
        self.blueprint = blueprint
        self.profile = profile
        self.header_names = header_names or []

    def group_with_indices(self) -> Tuple[List[List[pd.Series]], List[List[int]]]:
        series_groups, index_groups = self._group_internal()
        return series_groups, index_groups

    def group(self) -> List[List[pd.Series]]:
        series_groups, _ = self._group_internal()
        return series_groups

    def _group_internal(self) -> Tuple[List[List[pd.Series]], List[List[int]]]:
        from .periodic_structure_detector import (
            PeriodicStructureDetector,
            compute_anchor_strength,
        )

        anchor_detector = AnchorDetector(
            self.raw_df, self.blueprint, self.header_names, self.start_idx
        )
        anchors = anchor_detector.detect()
        anchor_strength = compute_anchor_strength(
            self.raw_df, self.start_idx, self.blueprint, self.header_names
        )

        periodic_det = PeriodicStructureDetector(
            self.raw_df, self.profile, self.start_idx
        )
        periodic = periodic_det.detect(anchor_strength)

        data_row_indices = [
            i
            for i in range(self.start_idx, len(self.raw_df))
            if not self.raw_df.iloc[i].isna().all()
        ]

        if not data_row_indices:
            return [], []

        use_periodic = (
            periodic.cycle_length > 1
            and periodic.periodicity_score >= 0.45
            and periodic.segmentation_mode in ("periodic", "hybrid")
            and (
                periodic.periodicity_score > anchor_strength + 0.1
                or not anchors
                or self.blueprint.span <= 1
            )
        )

        if use_periodic:
            if periodic.cycle_length > self.blueprint.span:
                self.blueprint = periodic_det.build_blueprint(periodic)
            return periodic_det.group_rows(periodic)

        segment_templates = _build_segment_templates(
            self.raw_df, self.blueprint, self._header_row_indices()
        )

        if self.blueprint.span <= 1:
            return self._group_single_row_span_indexed(
                data_row_indices, anchors, periodic
            )

        scorer = BoundaryScorer(
            self.raw_df,
            self.blueprint,
            anchors,
            self.profile,
            segment_templates,
        )
        return _group_with_scorer_indexed(
            self.raw_df, data_row_indices, self.blueprint, anchors, scorer
        )

    def _group_single_row_span_indexed(
        self,
        data_row_indices: List[int],
        anchors: List[AnchorColumn],
        periodic=None,
    ) -> Tuple[List[List[pd.Series]], List[List[int]]]:
        """Anchor-based boundaries, or periodic cycles when no anchors."""
        if (
            periodic
            and periodic.cycle_length > 1
            and periodic.periodicity_score >= 0.4
            and not anchors
        ):
            from .periodic_structure_detector import PeriodicStructureDetector

            det = PeriodicStructureDetector(
                self.raw_df, self.profile, self.start_idx
            )
            return det.group_rows(periodic)

        groups: List[List[pd.Series]] = []
        index_groups: List[List[int]] = []
        current: List[pd.Series] = []
        current_idx: List[int] = []
        seen_anchor = False

        for row_idx in data_row_indices:
            row = self.raw_df.iloc[row_idx]
            has_anchor = row_has_anchor(row, anchors, self.blueprint)

            if has_anchor and seen_anchor and current:
                groups.append(current)
                index_groups.append(current_idx)
                current = []
                current_idx = []

            current.append(row)
            current_idx.append(row_idx)
            if has_anchor:
                seen_anchor = True

        if current:
            groups.append(current)
            index_groups.append(current_idx)

        if not groups and data_row_indices:
            return (
                [[self.raw_df.iloc[i]] for i in data_row_indices],
                [[i] for i in data_row_indices],
            )

        return groups, index_groups

    def _header_row_indices(self) -> List[int]:
        return list(range(min(self.start_idx, len(self.raw_df))))


def detect_boundaries(
    raw_df: pd.DataFrame,
    start_idx: int,
    blueprint: StructuralBlueprint,
    profile: Optional[DatasetProfile] = None,
    header_names: Optional[List[str]] = None,
) -> BoundaryDetectionResult:
    grouper = DynamicRecordGrouper(raw_df, start_idx, blueprint, profile, header_names)
    anchors = AnchorDetector(raw_df, blueprint, header_names, start_idx).detect()
    segment_templates = _build_segment_templates(
        raw_df, blueprint, list(range(start_idx))
    )
    scorer = BoundaryScorer(raw_df, blueprint, anchors, profile, segment_templates)

    data_indices = [
        i
        for i in range(start_idx, len(raw_df))
        if not raw_df.iloc[i].isna().all()
    ]

    row_scores: List[RowBoundaryScore] = []
    prev: Optional[int] = None
    pos_in_group = 0
    for row_idx in data_indices:
        score = scorer.score_row(row_idx, prev, pos_in_group)
        row_scores.append(score)
        if score.is_boundary and prev is not None:
            pos_in_group = 0
        else:
            pos_in_group += 1
        prev = row_idx

    _, record_groups = grouper.group_with_indices()
    return BoundaryDetectionResult(
        anchor_columns=anchors,
        row_scores=row_scores,
        record_groups=record_groups,
    )


def reconstruct_with_dynamic_boundaries(
    raw_df: pd.DataFrame,
    start_idx: int,
    blueprint: StructuralBlueprint,
    profile: Optional[DatasetProfile] = None,
    header_names: Optional[List[str]] = None,
) -> List[List[Any]]:
    grouper = DynamicRecordGrouper(raw_df, start_idx, blueprint, profile, header_names)
    groups = grouper.group()
    records = []
    for group in groups:
        flattened = BlueprintReconstructor.flatten_row_group(group, blueprint)
        if flattened:
            records.append(flattened)
    return records


def validate_record_coherence(
    headers: List[str], dataframe: pd.DataFrame
) -> Dict[str, Any]:
    """Validate logical record integrity after reconstruction."""
    if dataframe is None or dataframe.empty:
        return {
            "record_coherence_score": 0.0,
            "anchor_uniqueness": 0.0,
            "duplicate_anchor_rows": [],
            "row_coherence_scores": [],
        }

    headers = list(headers)[: len(dataframe.columns)]
    anchor_col_indices = _find_anchor_columns_from_headers(headers)

    row_scores = []
    for i in range(len(dataframe)):
        row_scores.append(score_row_alignment(headers, dataframe.iloc[i].tolist()))

    anchor_uniqueness = 1.0
    duplicate_anchors: List[Any] = []
    if anchor_col_indices:
        col = dataframe.columns[anchor_col_indices[0]]
        values = dataframe[col].dropna().astype(str).tolist()
        if values:
            anchor_uniqueness = len(set(values)) / len(values)
            from collections import Counter

            counts = Counter(values)
            duplicate_anchors = [v for v, c in counts.items() if c > 1]

    internal_scores = []
    for i in range(len(dataframe)):
        row = dataframe.iloc[i]
        score = _score_internal_record_coherence(headers, row, anchor_col_indices)
        internal_scores.append(score)

    record_coherence = 0.0
    if row_scores:
        record_coherence = (
            0.4 * (sum(row_scores) / len(row_scores))
            + 0.3 * anchor_uniqueness
            + 0.3 * (sum(internal_scores) / len(internal_scores) if internal_scores else 0)
        )

    return {
        "record_coherence_score": record_coherence,
        "anchor_uniqueness": anchor_uniqueness,
        "duplicate_anchor_rows": duplicate_anchors,
        "row_coherence_scores": row_scores,
        "per_record_coherence": internal_scores,
    }


def reconstruct_with_threshold(
    raw_df: pd.DataFrame,
    start_idx: int,
    blueprint: StructuralBlueprint,
    profile: Optional[DatasetProfile] = None,
    header_names: Optional[List[str]] = None,
    threshold: float = DEFAULT_BOUNDARY_THRESHOLD,
) -> List[List[Any]]:
    grouper = DynamicRecordGrouper(raw_df, start_idx, blueprint, profile, header_names)
    anchors = AnchorDetector(raw_df, blueprint, header_names, start_idx).detect()
    segment_templates = _build_segment_templates(
        raw_df, blueprint, list(range(start_idx))
    )
    data_row_indices = [
        i
        for i in range(start_idx, len(raw_df))
        if not raw_df.iloc[i].isna().all()
    ]
    if not data_row_indices:
        return []

    if blueprint.span <= 1:
        groups, _ = grouper._group_single_row_span_indexed(data_row_indices, anchors)
    else:
        scorer = BoundaryScorer(
            raw_df, blueprint, anchors, profile, segment_templates, threshold
        )
        groups, _ = _group_with_scorer_indexed(
            raw_df, data_row_indices, blueprint, anchors, scorer
        )

    records = []
    for group in groups:
        flattened = BlueprintReconstructor.flatten_row_group(group, blueprint)
        if flattened:
            records.append(flattened)
    return records


def _group_with_scorer_indexed(
    raw_df: pd.DataFrame,
    data_row_indices: List[int],
    blueprint: StructuralBlueprint,
    anchors: List[AnchorColumn],
    scorer: BoundaryScorer,
) -> Tuple[List[List[pd.Series]], List[List[int]]]:
    groups: List[List[pd.Series]] = []
    index_groups: List[List[int]] = []
    current: List[pd.Series] = []
    current_idx: List[int] = []
    prev_idx: Optional[int] = None
    seen_anchor_in_group = False

    for row_idx in data_row_indices:
        row = raw_df.iloc[row_idx]
        boundary = scorer.score_row(row_idx, prev_idx, len(current))

        expected_offset = len(current) % blueprint.span
        start_new = False
        if not current:
            start_new = False
        elif boundary.is_boundary and boundary.segment_offset == 0:
            start_new = True
        elif (
            boundary.segment_offset == 0
            and len(current) > 0
            and expected_offset != 0
        ):
            start_new = True
        elif row_has_anchor(row, anchors, blueprint) and seen_anchor_in_group:
            start_new = True
        elif len(current) >= blueprint.span * 2 and boundary.segment_offset == 0:
            start_new = True

        if start_new and current:
            groups.append(current)
            index_groups.append(current_idx)
            current = []
            current_idx = []
            seen_anchor_in_group = False

        current.append(row)
        current_idx.append(row_idx)
        if row_has_anchor(row, anchors, blueprint):
            seen_anchor_in_group = True
        prev_idx = row_idx

    if current:
        groups.append(current)
        index_groups.append(current_idx)
    return groups, index_groups


def repair_record_boundaries(
    raw_df: pd.DataFrame,
    start_idx: int,
    blueprint: StructuralBlueprint,
    profile: Optional[DatasetProfile] = None,
    header_names: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
) -> List[List[Any]]:
    """Retry grouping with alternate thresholds to reduce drift."""
    thresholds = thresholds or [45.0, 35.0, 55.0, 25.0]
    best_records: List[List[Any]] = []
    best_score = -1.0

    for t in thresholds:
        records = reconstruct_with_threshold(
            raw_df, start_idx, blueprint, profile, header_names, t
        )
        if not records:
            continue
        df = BlueprintReconstructor.records_to_dataframe(records)
        if df is None or not header_names:
            continue
        df.columns = header_names[: len(df.columns)]
        coherence = validate_record_coherence(header_names, df)
        score = coherence["record_coherence_score"]
        if score > best_score:
            best_score = score
            best_records = records

    return best_records if best_records else reconstruct_with_dynamic_boundaries(
        raw_df, start_idx, blueprint, profile, header_names
    )


# --- helpers ---


def _match_any_anchor_pattern(value: Any) -> Tuple[bool, str]:
    if pd.isna(value):
        return False, ""
    text = str(value).strip()
    if not text:
        return False, ""
    for pattern, ptype in ANCHOR_PATTERNS:
        if pattern.match(text):
            return True, ptype
    return False, ""


def _looks_like_id(value: Any) -> bool:
    try:
        n = float(str(value))
        return n == int(n) and n >= 0
    except (ValueError, TypeError):
        return False


def infer_segment_offset(
    row: pd.Series,
    blueprint: StructuralBlueprint,
    position_in_group: Optional[int] = None,
) -> int:
    if not blueprint or blueprint.span <= 1:
        return 0

    row_cols = {
        i
        for i, v in enumerate(row)
        if pd.notna(v) and str(v).strip() != ""
    }
    if not row_cols:
        return 0

    if position_in_group is not None and blueprint.span > 1:
        expected = position_in_group % blueprint.span
        expected_cols = set(blueprint.row_to_cols_map.get(expected, []))
        if expected_cols:
            overlap = len(row_cols & expected_cols)
            density = overlap / len(expected_cols)
            if overlap > 0 and density >= 0.5:
                return expected

    best_offset = 0
    best_overlap = -1.0
    for offset in range(blueprint.span):
        expected = set(blueprint.row_to_cols_map.get(offset, []))
        if not expected:
            continue
        overlap = len(row_cols & expected)
        density = overlap / len(expected)
        weighted = overlap + density * 2
        if weighted > best_overlap:
            best_overlap = weighted
            best_offset = offset
    return best_offset


def row_has_anchor(
    row: pd.Series,
    anchors: List[AnchorColumn],
    blueprint: StructuralBlueprint,
) -> bool:
    if not anchors:
        return False

    seg0_cols = set(blueprint.row_to_cols_map.get(0, []))
    for anchor in anchors:
        if anchor.segment_offset > 0 and anchor.column_index not in seg0_cols:
            continue
        col_idx = anchor.column_index
        if col_idx >= len(row):
            continue
        val = row.iloc[col_idx]
        if pd.isna(val):
            continue
        matched, _ = _match_any_anchor_pattern(val)
        if matched:
            return True
        if anchor.pattern_type == "generic_id" and _looks_like_id(val):
            return True
    return False


def _signature_similarity(sig_a: str, sig_b: str) -> float:
    if not sig_a or not sig_b:
        return 0.0
    n = max(len(sig_a), len(sig_b))
    if n == 0:
        return 1.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    length_penalty = 1.0 - abs(len(sig_a) - len(sig_b)) / n
    return (matches / n) * length_penalty


def _build_segment_templates(
    raw_df: pd.DataFrame,
    blueprint: StructuralBlueprint,
    header_row_indices: List[int],
) -> Dict[int, str]:
    templates: Dict[int, str] = {}
    sorted_headers = sorted(header_row_indices)
    for offset in range(blueprint.span):
        if offset < len(sorted_headers):
            row = raw_df.iloc[sorted_headers[offset]]
            templates[offset] = build_row_signature(row)
    return templates


def _is_structural_reset(
    row: pd.Series, blueprint: StructuralBlueprint, segment_offset: int
) -> bool:
    if segment_offset != 0:
        return False
    non_null = [i for i, v in enumerate(row) if pd.notna(v) and str(v).strip()]
    if not non_null:
        return False
    seg0 = blueprint.row_to_cols_map.get(0, [])
    if not seg0:
        return False
    return non_null[0] == seg0[0]


def _datatype_transition(prev_row: pd.Series, row: pd.Series) -> bool:
    prev_types = {classify_cell(v) for v in prev_row if pd.notna(v)}
    curr_types = {classify_cell(v) for v in row if pd.notna(v)}
    if not prev_types or not curr_types:
        return False
    return prev_types != curr_types and "N" in curr_types and "T" in prev_types


def _find_anchor_columns_from_headers(headers: List[str]) -> List[int]:
    indices = []
    for i, h in enumerate(headers):
        name = str(h).lower().replace(" ", "_")
        if any(kw in name for kw in ANCHOR_KEYWORDS):
            if "id" in name or "txn" in name or "transaction" in name or "uuid" in name:
                indices.append(i)
    return indices[:2] if indices else []


def _score_internal_record_coherence(
    headers: List[str], row: pd.Series, anchor_indices: List[int]
) -> float:
    scores = []
    for i, header in enumerate(headers):
        if i >= len(row):
            break
        expected = infer_header_semantic_type(header)
        actual = infer_value_semantic_type(row.iloc[i])
        scores.append(
            1.0
            if expected == actual or actual == "empty"
            else (0.5 if expected == "categorical" else 0.0)
        )
    if not scores:
        return 0.5
    base = sum(scores) / len(scores)
    if anchor_indices:
        idx = anchor_indices[0]
        if idx < len(row):
            val = row.iloc[idx]
            matched, _ = _match_any_anchor_pattern(val)
            if matched:
                base = min(1.0, base + 0.1)
    return base
