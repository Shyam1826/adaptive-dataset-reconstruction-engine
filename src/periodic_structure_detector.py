"""Periodic row-cycle detection for dense numeric / scientific segmented datasets."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .confidence import clamp, combine_weighted
from .models import (
    DatasetProfile,
    StructuralBlueprint,
    BlueprintSegment,
    PeriodicStructure,
)
from .row_signature import build_row_signature, detect_repeating_patterns


@dataclass
class RowSignatureInfo:
    row_index: int
    signature: str
    width: int
    non_null_cols: List[int]


class PeriodicStructureDetector:
    """Detect repeating structural row cycles in the data region."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        profile: Optional[DatasetProfile] = None,
        start_idx: int = 0,
    ):
        self.raw_df = raw_df
        self.profile = profile
        self.start_idx = start_idx

    def detect(self, anchor_strength: float = 0.0) -> PeriodicStructure:
        row_infos = self._collect_row_signatures()
        if len(row_infos) < 4:
            return PeriodicStructure(0, [], [], 0.0, anchor_strength=anchor_strength)

        signatures = [r.signature for r in row_infos]
        widths = [r.width for r in row_infos]

        cycle_len, score, template_sigs, template_widths, signals = (
            self._find_best_cycle(signatures, widths)
        )

        if cycle_len <= 1 or score < 0.45:
            span, pattern = detect_repeating_patterns(signatures)
            if span > 1 and pattern:
                cycle_len = span
                template_sigs = pattern
                template_widths = widths[:span]
                score = max(score, self.profile.segmentation_likelihood if self.profile else 0.5)
                signals["profiler_pattern"] = score

        mode = self._choose_mode(score, anchor_strength)

        return PeriodicStructure(
            cycle_length=cycle_len,
            segment_signatures=template_sigs,
            segment_widths=template_widths,
            periodicity_score=clamp(score),
            segmentation_mode=mode,
            signals=signals,
            anchor_strength=anchor_strength,
        )

    def group_rows(
        self, periodic: PeriodicStructure
    ) -> Tuple[List[List[pd.Series]], List[List[int]]]:
        """Group data rows into logical records by detected cycle length."""
        row_infos = self._collect_row_signatures()
        if not row_infos or periodic.cycle_length <= 1:
            return self._one_row_groups(row_infos)

        cycle = periodic.cycle_length
        groups: List[List[pd.Series]] = []
        index_groups: List[List[int]] = []

        current: List[pd.Series] = []
        current_idx: List[int] = []

        for info in row_infos:
            row = self.raw_df.iloc[info.row_index]
            pos = len(current)

            if pos > 0 and pos % cycle == 0:
                groups.append(current)
                index_groups.append(current_idx)
                current = []
                current_idx = []

            current.append(row)
            current_idx.append(info.row_index)

        if current:
            groups.append(current)
            index_groups.append(current_idx)

        return groups, index_groups

    def build_blueprint(self, periodic: PeriodicStructure) -> StructuralBlueprint:
        """Build a multi-row blueprint from the first detected cycle in data."""
        row_infos = self._collect_row_signatures()
        if not row_infos or periodic.cycle_length <= 0:
            return StructuralBlueprint({}, 0, [], 0)

        cycle = min(periodic.cycle_length, len(row_infos))
        row_to_cols_map: Dict[int, List[int]] = {}
        segments: List[BlueprintSegment] = []
        global_offset = 0

        for offset in range(cycle):
            info = row_infos[offset]
            valid_cols = info.non_null_cols
            row_to_cols_map[offset] = valid_cols
            width = len(valid_cols)
            if width > 0:
                segments.append(
                    BlueprintSegment(
                        segment_id=offset,
                        row_offset=offset,
                        column_indices=valid_cols,
                        width=width,
                        global_start=global_offset,
                        global_end=global_offset + width - 1,
                    )
                )
                global_offset += width

        return StructuralBlueprint(
            row_to_cols_map=row_to_cols_map,
            span=cycle,
            segments=segments,
            total_columns=global_offset,
            cycle_length=cycle,
            signature_sequence=periodic.segment_signatures[:cycle],
        )

    def _collect_row_signatures(self) -> List[RowSignatureInfo]:
        infos: List[RowSignatureInfo] = []
        for idx in range(self.start_idx, len(self.raw_df)):
            row = self.raw_df.iloc[idx]
            if row.isna().all():
                continue
            sig = build_row_signature(row)
            sig_compact = sig.rstrip("E") or sig
            non_null_cols = [
                i
                for i, v in enumerate(row)
                if pd.notna(v) and str(v).strip() != ""
            ]
            infos.append(
                RowSignatureInfo(
                    row_index=idx,
                    signature=sig_compact,
                    width=len(non_null_cols),
                    non_null_cols=non_null_cols,
                )
            )
        return infos

    def _find_best_cycle(
        self, signatures: List[str], widths: List[int]
    ) -> Tuple[int, float, List[str], List[int], Dict[str, float]]:
        best_len = 0
        best_score = 0.0
        best_sigs: List[str] = []
        best_widths: List[int] = []
        best_signals: Dict[str, float] = {}

        max_cycle = min(12, len(signatures) // 2)
        for cycle_len in range(2, max_cycle + 1):
            if len(signatures) < cycle_len * 2:
                continue

            template = tuple(signatures[:cycle_len])
            template_w = widths[:cycle_len]
            matches = 0
            blocks = 0
            width_matches = 0

            for i in range(0, len(signatures) - cycle_len + 1, cycle_len):
                block = tuple(signatures[i : i + cycle_len])
                block_w = widths[i : i + cycle_len]
                blocks += 1
                if _block_signature_match(block, template):
                    matches += 1
                if _block_width_match(block_w, template_w):
                    width_matches += 1

            if blocks == 0:
                continue

            sig_ratio = matches / blocks
            width_ratio = width_matches / blocks
            cyclic_sim = _cyclic_similarity(signatures, cycle_len)

            score = combine_weighted(
                {
                    "signature_repeat": sig_ratio,
                    "width_repeat": width_ratio,
                    "cyclic_sim": cyclic_sim,
                },
                {
                    "signature_repeat": 0.45,
                    "width_repeat": 0.30,
                    "cyclic_sim": 0.25,
                },
            )

            if score > best_score:
                best_score = score
                best_len = cycle_len
                best_sigs = list(template)
                best_widths = list(template_w)
                best_signals = {
                    "signature_repeat": sig_ratio,
                    "width_repeat": width_ratio,
                    "cyclic_sim": cyclic_sim,
                }

        return best_len, best_score, best_sigs, best_widths, best_signals

    def _choose_mode(self, periodicity_score: float, anchor_strength: float) -> str:
        if periodicity_score >= 0.55 and periodicity_score > anchor_strength + 0.15:
            return "periodic"
        if anchor_strength >= 0.5 and periodicity_score < 0.45:
            return "anchor"
        if periodicity_score >= 0.4 and anchor_strength >= 0.4:
            return "hybrid"
        return "periodic" if periodicity_score >= anchor_strength else "anchor"

    def _one_row_groups(
        self, row_infos: List[RowSignatureInfo]
    ) -> Tuple[List[List[pd.Series]], List[List[int]]]:
        groups = [[self.raw_df.iloc[i.row_index]] for i in row_infos]
        indices = [[i.row_index] for i in row_infos]
        return groups, indices


def _block_signature_match(block: Tuple[str, ...], template: Tuple[str, ...]) -> bool:
    if len(block) != len(template):
        return False
    sims = [_signature_similarity(a, b) for a, b in zip(block, template)]
    return sum(sims) / len(sims) >= 0.72


def _block_width_match(block: Tuple[int, ...], template: Tuple[int, ...]) -> bool:
    if len(block) != len(template):
        return False
    return all(abs(a - b) <= 1 for a, b in zip(block, template))


def _signature_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    n = max(len(a), len(b))
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / n


def _cyclic_similarity(signatures: List[str], cycle_len: int) -> float:
    if cycle_len <= 0 or len(signatures) < cycle_len * 2:
        return 0.0
    offsets = []
    for offset in range(cycle_len):
        chars = [signatures[i] for i in range(offset, len(signatures), cycle_len)]
        if len(chars) < 2:
            continue
        sims = [
            _signature_similarity(chars[i], chars[i - 1])
            for i in range(1, len(chars))
        ]
        offsets.append(sum(sims) / len(sims))
    return sum(offsets) / len(offsets) if offsets else 0.0


def compute_anchor_strength(
    raw_df: pd.DataFrame,
    start_idx: int,
    blueprint: Optional[StructuralBlueprint],
    header_names: Optional[List[str]] = None,
) -> float:
    from .boundary_detector import AnchorDetector

    if not blueprint:
        return 0.0
    anchors = AnchorDetector(raw_df, blueprint, header_names, start_idx).detect()
    if not anchors:
        return 0.0
    return clamp(max(a.confidence for a in anchors))


def validate_periodic_reconstruction(
    records: List[List[Any]],
    periodic: PeriodicStructure,
    expected_columns: Optional[int] = None,
) -> Dict[str, Any]:
    if not records:
        return {"periodic_valid": False, "completeness": 0.0}

    lengths = [len(r) for r in records]
    expected_len = periodic.cycle_length and sum(periodic.segment_widths)
    if expected_len <= 0 and expected_columns:
        expected_len = expected_columns

    completeness_scores = []
    for rec in records:
        non_null = sum(1 for v in rec if v is not None and str(v).strip() != "")
        if expected_len > 0:
            completeness_scores.append(min(1.0, non_null / expected_len))
        else:
            completeness_scores.append(non_null / max(len(rec), 1))

    length_stability = 1.0
    if len(set(lengths)) == 1:
        length_stability = 1.0
    elif lengths:
        most = Counter(lengths).most_common(1)[0][0]
        length_stability = lengths.count(most) / len(lengths)

    return {
        "periodic_valid": periodic.periodicity_score >= 0.45,
        "completeness": sum(completeness_scores) / len(completeness_scores),
        "length_stability": length_stability,
        "periodicity_score": periodic.periodicity_score,
        "cycle_length": periodic.cycle_length,
    }


def enhance_header_info_blueprint(
    raw_df: pd.DataFrame,
    header_info,
    profile: Optional[DatasetProfile] = None,
):
    """Augment blueprint from periodic data cycles when headers are single-row."""
    from .models import HeaderInfo

    if not isinstance(header_info, HeaderInfo):
        return header_info

    start = header_info.data_start_row or (
        (max(header_info.header_rows) + 1) if header_info.header_rows else 0
    )
    anchor_strength = compute_anchor_strength(
        raw_df, start, header_info.blueprint, None
    )
    detector = PeriodicStructureDetector(raw_df, profile, start)
    periodic = detector.detect(anchor_strength)

    if periodic.cycle_length <= 1 or periodic.periodicity_score < 0.45:
        return header_info

    header_span = header_info.blueprint.span if header_info.blueprint else 1
    use_periodic = (
        periodic.segmentation_mode in ("periodic", "hybrid")
        and periodic.cycle_length > header_span
        and periodic.periodicity_score >= 0.45
    )

    if use_periodic:
        periodic_bp = detector.build_blueprint(periodic)
        header_info.blueprint = periodic_bp
        header_info.schema_width = periodic_bp.total_columns
        header_info.header_span = max(header_info.header_span, periodic.cycle_length)

    header_info.periodic_structure = periodic
    return header_info
