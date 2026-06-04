"""Multi-hypothesis header boundary inference with confidence scoring."""

import pandas as pd
from typing import List, Optional

from .confidence import combine_weighted, clamp, normalize_linear
from .header_boundary_detector import HeaderBoundaryDetector, trim_header_candidates
from .hypothesis import HypothesisSelector, ScoredHypothesis
from .models import DatasetProfile, HeaderInfo
from .structure_blueprint import BlueprintGenerator


class HeaderHypothesisEngine:
    """Evaluate multiple header-span / data-start hypotheses and pick the best."""

    def __init__(self, raw_df: pd.DataFrame, profile: DatasetProfile):
        self.raw_df = raw_df
        self.profile = profile

    def evaluate(
        self,
        row_confidences: List[float],
    ) -> ScoredHypothesis[HeaderInfo]:
        boundary = HeaderBoundaryDetector(
            self.raw_df, self.profile, row_confidences
        ).detect()

        hypotheses: List[ScoredHypothesis[HeaderInfo]] = []

        primary = self._build_header_info(
            boundary.data_start_row,
            trim_header_candidates(
                self._candidate_rows_from_top(row_confidences),
                boundary,
            ),
            row_confidences,
            boundary.boundary_confidence,
            "primary_boundary",
        )
        if primary:
            hypotheses.append(primary)

        max_scan = min(12, self.profile.total_rows)
        for data_start in range(0, max_scan):
            if data_start == boundary.data_start_row and primary:
                continue
            header_rows = list(range(data_start))
            header_rows = [
                r
                for r in header_rows
                if not self.raw_df.iloc[r].isna().all()
            ]
            if not header_rows and data_start > 0:
                continue
            info = self._build_header_info(
                data_start,
                header_rows,
                row_confidences,
                self._score_hypothesis(data_start, header_rows, row_confidences),
                f"data_start_{data_start}",
            )
            if info:
                hypotheses.append(info)

        best = HypothesisSelector.select_best(hypotheses, min_confidence=0.15)
        if best:
            return best

        return ScoredHypothesis(
            name="empty",
            value=HeaderInfo([], 0, self.profile.total_columns, 0.0, data_start_row=0),
            confidence=0.0,
        )

    def _score_hypothesis(
        self,
        data_start: int,
        header_rows: List[int],
        row_confidences: List[float],
    ) -> float:
        n = self.profile.total_rows
        if data_start > n:
            return 0.0

        signals: dict = {}

        if header_rows:
            avg_header = sum(row_confidences[r] for r in header_rows) / len(
                header_rows
            )
            signals["header_row_confidence"] = normalize_linear(avg_header, 0, 100)
        else:
            signals["header_row_confidence"] = 0.3 if data_start == 0 else 0.0

        boundary_det = HeaderBoundaryDetector(
            self.raw_df, self.profile, row_confidences
        )
        if data_start < n:
            ds_score, _ = boundary_det._data_start_score(data_start)
            signals["data_start_strength"] = normalize_linear(ds_score, 0, 120)
        else:
            signals["data_start_strength"] = 0.0

        if data_start > 0 and data_start < n:
            label_penalty = 0.0
            for r in range(data_start, min(data_start + 3, n)):
                if boundary_det._row_looks_like_label_header(self.raw_df.iloc[r]):
                    label_penalty += 0.3
            signals["no_label_in_data"] = clamp(1.0 - label_penalty)
        else:
            signals["no_label_in_data"] = 0.8

        if self.profile.repeating_signature_span > 0 and data_start < n:
            remaining = [
                self.profile.row_profiles[i].signature
                for i in range(data_start, n)
                if not self.raw_df.iloc[i].isna().all()
            ]
            if len(remaining) >= self.profile.repeating_signature_span * 2:
                signals["periodicity_fit"] = clamp(
                    self.profile.segmentation_likelihood
                    if hasattr(self.profile, "segmentation_likelihood")
                    else 0.6
                )
            else:
                signals["periodicity_fit"] = 0.4
        else:
            signals["periodicity_fit"] = 0.5

        signals["span_plausibility"] = clamp(
            1.0 - abs(len(header_rows) - 1) * 0.08
            if len(header_rows) <= 5
            else 0.3
        )

        return combine_weighted(
            signals,
            {
                "header_row_confidence": 0.25,
                "data_start_strength": 0.30,
                "no_label_in_data": 0.20,
                "periodicity_fit": 0.15,
                "span_plausibility": 0.10,
            },
        )

    def _build_header_info(
        self,
        data_start: int,
        header_rows: List[int],
        row_confidences: List[float],
        confidence: float,
        name: str,
    ) -> Optional[ScoredHypothesis[HeaderInfo]]:
        if header_rows:
            data_start = max(data_start, max(header_rows) + 1)

        if not header_rows:
            return ScoredHypothesis(
                name=name,
                value=HeaderInfo(
                    [],
                    0,
                    self.profile.total_columns,
                    max(row_confidences) if row_confidences else 0.0,
                    data_start_row=data_start,
                ),
                confidence=confidence * 0.5,
            )

        blueprint = BlueprintGenerator.generate_from_headers(
            self.raw_df, header_rows
        )
        schema_width = (
            blueprint.total_columns
            if blueprint.total_columns > 0
            else self.profile.total_columns
        )
        info = HeaderInfo(
            header_rows=header_rows,
            header_span=len(header_rows),
            schema_width=schema_width,
            confidence=max(row_confidences[r] for r in header_rows),
            blueprint=blueprint,
            data_start_row=data_start,
        )
        return ScoredHypothesis(name=name, value=info, confidence=confidence)

    def _candidate_rows_from_top(self, confidences: List[float]) -> List[int]:
        rows = []
        for i, c in enumerate(confidences):
            if c >= 35.0:
                rows.append(i)
            elif rows and i == rows[-1] + 1:
                break
        return rows if rows else ([0] if confidences and confidences[0] >= 25 else [])
