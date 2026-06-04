from typing import List, Optional

from .confidence import clamp, combine_weighted, score_to_percent
from .models import ReconstructionResult
from .validator import ValidationEngine


class ReconstructionScorer:
    """Layer 9 — Confidence-driven reconstruction scoring across all hypotheses."""

    SCORE_WEIGHTS = {
        "semantic_alignment": 0.22,
        "record_coherence": 0.18,
        "signature_consistency": 0.14,
        "row_completeness": 0.12,
        "null_stability": 0.10,
        "datatype_consistency": 0.12,
        "anchor_continuity": 0.07,
        "anomaly_penalty": 0.05,
    }

    def __init__(self, results: List[ReconstructionResult], profile=None):
        self.results = results
        self.profile = profile

    def score_all(self):
        for result in self.results:
            if result.dataframe is None or result.dataframe.empty:
                result.validation_score = 0.0
                result.confidence_score = 0.0
                result.structural_stability_score = 0.0
                continue

            validator = ValidationEngine(result.dataframe)
            report = validator.validate()
            result.validation_report = report

            signals = self._build_signals(report, result)
            confidence_0_1 = combine_weighted(signals, self.SCORE_WEIGHTS)
            result.structural_stability_score = signals.get("signature_consistency", 0.0)
            result.validation_score = score_to_percent(confidence_0_1)
            result.confidence_score = result.validation_score
            result.validation_report["confidence_signals"] = signals
            result.validation_report["global_confidence"] = confidence_0_1

    def _build_signals(self, report: dict, result: ReconstructionResult) -> dict:
        null_ratio = report.get("null_ratio", 1.0)
        col_count = max(report.get("col_count", 1), 1)
        row_count = report.get("row_count", 0)

        signals = {
            "semantic_alignment": report.get("semantic_alignment_score", 0.0),
            "record_coherence": report.get("record_coherence_score", 0.0),
            "signature_consistency": report.get("signature_consistency", 0.0),
            "row_completeness": clamp(1.0 - null_ratio * 0.85),
            "null_stability": clamp(1.0 - null_ratio),
            "datatype_consistency": clamp(
                report.get("numeric_col_count", 0) / col_count
                if col_count
                else 0.5
            ),
            "anchor_continuity": report.get("anchor_uniqueness", 1.0),
            "anomaly_penalty": clamp(
                1.0 - len(report.get("duplicate_anchor_rows", [])) * 0.15
            ),
        }

        if self.profile:
            signals["profile_fit"] = clamp(
                0.5 * self.profile.segmentation_likelihood
                + 0.5 * self.profile.row_similarity_stability
            )
            if "Segmented" in result.strategy_name:
                signals["strategy_fit"] = self.profile.segmentation_likelihood
            elif "Sparse" in result.strategy_name or "Positional" in result.strategy_name:
                signals["strategy_fit"] = clamp(1.0 - self.profile.sparsity_score * 0.5)
            elif "Streaming" in result.strategy_name:
                signals["strategy_fit"] = clamp(self.profile.null_ratio)
            else:
                signals["strategy_fit"] = 0.5

            if self.profile.segmentation_likelihood > 0.55:
                rep = result.validation_report
                if rep.get("row_count", 0) > 0:
                    col_count = rep.get("col_count", 1)
                    expected_dense = self.profile.total_columns
                    if col_count >= expected_dense * 0.5:
                        signals["periodic_completeness"] = clamp(
                            self.profile.segmentation_likelihood
                        )

        if null_ratio < 0.15:
            signals["density_bonus"] = 0.1
        if signals["semantic_alignment"] < 0.45:
            signals["semantic_alignment"] *= 0.5

        return {k: clamp(v) for k, v in signals.items()}

    def get_best_reconstruction(self) -> Optional[ReconstructionResult]:
        if not self.results:
            return None

        valid = [
            r for r in self.results if r.dataframe is not None and not r.dataframe.empty
        ]
        if not valid:
            return None

        self.score_all()
        valid.sort(key=lambda x: x.confidence_score, reverse=True)
        return valid[0]
