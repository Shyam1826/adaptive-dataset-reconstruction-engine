"""Layer 10 — Self-correction loop for low-confidence reconstructions."""

from typing import List, Optional, Tuple

import pandas as pd

from .alignment_optimizer import optimize_column_alignment
from .boundary_detector import repair_record_boundaries
from .confidence import ConfidenceReport, clamp
from .header_hypothesis import HeaderHypothesisEngine
from .header_detector import HeaderDetector
from .header_alignment import HeaderAligner
from .models import DatasetProfile, HeaderInfo, ReconstructionResult
from .scorer import ReconstructionScorer
from .structure_blueprint import BlueprintReconstructor, validate_semantic_alignment
from .boundary_detector import validate_record_coherence
from .strategies.segmented import SegmentedBlockStrategy
from .strategies.positional import PositionalStrategy
from .strategies.streaming import StreamingStrategy
from .strategies.hybrid import HybridStrategy
from .strategies.sparse import SparseAlignmentStrategy


CORRECTION_THRESHOLD = 0.62
MAX_ITERATIONS = 2

STRATEGIES = [
    ("SegmentedBlockStrategy", SegmentedBlockStrategy),
    ("PositionalStrategy", PositionalStrategy),
    ("StreamingStrategy", StreamingStrategy),
    ("SparseAlignmentStrategy", SparseAlignmentStrategy),
    ("HybridStrategy", HybridStrategy),
]


class SelfCorrectionEngine:
    """Retry alternative hypotheses when global confidence is low."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        profile: DatasetProfile,
        header_info: HeaderInfo,
    ):
        self.raw_df = raw_df
        self.profile = profile
        self.header_info = header_info

    def improve(
        self,
        result: ReconstructionResult,
        confidence_report: ConfidenceReport,
    ) -> Tuple[ReconstructionResult, ConfidenceReport]:
        if result is None or result.dataframe is None:
            return result, confidence_report

        global_conf = confidence_report.global_confidence
        if global_conf >= CORRECTION_THRESHOLD:
            return result, confidence_report

        best = result
        best_conf = global_conf

        for iteration in range(MAX_ITERATIONS):
            candidates: List[ReconstructionResult] = [best]

            alt_header = self._try_header_hypotheses()
            if alt_header:
                candidates.extend(self._reconstruct_all(alt_header))

            boundary_fixed = self._try_boundary_repair(best)
            if boundary_fixed is not None:
                candidates.append(boundary_fixed)

            align_fixed = self._try_alignment_optimization(best)
            if align_fixed is not None:
                candidates.append(align_fixed)

            valid = [c for c in candidates if c and c.dataframe is not None and not c.dataframe.empty]
            if not valid:
                break

            scorer = ReconstructionScorer(valid)
            improved = scorer.get_best_reconstruction()
            if improved is None:
                break

            improved_conf = improved.confidence_score / 100.0
            if improved_conf > best_conf:
                best = improved
                best_conf = improved_conf
                confidence_report.add_layer(
                    f"correction_{iteration}",
                    improved_conf,
                    {"strategy": improved.strategy_name},
                )
            else:
                break

        return best, confidence_report

    def _try_header_hypotheses(self) -> Optional[HeaderInfo]:
        detector = HeaderDetector(self.raw_df, self.profile)
        confidences = [
            detector._calculate_row_confidence(i)
            for i in range(self.profile.total_rows)
        ]
        engine = HeaderHypothesisEngine(self.raw_df, self.profile)
        scored = engine.evaluate(confidences)
        if scored.confidence > 0.2 and scored.value.header_rows != self.header_info.header_rows:
            return scored.value
        return None

    def _reconstruct_all(self, header_info: HeaderInfo) -> List[ReconstructionResult]:
        results = []
        for name, StrategyClass in STRATEGIES:
            df = StrategyClass(self.raw_df, self.profile, header_info).reconstruct()
            if df is not None:
                results.append(
                    ReconstructionResult(
                        dataframe=df,
                        strategy_name=f"{name}_corrected",
                        confidence_score=0.0,
                        validation_score=0.0,
                        validation_report={},
                    )
                )
        return results

    def _try_boundary_repair(
        self, result: ReconstructionResult
    ) -> Optional[ReconstructionResult]:
        if not self.header_info.blueprint or not self.header_info.header_rows:
            return None
        headers = list(result.dataframe.columns)
        start = self.header_info.data_start_row or (
            max(self.header_info.header_rows) + 1
        )
        repaired = repair_record_boundaries(
            self.raw_df,
            start,
            self.header_info.blueprint,
            self.profile,
            headers,
        )
        if not repaired:
            return None
        df = BlueprintReconstructor.records_to_dataframe(repaired)
        if df is None:
            return None
        df.columns = headers[: len(df.columns)]
        aligner = HeaderAligner(self.raw_df, self.header_info)
        df = aligner.align(df, auto_realign=True)
        return ReconstructionResult(
            dataframe=df,
            strategy_name=f"{result.strategy_name}_boundary_repair",
            confidence_score=0.0,
            validation_score=0.0,
            validation_report={},
        )

    def _try_alignment_optimization(
        self, result: ReconstructionResult
    ) -> Optional[ReconstructionResult]:
        headers = list(result.dataframe.columns)
        optimized, score, shift, _ = optimize_column_alignment(
            headers, result.dataframe
        )
        if shift == 0:
            return None
        return ReconstructionResult(
            dataframe=optimized,
            strategy_name=f"{result.strategy_name}_align_opt",
            confidence_score=score * 100,
            validation_score=0.0,
            validation_report={"alignment_shift": shift, "semantic_score": score},
        )
