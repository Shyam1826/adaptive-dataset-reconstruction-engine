import pandas as pd
from typing import List, Optional

from .models import ReconstructionResult
from .profiler import DatasetProfiler
from .header_detector import HeaderDetector
from .confidence import ConfidenceReport
from .strategies.segmented import SegmentedBlockStrategy
from .strategies.positional import PositionalStrategy
from .strategies.streaming import StreamingStrategy
from .strategies.sparse import SparseAlignmentStrategy
from .strategies.hybrid import HybridStrategy
from .scorer import ReconstructionScorer
from .self_correction import SelfCorrectionEngine


class AdaptiveReconstructionEngine:
    """Confidence-driven adaptive semi-structured table reconstruction."""

    STRATEGIES = [
        ("SegmentedBlockStrategy", SegmentedBlockStrategy),
        ("PositionalStrategy", PositionalStrategy),
        ("StreamingStrategy", StreamingStrategy),
        ("SparseAlignmentStrategy", SparseAlignmentStrategy),
        ("HybridStrategy", HybridStrategy),
    ]

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df
        self.confidence_report = ConfidenceReport()

    def run(self) -> Optional[ReconstructionResult]:
        # Layer 1 — Profiling
        profile = DatasetProfiler(self.raw_df).profile()
        self.confidence_report.add_layer("profiling", profile.profile_confidence)

        # Layer 3 — Header hypotheses (boundary-aware)
        detector = HeaderDetector(self.raw_df, profile)
        header_info = detector.detect()
        self.confidence_report.layers.update(detector.confidence_report.layers)

        from .periodic_structure_detector import enhance_header_info_blueprint

        header_info = enhance_header_info_blueprint(
            self.raw_df, header_info, profile
        )
        if header_info.periodic_structure:
            self.confidence_report.add_layer(
                "periodic",
                header_info.periodic_structure.periodicity_score,
                header_info.periodic_structure.signals,
            )

        if header_info.blueprint:
            self.confidence_report.add_layer(
                "blueprint",
                min(1.0, header_info.hypothesis_confidence + 0.1),
                {"total_columns": header_info.blueprint.total_columns},
            )

        # Layer 5 — Multi-strategy reconstruction hypotheses
        results: List[ReconstructionResult] = []
        for name, StrategyClass in self.STRATEGIES:
            strategy = StrategyClass(self.raw_df, profile, header_info)
            recon_df = strategy.reconstruct()
            if recon_df is not None and not recon_df.empty:
                results.append(
                    ReconstructionResult(
                        dataframe=recon_df,
                        strategy_name=name,
                        confidence_score=0.0,
                        validation_score=0.0,
                        validation_report={},
                    )
                )

        if not results:
            return None

        # Layer 9 — Score all hypotheses, select best
        scorer = ReconstructionScorer(results, profile)
        best_result = scorer.get_best_reconstruction()
        if best_result is None:
            return None

        self.confidence_report.add_layer(
            "reconstruction",
            best_result.confidence_score / 100.0,
            {"strategy": best_result.strategy_name},
        )

        # Layers 7-8 — Header alignment + semantic optimization
        best_result = self._finalize(best_result, header_info, profile)

        # Layer 10 — Self-correction if confidence low
        corrector = SelfCorrectionEngine(self.raw_df, profile, header_info)
        best_result, self.confidence_report = corrector.improve(
            best_result, self.confidence_report
        )

        best_result.validation_report["pipeline_confidence"] = (
            self.confidence_report.to_dict()
        )
        return best_result

    def _finalize(self, best_result: ReconstructionResult, header_info, profile):
        from .header_alignment import HeaderAligner
        from .structure_blueprint import validate_semantic_alignment
        from .boundary_detector import (
            repair_record_boundaries,
            validate_record_coherence,
        )
        from .structure_blueprint import BlueprintReconstructor

        aligner = HeaderAligner(self.raw_df, header_info)
        best_result.dataframe = aligner.align(best_result.dataframe)

        headers = list(best_result.dataframe.columns)
        coherence = validate_record_coherence(headers, best_result.dataframe)

        if (
            coherence.get("record_coherence_score", 1.0) < 0.68
            and header_info.blueprint
            and header_info.header_rows
        ):
            start_idx = header_info.data_start_row or (
                max(header_info.header_rows) + 1
            )
            repaired = repair_record_boundaries(
                self.raw_df,
                start_idx,
                header_info.blueprint,
                profile,
                headers,
            )
            if repaired:
                repaired_df = BlueprintReconstructor.records_to_dataframe(repaired)
                if repaired_df is not None:
                    repaired_df.columns = headers[: len(repaired_df.columns)]
                    new_coh = validate_record_coherence(headers, repaired_df)
                    if new_coh.get("record_coherence_score", 0) > coherence.get(
                        "record_coherence_score", 0
                    ):
                        best_result.dataframe = repaired_df
                        coherence = new_coh

        alignment = validate_semantic_alignment(headers, best_result.dataframe)
        best_result.validation_report.update(alignment)
        best_result.validation_report.update(coherence)

        self.confidence_report.add_layer(
            "semantic", alignment.get("semantic_alignment_score", 0.0)
        )
        self.confidence_report.add_layer(
            "boundary", coherence.get("record_coherence_score", 0.0)
        )

        return best_result

    def export(self, result: ReconstructionResult, output_path: str):
        if result and result.dataframe is not None:
            result.dataframe.to_csv(output_path, index=False)
            print(f"Exported clean dataset to {output_path}")
