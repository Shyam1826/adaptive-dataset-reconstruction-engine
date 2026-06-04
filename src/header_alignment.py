import pandas as pd
from typing import List

from .models import HeaderInfo
from .alignment_optimizer import optimize_column_alignment
from .structure_blueprint import (
    BlueprintReconstructor,
    validate_semantic_alignment,
)


class HeaderAligner:
    """Layer 7-8 — Header alignment with optimization-based semantic realignment."""

    def __init__(self, raw_df: pd.DataFrame, header_info: HeaderInfo):
        self.raw_df = raw_df
        self.header_info = header_info

    def align(
        self, reconstructed_df: pd.DataFrame, *, auto_realign: bool = True
    ) -> pd.DataFrame:
        if reconstructed_df is None or reconstructed_df.empty:
            return reconstructed_df

        extracted_headers = self._extract_merged_headers()
        target_len = len(reconstructed_df.columns)

        final_headers = []
        for i in range(target_len):
            if i < len(extracted_headers):
                final_headers.append(extracted_headers[i])
            else:
                final_headers.append(f"unknown_col_{i + 1}")

        final_headers = self._deduplicate_headers(final_headers)
        reconstructed_df = reconstructed_df.copy()
        reconstructed_df.columns = final_headers[: len(reconstructed_df.columns)]

        if auto_realign:
            headers = list(reconstructed_df.columns)
            alignment = validate_semantic_alignment(headers, reconstructed_df)
            should_try = (
                alignment["semantic_alignment_score"] < 0.88
                or len(alignment.get("misaligned_columns", [])) > 0
            )
            if should_try:
                optimized, new_score, shift, _ = optimize_column_alignment(
                    headers, reconstructed_df
                )
                if shift != 0 and new_score > alignment["semantic_alignment_score"]:
                    reconstructed_df = optimized

        return reconstructed_df

    def _extract_merged_headers(self) -> List[str]:
        if not self.header_info.header_rows or not self.header_info.blueprint:
            return []

        return BlueprintReconstructor.extract_headers(
            self.raw_df,
            self.header_info.header_rows,
            self.header_info.blueprint,
        )

    def _deduplicate_headers(self, headers: List[str]) -> List[str]:
        seen = {}
        deduped = []
        for h in headers:
            if h in seen:
                seen[h] += 1
                deduped.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                deduped.append(h)
        return deduped
