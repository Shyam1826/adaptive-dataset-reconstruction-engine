"""Strategy E — Sparse alignment reconstruction for overlapping sparse rows."""

import pandas as pd
from typing import Optional

from .base import BaseStrategy
from ..structure_blueprint import BlueprintReconstructor


class SparseAlignmentStrategy(BaseStrategy):
    """Merges non-overlapping sparse rows; uses blueprint when structure is clear."""

    def reconstruct(self) -> Optional[pd.DataFrame]:
        start_idx = self.get_data_start_idx()
        if start_idx >= self.profile.total_rows:
            return None

        blueprint = self.header_info.blueprint
        if (
            blueprint
            and blueprint.span >= 1
            and blueprint.total_columns > 0
            and self.profile.sparsity_score > 0.25
        ):
            header_names = None
            if self.header_info.header_rows:
                header_names = BlueprintReconstructor.extract_headers(
                    self.raw_df,
                    self.header_info.header_rows,
                    blueprint,
                )
            records = BlueprintReconstructor.reconstruct_records(
                self.raw_df,
                start_idx,
                blueprint,
                self.profile,
                header_names,
                dynamic=True,
            )
            result = BlueprintReconstructor.records_to_dataframe(records)
            if result is not None and not result.empty:
                null_ratio = result.isna().sum().sum() / max(result.size, 1)
                if null_ratio < 0.5:
                    return result

        return self._sparse_merge(start_idx)

    def _sparse_merge(self, start_idx: int) -> Optional[pd.DataFrame]:
        data = self.raw_df.iloc[start_idx:].copy()
        merged_rows = []
        current_merged = None
        current_mask = None

        def overlaps(m1, m2):
            return any(a and b for a, b in zip(m1, m2))

        for _, row in data.iterrows():
            mask = row.notna().tolist()
            if not any(mask):
                continue
            if current_merged is None:
                current_merged = row.copy()
                current_mask = mask
            elif not overlaps(current_mask, mask):
                for i, (ok, val) in enumerate(zip(mask, row)):
                    if ok:
                        current_merged.iloc[i] = val
                        current_mask[i] = True
            else:
                merged_rows.append(current_merged)
                current_merged = row.copy()
                current_mask = mask

        if current_merged is not None:
            merged_rows.append(current_merged)

        if not merged_rows:
            return None

        df = pd.DataFrame(merged_rows).reset_index(drop=True)
        df = df.dropna(axis=1, how="all")
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        return df
