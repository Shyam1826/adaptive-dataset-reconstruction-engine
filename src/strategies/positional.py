import pandas as pd
from typing import Optional

from .base import BaseStrategy
from ..structure_blueprint import BlueprintReconstructor


class PositionalStrategy(BaseStrategy):
    """Strategy B — Positional Reconstruction
    Reconstructs via the shared blueprint when available; falls back to sparse merge.
    """

    def reconstruct(self) -> Optional[pd.DataFrame]:
        start_idx = self.get_data_start_idx()
        if start_idx >= self.profile.total_rows:
            return None

        blueprint = self.header_info.blueprint
        if blueprint and blueprint.span >= 1 and blueprint.total_columns > 0:
            header_names = None
            if self.header_info.header_rows and self.header_info.blueprint:
                header_names = BlueprintReconstructor.extract_headers(
                    self.raw_df,
                    self.header_info.header_rows,
                    self.header_info.blueprint,
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
            if result is not None:
                return result

        return self._reconstruct_sparse_merge(start_idx)

    def _reconstruct_sparse_merge(self, start_idx: int) -> Optional[pd.DataFrame]:
        data = self.raw_df.iloc[start_idx:].copy()

        merged_rows = []
        current_merged = None
        current_mask = None

        def overlaps(mask1, mask2):
            return any(m1 and m2 for m1, m2 in zip(mask1, mask2))

        for _, row in data.iterrows():
            mask = row.notna().tolist()
            if not any(mask):
                continue

            if current_merged is None:
                current_merged = row.copy()
                current_mask = mask
            else:
                if not overlaps(current_mask, mask):
                    for i, (is_valid, val) in enumerate(zip(mask, row)):
                        if is_valid:
                            current_merged[current_merged.index[i]] = val
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
