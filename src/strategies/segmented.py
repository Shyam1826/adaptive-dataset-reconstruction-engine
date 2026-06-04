import pandas as pd
from typing import Optional

from .base import BaseStrategy
from ..structure_blueprint import BlueprintReconstructor


class SegmentedBlockStrategy(BaseStrategy):
    """Strategy A — Segmented Block Reconstruction
    Uses dynamic record boundary detection with the shared structural blueprint.
    """

    def reconstruct(self) -> Optional[pd.DataFrame]:
        start_idx = self.get_data_start_idx()
        if start_idx >= self.profile.total_rows:
            return None

        blueprint = self.header_info.blueprint
        if not blueprint or blueprint.span <= 0:
            return None

        header_names = self._provisional_header_names()
        records = BlueprintReconstructor.reconstruct_records(
            self.raw_df,
            start_idx,
            blueprint,
            self.profile,
            header_names,
            dynamic=True,
        )
        return BlueprintReconstructor.records_to_dataframe(records)

    def _provisional_header_names(self) -> list:
        if not self.header_info.blueprint or not self.header_info.header_rows:
            return []
        return BlueprintReconstructor.extract_headers(
            self.raw_df,
            self.header_info.header_rows,
            self.header_info.blueprint,
        )
