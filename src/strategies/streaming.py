import pandas as pd
from typing import Optional

from .base import BaseStrategy
from ..structure_blueprint import BlueprintReconstructor


class StreamingStrategy(BaseStrategy):
    """Strategy C — Streaming Reconstruction
    Chunks value streams using blueprint-defined schema width.
    """

    def reconstruct(self) -> Optional[pd.DataFrame]:
        start_idx = self.get_data_start_idx()
        if start_idx >= self.profile.total_rows:
            return None

        blueprint = self.header_info.blueprint
        if blueprint and blueprint.span > 1:
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
            if result is not None:
                return result

        target_schema_width = self._schema_width()

        if target_schema_width <= 0:
            return None

        stream = []
        for idx in range(start_idx, self.profile.total_rows):
            row = self.raw_df.iloc[idx]
            if blueprint and blueprint.span == 1 and 0 in blueprint.row_to_cols_map:
                for col_idx in blueprint.row_to_cols_map[0]:
                    if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                        stream.append(row.iloc[col_idx])
            else:
                stream.extend(row.dropna().tolist())

        if not stream:
            return None

        records = []
        for i in range(0, len(stream), target_schema_width):
            chunk = stream[i : i + target_schema_width]
            if len(chunk) < target_schema_width:
                chunk.extend([None] * (target_schema_width - len(chunk)))
            records.append(chunk)

        return BlueprintReconstructor.records_to_dataframe(records)

    def _schema_width(self) -> int:
        blueprint = self.header_info.blueprint
        if blueprint and blueprint.total_columns > 0:
            return blueprint.total_columns
        if self.header_info.schema_width > 0:
            return self.header_info.schema_width
        return self.profile.total_columns
