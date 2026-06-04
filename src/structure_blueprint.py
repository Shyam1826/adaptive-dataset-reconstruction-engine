import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from .models import StructuralBlueprint, BlueprintSegment
from .row_signature import classify_cell


class BlueprintGenerator:
    """Builds a shared structural blueprint from detected header rows."""

    @staticmethod
    def generate_from_headers(raw_df: pd.DataFrame, header_rows: List[int]) -> StructuralBlueprint:
        if not header_rows:
            return StructuralBlueprint({}, 0, [], 0)

        header_rows = sorted(header_rows)
        span = len(header_rows)
        row_to_cols_map: Dict[int, List[int]] = {}
        segments: List[BlueprintSegment] = []
        global_offset = 0

        for offset, row_idx in enumerate(header_rows):
            row = raw_df.iloc[row_idx]
            valid_cols = []
            for col_idx, val in enumerate(row):
                if pd.notna(val) and str(val).strip() != "":
                    valid_cols.append(col_idx)

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
            span=span,
            segments=segments,
            total_columns=global_offset,
        )


class BlueprintReconstructor:
    """Reconstructs headers and data using the same structural blueprint."""

    @staticmethod
    def flatten_row_group(
        rows: List[pd.Series],
        blueprint: StructuralBlueprint,
        *,
        for_headers: bool = False,
    ) -> List[Any]:
        if not blueprint or blueprint.span <= 0:
            return []

        flattened: List[Any] = []
        for row_offset in range(blueprint.span):
            valid_cols = blueprint.row_to_cols_map.get(row_offset, [])
            if row_offset >= len(rows):
                for _ in valid_cols:
                    flattened.append(
                        f"missing_{'header' if for_headers else 'value'}_{row_offset}"
                        if for_headers
                        else None
                    )
                continue

            row = rows[row_offset]
            for col_idx in valid_cols:
                if col_idx < len(row):
                    val = row.iloc[col_idx]
                    if pd.notna(val) and str(val).strip() != "":
                        if for_headers:
                            flattened.append(str(val).replace("\n", " ").strip())
                        else:
                            flattened.append(val)
                    else:
                        flattened.append(
                            f"missing_header_{row_offset}_{col_idx}"
                            if for_headers
                            else None
                        )
                else:
                    flattened.append(
                        f"missing_header_{row_offset}_{col_idx}"
                        if for_headers
                        else None
                    )
        return flattened

    @staticmethod
    def extract_headers(
        raw_df: pd.DataFrame,
        header_rows: List[int],
        blueprint: StructuralBlueprint,
    ) -> List[str]:
        if not header_rows or not blueprint:
            return []

        header_rows = sorted(header_rows)
        rows = [raw_df.iloc[idx] for idx in header_rows]
        return BlueprintReconstructor.flatten_row_group(rows, blueprint, for_headers=True)

    @staticmethod
    def reconstruct_records(
        raw_df: pd.DataFrame,
        start_idx: int,
        blueprint: StructuralBlueprint,
        profile=None,
        header_names: Optional[List[str]] = None,
        *,
        dynamic: bool = True,
    ) -> List[List[Any]]:
        if not blueprint or blueprint.span <= 0 or start_idx >= len(raw_df):
            return []

        if dynamic:
            from .boundary_detector import reconstruct_with_dynamic_boundaries

            return reconstruct_with_dynamic_boundaries(
                raw_df, start_idx, blueprint, profile, header_names
            )

        records: List[List[Any]] = []
        current_group: List[pd.Series] = []

        for idx in range(start_idx, len(raw_df)):
            row = raw_df.iloc[idx]
            if row.isna().all():
                continue

            current_group.append(row)
            if len(current_group) >= blueprint.span:
                records.append(
                    BlueprintReconstructor.flatten_row_group(current_group, blueprint)
                )
                current_group = []

        if current_group:
            records.append(
                BlueprintReconstructor.flatten_row_group(current_group, blueprint)
            )

        return records

    @staticmethod
    def records_to_dataframe(records: List[List[Any]]) -> Optional[pd.DataFrame]:
        if not records:
            return None

        max_len = max(len(r) for r in records)
        if max_len == 0:
            return None

        normalized = []
        for record in records:
            row = list(record)
            if len(row) < max_len:
                row.extend([None] * (max_len - len(row)))
            normalized.append(row[:max_len])

        return pd.DataFrame(
            normalized, columns=[f"col_{i}" for i in range(max_len)]
        )


def infer_header_semantic_type(header: str) -> str:
    name = str(header).lower()
    if any(k in name for k in ["date", "time", "stamp", "year", "month", "day"]):
        return "datetime"
    if any(
        k in name
        for k in [
            "amount",
            "total",
            "quantity",
            "score",
            "price",
            "value",
            "sum",
            "count",
            "qty",
            "rate",
            "cost",
        ]
    ):
        return "numeric"
    return "categorical"


def infer_value_semantic_type(value: Any) -> str:
    if pd.isna(value) or value == "":
        return "empty"

    cell_class = classify_cell(value)
    if cell_class == "D":
        return "datetime"
    if cell_class == "N":
        return "numeric"
    if cell_class in ("T", "M"):
        return "categorical"
    return "empty"


def semantic_type_match(header_type: str, value_type: str) -> float:
    if value_type == "empty":
        return 0.5
    if header_type == value_type:
        return 1.0
    if header_type == "categorical" and value_type in ("categorical", "numeric"):
        return 0.8
    return 0.0


def score_row_alignment(headers: List[str], values: List[Any]) -> float:
    if not headers:
        return 0.0

    scores = []
    for i, header in enumerate(headers):
        expected = infer_header_semantic_type(header)
        actual = infer_value_semantic_type(values[i] if i < len(values) else None)
        scores.append(semantic_type_match(expected, actual))

    return sum(scores) / len(scores) if scores else 0.0


def validate_semantic_alignment(
    headers: List[str], dataframe: pd.DataFrame
) -> Dict[str, Any]:
    if dataframe is None or dataframe.empty:
        return {
            "semantic_alignment_score": 0.0,
            "column_scores": {},
            "misaligned_columns": [],
        }

    headers = list(headers)[: len(dataframe.columns)]
    column_scores: Dict[str, float] = {}
    misaligned: List[str] = []

    for i, col in enumerate(dataframe.columns):
        header_name = headers[i] if i < len(headers) else str(col)
        col_values = dataframe.iloc[:, i].dropna().tolist()
        if not col_values:
            column_scores[str(col)] = 0.5
            continue

        col_score = score_row_alignment(
            [header_name] * len(col_values),
            col_values,
        )
        column_scores[str(col)] = col_score
        if col_score < 0.5:
            misaligned.append(str(col))

    overall = (
        sum(column_scores.values()) / len(column_scores) if column_scores else 0.0
    )
    return {
        "semantic_alignment_score": overall,
        "column_scores": column_scores,
        "misaligned_columns": misaligned,
    }


def attempt_column_realignment(
    headers: List[str], dataframe: pd.DataFrame, max_shift: Optional[int] = None
) -> Tuple[pd.DataFrame, float, int]:
    """Shift data columns locally to maximize semantic consistency. Headers stay fixed."""
    if dataframe is None or dataframe.empty:
        return dataframe, 0.0, 0

    n_cols = len(dataframe.columns)
    if max_shift is None:
        max_shift = min(6, max(1, n_cols - 1))

    best_df = dataframe.copy()
    best_score = score_row_alignment(headers, best_df.iloc[0].tolist())
    best_shift = 0

    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            continue

        shifted = dataframe.copy()
        for row_idx in range(len(shifted)):
            values = shifted.iloc[row_idx].tolist()
            rolled = _roll_values(values, shift)
            shifted.iloc[row_idx] = rolled

        score = _score_dataframe_alignment(headers, shifted)
        if score > best_score:
            best_score = score
            best_df = shifted
            best_shift = shift

    return best_df, best_score, best_shift


def _roll_values(values: List[Any], shift: int) -> List[Any]:
    if not values or shift == 0:
        return values
    n = len(values)
    shift = shift % n
    if shift == 0:
        return values
    return values[-shift:] + values[:-shift]


def _score_dataframe_alignment(headers: List[str], dataframe: pd.DataFrame) -> float:
    if dataframe.empty:
        return 0.0
    row_scores = [
        score_row_alignment(headers, dataframe.iloc[i].tolist())
        for i in range(len(dataframe))
    ]
    return sum(row_scores) / len(row_scores) if row_scores else 0.0
