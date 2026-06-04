"""Optimization-based semantic realignment (column shifts and local corrections)."""

from typing import List, Optional, Tuple

import pandas as pd

from .confidence import clamp
from .structure_blueprint import (
    score_row_alignment,
    validate_semantic_alignment,
)


def optimize_column_alignment(
    headers: List[str],
    dataframe: pd.DataFrame,
    max_shift: Optional[int] = None,
) -> Tuple[pd.DataFrame, float, int, dict]:
    """
    Search shift hypotheses and return the alignment with highest semantic score.
    """
    if dataframe is None or dataframe.empty:
        return dataframe, 0.0, 0, {}

    n_cols = len(dataframe.columns)
    if max_shift is None:
        max_shift = min(8, max(1, n_cols - 1))

    headers = list(headers)[:n_cols]
    best_df = dataframe.copy()
    best_score = _global_alignment_score(headers, best_df)
    best_shift = 0
    scores_log = {0: best_score}

    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            continue
        candidate = _apply_shift(dataframe, shift)
        score = _global_alignment_score(headers, candidate)
        scores_log[shift] = score
        if score > best_score:
            best_score = score
            best_df = candidate
            best_shift = shift

    best_df = best_df.copy()
    best_df.columns = headers
    return best_df, best_score, best_shift, scores_log


def _apply_shift(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    shifted = df.copy()
    for i in range(len(shifted)):
        values = shifted.iloc[i].tolist()
        n = len(values)
        if n == 0:
            continue
        s = shift % n
        if s == 0:
            continue
        rolled = values[-s:] + values[:-s]
        shifted.iloc[i] = rolled
    return shifted


def _global_alignment_score(headers: List[str], df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    row_scores = [
        score_row_alignment(headers, df.iloc[i].tolist()) for i in range(len(df))
    ]
    base = sum(row_scores) / len(row_scores) if row_scores else 0.0
    report = validate_semantic_alignment(headers, df)
    semantic = report.get("semantic_alignment_score", base)
    return clamp(0.6 * semantic + 0.4 * base)
