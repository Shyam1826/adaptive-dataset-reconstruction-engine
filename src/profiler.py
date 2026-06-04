import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .models import RowProfile, DatasetProfile
from .row_signature import build_row_signature, classify_cell, detect_repeating_patterns
from .confidence import clamp, normalize_linear

class DatasetProfiler:
    """Layer 1 — Dataset Profiling
    Analyzes row density, null distribution, text/numeric ratios, and generates
    structural statistics that guide downstream layers.
    """

    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df
    
    def _is_numeric(self, val: Any) -> bool:

        if pd.isna(val):
            return False

        import datetime

        # Ignore datetime values
        if isinstance(val, (datetime.datetime, datetime.date)):
            return False

        try:
            float(str(val).replace(",", ""))
            return True

        except:
            return False
            
    def _is_text(self, val: Any) -> bool:
        if pd.isna(val):
            return False
        # Treat as text if it's a string containing some alphabetic characters 
        # (filters out stringified pure numbers)
        return isinstance(val, str) and any(c.isalpha() for c in val)

    def _profile_row(self, row_idx: int, row: pd.Series) -> RowProfile:
        total_cells = len(row)
        non_nulls = row.dropna()
        non_null_count = len(non_nulls)
        
        if total_cells == 0 or non_null_count == 0:
            return RowProfile(
                row_index=row_idx,
                non_null_count=0,
                text_count=0,
                numeric_count=0,
                text_ratio=0.0,
                numeric_ratio=0.0,
                alpha_ratio=0.0,
                first_non_null_pos=-1,
                last_non_null_pos=-1,
                unique_token_ratio=0.0,
                signature="E" * total_cells
            )
            
        text_count = sum(self._is_text(val) for val in non_nulls)
        numeric_count = sum(self._is_numeric(val) for val in non_nulls)
        
        text_ratio = text_count / non_null_count
        numeric_ratio = numeric_count / non_null_count
        
        # alpha_ratio: approx ratio of alpha chars in all cell strings combined
        str_vals = [str(x) for x in non_nulls]
        total_chars = sum(len(s) for s in str_vals)
        alpha_chars = sum(sum(c.isalpha() for c in s) for s in str_vals)
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0.0
        
        # position of first and last non-null
        is_not_null = row.notna()
        non_null_indices = np.where(is_not_null)[0]
        first_pos = int(non_null_indices[0])
        last_pos = int(non_null_indices[-1])
        
        # unique token ratio to identify potential key/header rows
        unique_tokens = len(set(str_vals))
        unique_token_ratio = unique_tokens / non_null_count if non_null_count > 0 else 0.0
        
        signature = build_row_signature(row)
        
        return RowProfile(
            row_index=row_idx,
            non_null_count=non_null_count,
            text_count=text_count,
            numeric_count=numeric_count,
            text_ratio=text_ratio,
            numeric_ratio=numeric_ratio,
            alpha_ratio=alpha_ratio,
            first_non_null_pos=first_pos,
            last_non_null_pos=last_pos,
            unique_token_ratio=unique_token_ratio,
            signature=signature
        )

    def profile(self) -> DatasetProfile:
        total_rows, total_columns = self.raw_df.shape
        
        if total_rows == 0 or total_columns == 0:
            return DatasetProfile(
                total_rows, total_columns, 0.0, 1.0, 0.0, 0.0, 0.0
            )
            
        row_profiles = []
        global_non_null = 0
        global_text = 0
        global_numeric = 0
        global_datetime = 0

        for idx in range(total_rows):
            row = self.raw_df.iloc[idx]
            rp = self._profile_row(idx, row)
            row_profiles.append(rp)
            
            global_non_null += rp.non_null_count
            global_text += rp.text_count
            global_numeric += rp.numeric_count
            row = self.raw_df.iloc[idx]
            global_datetime += sum(
                1 for v in row if pd.notna(v) and classify_cell(v) == "D"
            )
            
        total_cells = total_rows * total_columns
        row_density = global_non_null / total_cells if total_cells > 0 else 0.0
        null_ratio = 1.0 - row_density
        
        text_ratio = global_text / global_non_null if global_non_null > 0 else 0.0
        numeric_ratio = global_numeric / global_non_null if global_non_null > 0 else 0.0
        datetime_ratio = (
            global_datetime / global_non_null if global_non_null > 0 else 0.0
        )
        sparsity_score = clamp(null_ratio)
        
        # Calculate structural entropy based on row density distribution
        densities = [rp.non_null_count / total_columns for rp in row_profiles]
        sum_densities = sum(densities)
        if sum_densities > 0:
            probs = [d / sum_densities for d in densities if d > 0]
            entropy = float(-sum(p * np.log2(p) for p in probs))
        else:
            entropy = 0.0
            
        # Detect very simple contiguous blocks that are likely table regions
        likely_table_regions = []
        current_region = None
        for rp in row_profiles:
            # We consider a row part of a potential table if its density > 10%
            # (Very messy Excel files often have low density)
            is_populated = (rp.non_null_count / total_columns) > 0.05 if total_columns > 0 else False
            
            if is_populated:
                if current_region is None:
                    current_region = {"start": rp.row_index, "end": rp.row_index}
                else:
                    current_region["end"] = rp.row_index
            else:
                if current_region is not None:
                    if current_region["end"] - current_region["start"] >= 1:
                        likely_table_regions.append(current_region)
                    current_region = None
        
        if current_region is not None and (current_region["end"] - current_region["start"] >= 1):
            likely_table_regions.append(current_region)
            
        signatures = [rp.signature for rp in row_profiles]
        span, pattern = detect_repeating_patterns(signatures)

        row_similarity_stability = self._row_similarity_stability(signatures)
        segmentation_likelihood = self._segmentation_likelihood(
            signatures, span, row_similarity_stability
        )
        stability_interval = (
            max(0.0, row_similarity_stability - 0.15),
            min(1.0, row_similarity_stability + 0.15),
        )
        profile_confidence = clamp(
            0.35 * row_density
            + 0.25 * row_similarity_stability
            + 0.25 * segmentation_likelihood
            + 0.15 * (1.0 - min(entropy / 4.0, 1.0))
        )

        return DatasetProfile(
            total_rows=total_rows,
            total_columns=total_columns,
            row_density=row_density,
            null_ratio=null_ratio,
            text_ratio=text_ratio,
            numeric_ratio=numeric_ratio,
            datetime_ratio=datetime_ratio,
            row_profiles=row_profiles,
            likely_table_regions=likely_table_regions,
            structural_entropy=entropy,
            repeating_signature_span=span,
            repeating_signature_pattern=pattern,
            row_similarity_stability=row_similarity_stability,
            segmentation_likelihood=segmentation_likelihood,
            sparsity_score=sparsity_score,
            profile_confidence=profile_confidence,
            stability_interval=stability_interval,
        )

    def _row_similarity_stability(self, signatures: List[str]) -> float:
        active = [s for s in signatures if set(s) != {"E"}]
        if len(active) < 2:
            return 0.0
        sims = []
        for i in range(1, len(active)):
            a, b = active[i - 1], active[i]
            n = max(len(a), len(b), 1)
            matches = sum(1 for x, y in zip(a, b) if x == y)
            sims.append(matches / n)
        return sum(sims) / len(sims) if sims else 0.0

    def _segmentation_likelihood(
        self, signatures: List[str], span: int, stability: float
    ) -> float:
        if span <= 0:
            return stability * 0.5
        active = [s for s in signatures if set(s) != {"E"}]
        if len(active) < span * 2:
            return stability * 0.4
        blocks = [
            tuple(active[i : i + span]) for i in range(0, len(active) - span + 1, span)
        ]
        if len(blocks) < 2:
            return 0.3
        from collections import Counter

        counts = Counter(blocks)
        _, top = counts.most_common(1)[0]
        ratio = top / len(blocks)
        return clamp(0.5 * ratio + 0.5 * stability)
