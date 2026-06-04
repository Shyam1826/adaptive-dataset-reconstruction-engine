import pandas as pd
import datetime
from typing import Any, List, Tuple

def classify_cell(val: Any) -> str:
    """
    T = text-like
    N = numeric-like
    D = datetime-like
    E = empty/null
    M = mixed/unknown
    """
    if pd.isna(val) or val == "":
        return "E"
        
    if isinstance(val, (datetime.datetime, datetime.date)):
        return "D"
        
    # Try datetime parse first if string
    if isinstance(val, str):
        # basic check to avoid heavy try/except for clear texts
        # But pandas to_datetime works to safely check
        try:
            pd.to_datetime(val, format='mixed')
            return "D"
        except:
            pass

    # Try numeric
    try:
        float(str(val).replace(",", "").replace("$", "").strip())
        return "N"
    except (ValueError, TypeError):
        pass

    # Basic text check
    if isinstance(val, str) and any(c.isalpha() for c in val):
        return "T"
        
    return "M"

def build_row_signature(row: pd.Series) -> str:
    parts = []
    for val in row:
        parts.append(classify_cell(val))
    # We can omit trailing 'E's or just keep the full raw signature. 
    # Usually returning the raw representation is best for pattern matching across full widths.
    return "".join(parts)

def detect_repeating_patterns(signatures: List[str], max_pattern_len: int = 15) -> Tuple[int, List[str]]:
    """
    Finds the dominant repeating pattern in a sequence of signatures.
    Returns (pattern_length, pattern).
    If no clear pattern is found, returns (0, []).
    """
    if not signatures:
        return 0, []
        
    # Remove entirely empty 'E...' rows from pattern analysis as they might be whitespace breaks
    active_sigs = [s for s in signatures if set(s) != {'E'}]
    if not active_sigs:
        return 0, []
        
    n = len(active_sigs)
    
    # Try different sequence lengths
    for span in range(2, max_pattern_len + 1):
        if n < span * 2:
            break
            
        # Extract potential pattern blocks
        # We also need to check different offset starts, in case headers take up a few rows.
        for offset in range(span):
            blocks = []
            for i in range(offset, n - span + 1, span):
                blocks.append(tuple(active_sigs[i:i+span]))
                
            if len(blocks) < 2:
                continue
                
            # Find the most common block
            from collections import Counter
            block_counts = Counter(blocks)
            most_common_block, count = block_counts.most_common(1)[0]
            
            match_ratio = count / len(blocks)
            
            # If > 60% of the blocks match, we found a repeating signature pattern!
            if match_ratio > 0.6:
                return span, list(most_common_block)
            
    return 0, []
