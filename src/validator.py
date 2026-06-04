import pandas as pd
from typing import Dict, Any

class ValidationEngine:
    """Layer 4 — Validation Engine
    Validates a reconstructed dataframe, checking datatypes, anomalies, 
    and generating structural metrics useful for scoring.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate(self) -> Dict[str, Any]:
        if self.df.empty:
            return {
                "is_valid": False,
                "null_ratio": 1.0,
                "row_count": 0,
                "col_count": 0,
                "inferred_numeric_cols": 0
            }
            
        total_cells = self.df.size
        null_count = self.df.isna().sum().sum()
        null_ratio = float(null_count / total_cells) if total_cells > 0 else 1.0
        
        inferred_types = {}
        numeric_count = 0
        
        for col in self.df.columns:
            # Try to convert to numeric to see if column is predominantly numeric
            col_data = self.df[col].dropna()
            if col_data.empty:
                inferred_types[col] = "empty"
                continue
                
            try:
                # Force mixed types to drop through
                pd.to_numeric(col_data)
                inferred_types[col] = "numeric"
                numeric_count += 1
            except (ValueError, TypeError):
                # Try datetime
                try:
                    pd.to_datetime(col_data, format='mixed', errors='raise')
                    inferred_types[col] = "datetime"
                except (ValueError, TypeError, Exception):
                    inferred_types[col] = "categorical_or_text"
                    
        # Compute signature consistency
        from .row_signature import classify_cell
        signature_consistency = 0.0
        if len(self.df.columns) > 0 and len(self.df) > 0:
            column_consistencies = []
            for col in self.df.columns:
                col_data = self.df[col].dropna()
                if len(col_data) == 0:
                    continue
                # Classify all cells in this column
                classes = [classify_cell(val) for val in col_data]
                if not classes:
                    continue
                # Ratio of the most common class
                most_common = max(set(classes), key=classes.count)
                consistency = classes.count(most_common) / len(classes)
                column_consistencies.append(consistency)
            if column_consistencies:
                signature_consistency = sum(column_consistencies) / len(column_consistencies)
                    
        from .structure_blueprint import validate_semantic_alignment
        from .boundary_detector import validate_record_coherence

        alignment = validate_semantic_alignment(list(self.df.columns), self.df)
        semantic_alignment_score = alignment.get("semantic_alignment_score", 1.0)
        coherence = validate_record_coherence(list(self.df.columns), self.df)
        record_coherence_score = coherence.get("record_coherence_score", 1.0)

        return {
            "is_valid": True,
            "null_ratio": null_ratio,
            "row_count": len(self.df),
            "col_count": len(self.df.columns),
            "inferred_types": inferred_types,
            "numeric_col_count": numeric_count,
            "signature_consistency": signature_consistency,
            "semantic_alignment_score": semantic_alignment_score,
            "record_coherence_score": record_coherence_score,
            "anchor_uniqueness": coherence.get("anchor_uniqueness", 1.0),
            "duplicate_anchor_rows": coherence.get("duplicate_anchor_rows", []),
        }
