from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

@dataclass
class RowProfile:
    row_index: int
    non_null_count: int
    text_count: int
    numeric_count: int
    text_ratio: float
    numeric_ratio: float
    alpha_ratio: float
    first_non_null_pos: int
    last_non_null_pos: int
    unique_token_ratio: float
    header_confidence: float = 0.0
    signature: str = ""

@dataclass
class DatasetProfile:
    total_rows: int
    total_columns: int
    row_density: float
    null_ratio: float
    text_ratio: float
    numeric_ratio: float
    datetime_ratio: float = 0.0
    row_profiles: List[RowProfile] = field(default_factory=list)
    likely_table_regions: List[Dict[str, int]] = field(default_factory=list)
    structural_entropy: float = 0.0
    repeating_signature_span: int = 0
    repeating_signature_pattern: List[str] = None
    row_similarity_stability: float = 0.0
    segmentation_likelihood: float = 0.0
    sparsity_score: float = 0.0
    profile_confidence: float = 0.0
    stability_interval: tuple = (0.0, 1.0)

@dataclass
class BlueprintSegment:
    segment_id: int
    row_offset: int
    column_indices: List[int]
    width: int
    global_start: int
    global_end: int


@dataclass
class StructuralBlueprint:
    # Maps row_offset -> list of valid column indices
    # e.g., {0: [0, 1, 2], 1: [0, 2], 2: [1, 3]}
    row_to_cols_map: Dict[int, List[int]]
    span: int
    segments: List[BlueprintSegment] = field(default_factory=list)
    total_columns: int = 0
    cycle_length: int = 0
    signature_sequence: List[str] = field(default_factory=list)


@dataclass
class PeriodicStructure:
    cycle_length: int
    segment_signatures: List[str]
    segment_widths: List[int]
    periodicity_score: float
    segmentation_mode: str = "periodic"
    signals: Dict[str, Any] = field(default_factory=dict)
    anchor_strength: float = 0.0
    
@dataclass
class HeaderInfo:
    header_rows: List[int]
    header_span: int
    schema_width: int
    confidence: float
    blueprint: Optional[StructuralBlueprint] = None
    data_start_row: int = 0
    hypothesis_name: str = ""
    hypothesis_confidence: float = 0.0
    periodic_structure: Optional["PeriodicStructure"] = None


@dataclass
class HeaderBoundaryResult:
    data_start_row: int
    header_rows: List[int]
    boundary_confidence: float
    row_scores: List[float] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnchorColumn:
    column_index: int
    global_index: int
    header_label: str
    pattern_type: str
    confidence: float
    segment_offset: int = 0


@dataclass
class RowBoundaryScore:
    row_index: int
    score: float
    is_boundary: bool
    segment_offset: int


@dataclass
class ReconstructionResult:
    dataframe: pd.DataFrame
    strategy_name: str
    confidence_score: float
    validation_score: float
    validation_report: Dict[str, Any]
    structural_stability_score: float = 0.0
