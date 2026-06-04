from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

from ..models import DatasetProfile, HeaderInfo

class BaseStrategy(ABC):
    """Abstract base class for all reconstruction strategies."""
    
    def __init__(self, raw_df: pd.DataFrame, profile: DatasetProfile, header_info: HeaderInfo):
        self.raw_df = raw_df
        self.profile = profile
        self.header_info = header_info

    @abstractmethod
    def reconstruct(self) -> Optional[pd.DataFrame]:
        """Perform the dataset reconstruction and return the structured DataFrame.
        Returns None if the strategy fundamentally fails to apply."""
        pass
        
    def get_data_start_idx(self) -> int:
        """Helper to find where the actual data starts (after headers)."""
        if self.header_info.data_start_row > 0:
            return self.header_info.data_start_row
        if not self.header_info.header_rows:
            return 0
        return max(self.header_info.header_rows) + 1
