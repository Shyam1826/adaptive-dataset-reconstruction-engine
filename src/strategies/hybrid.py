import pandas as pd
from typing import Optional
from .base import BaseStrategy
from .positional import PositionalStrategy
from .streaming import StreamingStrategy

class HybridStrategy(BaseStrategy):
    """Strategy D — Hybrid Reconstruction
    Combines Positional and Streaming. Relies heavily on checking sparsity bounds.
    """
    
    def reconstruct(self) -> Optional[pd.DataFrame]:
        # Favor positional first as it preserves internal schema structures much better
        pos_df = PositionalStrategy(self.raw_df, self.profile, self.header_info).reconstruct()
        
        if pos_df is not None:
            # Check null density of the reconstructed dataframe
            null_ratio = pos_df.isna().sum().sum() / (pos_df.shape[0] * pos_df.shape[1])
            
            # If the matrix is fairly continuous and tight, accept it
            if null_ratio < 0.35:
                return pos_df
                
        # If positional resulted in a highly sparse dataframe, the structure is likely broken
        # Fallback to flattening the stream
        stream_df = StreamingStrategy(self.raw_df, self.profile, self.header_info).reconstruct()
        return stream_df
