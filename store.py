"""
Global in-memory data store for Agent360.
Maintains state across API requests within a single server session.
"""
from typing import Optional, Dict, List
import pandas as pd


class AppStore:
    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.clustered_df: Optional[pd.DataFrame] = None
        self.selected_features: List[str] = []
        self.numeric_columns: List[str] = []
        self.all_columns: List[str] = []
        self.optimal_k: Optional[int] = None
        self.cluster_count: Optional[int] = None
        self.tier_mapping: Dict[int, str] = {}
        self.agent_id_col: Optional[str] = None

    def reset(self):
        self.__init__()


store = AppStore()
