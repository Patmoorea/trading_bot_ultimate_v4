from typing import Dict, Any
import numpy as np
import logging
from pandas import DataFrame, read_csv, to_numeric
class TradingEngine:
    def __init__(self):
        self.ai_enabled = False
        self.logger = logging.getLogger(__name__)
    def load_data(self, filepath: str) -> DataFrame:
        """Charge les données avec vérification des types"""
        df = read_csv(filepath)
        num_cols = ['open', 'high', 'low', 'close', 'volume']
        df[num_cols] = df[num_cols].apply(to_numeric, errors='coerce')
        return df.dropna()
