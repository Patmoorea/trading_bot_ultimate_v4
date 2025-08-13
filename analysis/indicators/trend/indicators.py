import numpy as np
import pandas as pd
from typing import Dict, Any
class TrendIndicators:
    def __init__(self):
        self.required_history = 100
    def ichimoku(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calcul de l'Ichimoku Cloud"""
        high = data['high']
        low = data['low']
        # Tenkan-sen (Conversion Line)
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        # Kijun-sen (Base Line)
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        # Senkou Spans
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        chikou_span = data['close'].shift(-26)
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
