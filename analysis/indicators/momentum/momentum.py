import pandas as pd
import numpy as np
from typing import Dict
class MomentumIndicators:
    def __init__(self):
        self.lookback_periods = {
            'rsi': 14,
            'stoch': 14,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        }
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    def stochastic_rsi(self, data: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Stochastic RSI"""
        rsi_values = self.rsi(data, period)
        stoch_rsi = (rsi_values - rsi_values.rolling(period).min()) / \
                    (rsi_values.rolling(period).max() - rsi_values.rolling(period).min())
        k = stoch_rsi.rolling(3).mean() * 100  # %K line
        d = k.rolling(3).mean()  # %D line (signal)
        return {'k': k, 'd': d}
    def macd(self, data: pd.Series) -> Dict[str, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        exp1 = data.ewm(span=self.lookback_periods['macd']['fast']).mean()
        exp2 = data.ewm(span=self.lookback_periods['macd']['slow']).mean()
        macd_line = exp1 - exp2
        signal = macd_line.ewm(span=self.lookback_periods['macd']['signal']).mean()
        histogram = macd_line - signal
        return {
            'macd': macd_line,
            'signal': signal,
            'histogram': histogram
        }
