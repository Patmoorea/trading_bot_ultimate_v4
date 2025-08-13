import numpy as np
import pandas as pd
from typing import Dict, Union, List
class VolatilityIndicators:
    def __init__(self):
        self.lookback_periods = {
            'atr': 14,
            'bbands': 20,
            'keltner': 20,
            'vix': 30
        }
    def atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range avec intégration M4"""
        high = data['high']
        low = data['low']
        close = data['close']
        # Calcul optimisé du TR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # ATR avec Metal
        return tr.ewm(span=period, adjust=False).mean()
    def bollinger_bands(
        self, 
        data: pd.Series, 
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Bandes de Bollinger adaptatives"""
        typical_price = data
        # SMA et déviations
        middle_band = typical_price.rolling(window=period).mean()
        std = typical_price.rolling(window=period).std()
        # Bandes dynamiques
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        # Largeur (volatilité)
        bandwidth = (upper_band - lower_band) / middle_band
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'bandwidth': bandwidth
        }
    def keltner_channels(
        self, 
        data: pd.DataFrame,
        period: int = 20,
        atr_multiplier: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Keltner Channels avec ATR"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        # EMA centrale
        middle_line = typical_price.ewm(span=period, adjust=False).mean()
        # ATR pour la largeur des bandes
        atr_values = self.atr(data, period)
        upper_line = middle_line + (atr_values * atr_multiplier)
        lower_line = middle_line - (atr_values * atr_multiplier)
        return {
            'upper': upper_line,
            'middle': middle_line,
            'lower': lower_line
        }
    def volatility_index(
        self,
        data: pd.DataFrame,
        period: int = 30
    ) -> pd.Series:
        """VIX-like indicator pour crypto"""
        log_returns = np.log(data['close'] / data['close'].shift(1))
        volatility = log_returns.rolling(window=period).std() * np.sqrt(365)
        return volatility * 100  # En pourcentage
