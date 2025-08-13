import pandas as pd
import numpy as np
from typing import Dict, List, Union
class VolumeAnalysis:
    def __init__(self):
        self.volume_thresholds = {
            'whale': 100000,  # USDC
            'institutional': 50000,
            'retail': 1000
        }
    def vwap(self, data: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price sur M4"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    def on_balance_volume(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """OBV avec détection de divergence"""
        obv = pd.Series(index=close.index, dtype='float64')
        obv.iloc[0] = volume.iloc[0]
        # Calcul optimisé de l'OBV
        close_diff = close.diff()
        volume_pos = volume.where(close_diff > 0, 0)
        volume_neg = volume.where(close_diff < 0, 0)
        obv = obv.shift(1) + volume_pos - volume_neg
        return obv.fillna(0)
    def volume_profile(
        self,
        data: pd.DataFrame,
        price_levels: int = 100
    ) -> Dict[str, Union[pd.Series, float]]:
        """Profile de volume temps réel"""
        # Calcul des niveaux de prix
        price_min = data['low'].min()
        price_max = data['high'].max()
        step = (price_max - price_min) / price_levels
        # Construction du profile
        levels = np.arange(price_min, price_max, step)
        profile = pd.Series(0, index=levels)
        # Agrégation du volume par niveau
        for idx, row in data.iterrows():
            level_idx = int((row['close'] - price_min) / step)
            if 0 <= level_idx < price_levels:
                profile.iloc[level_idx] += row['volume']
        # Point de contrôle
        poc_level = profile.idxmax()
        return {
            'profile': profile,
            'poc': poc_level,
            'value_area_high': self._find_value_area_level(profile, poc_level, 0.7),
            'value_area_low': self._find_value_area_level(profile, poc_level, 0.7, high=False)
        }
    def _find_value_area_level(
        self,
        profile: pd.Series,
        poc: float,
        threshold: float,
        high: bool = True
    ) -> float:
        """Trouve les niveaux de Value Area"""
        total_volume = profile.sum()
        target_volume = total_volume * threshold
        cumsum = profile[profile.index >= poc].cumsum() if high \
                else profile[profile.index <= poc].cumsum()
        return cumsum[cumsum >= target_volume].index[0]
