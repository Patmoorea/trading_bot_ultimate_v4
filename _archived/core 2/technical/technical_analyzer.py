import pandas as pd
from functools import lru_cache
class TechnicalAnalyzer:
    """Analyse technique avec cache et indicateurs"""
    def __init__(self):
        self.indicators = {}
    def calculate_rsi_enhanced(self, data, window=14):
        """Calculate RSI with enhanced logic and NaN handling"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # 50 = valeur neutre quand pas assez de données
    def analyze_with_cache(self, data):
        """Analyze with caching using hashable input"""
        data_tuple = tuple(data['close'].values)
        return self._cached_analysis(data_tuple)
    @lru_cache(maxsize=128)
    def _cached_analysis(self, data_tuple):
        data = pd.DataFrame({'close': data_tuple})
        return self.calculate_rsi_enhanced(data)
