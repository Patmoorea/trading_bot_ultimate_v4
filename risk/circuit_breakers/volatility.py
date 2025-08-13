from datetime import datetime
from decimal import Decimal
from typing import List, Dict
class VolatilitySurgeDetector:
    def __init__(self, 
                 base_volatility: float = 0.02,
                 surge_multiplier: float = 3.0,
                 window_size: int = 20):
        self.base_volatility = Decimal(str(base_volatility))
        self.surge_multiplier = Decimal(str(surge_multiplier))
        self.window_size = window_size
        self.price_history: List[Dict] = []
        self.volatility_history: List[Dict] = []
        self.last_update = datetime(2025, 5, 25, 0, 53, 43)
        self.cluster_threshold = 3
    def add_price(self, price: float) -> None:
        current_time = datetime.utcnow()
        self.price_history.append({
            'timestamp': current_time,
            'price': Decimal(str(price))
        })
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)
            self._calculate_volatility()
    def _calculate_volatility(self) -> None:
        if len(self.price_history) < 2:
            return
        prices = [entry['price'] for entry in self.price_history]
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]
        # Calcul de la volatilité réalisée
        squared_returns = [r * r for r in returns]
        realized_vol = (sum(squared_returns) / len(squared_returns)).sqrt()
        self.volatility_history.append({
            'timestamp': datetime.utcnow(),
            'value': realized_vol
        })
        if len(self.volatility_history) > self.window_size:
            self.volatility_history.pop(0)
    def is_surge_detected(self) -> bool:
        if not self.volatility_history:
            return False
        current_vol = self.volatility_history[-1]['value']
        return current_vol > (self.base_volatility * self.surge_multiplier)
    def get_cluster_count(self) -> int:
        if len(self.volatility_history) < 2:
            return 0
        clusters = 0
        in_cluster = False
        for vol in self.volatility_history:
            is_high = vol['value'] > (self.base_volatility * self.surge_multiplier)
            if is_high and not in_cluster:
                clusters += 1
                in_cluster = True
            elif not is_high:
                in_cluster = False
        return clusters
