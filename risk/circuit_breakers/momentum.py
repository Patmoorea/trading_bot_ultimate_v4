from decimal import Decimal
from typing import List, Dict
from datetime import datetime
class MomentumSurgeDetector:
    def __init__(self, lookback_periods: int = 14, surge_threshold: float = 3.0):
        self.lookback = lookback_periods
        self.surge_threshold = Decimal(str(surge_threshold))
        self.price_history: List[Dict] = []
        self.momentum_values: List[Decimal] = []
        self.max_acceleration = Decimal('0.25')
    def add_price(self, price: float, volume: float) -> None:
        current_time = datetime.utcnow()
        self.price_history.append({
            'timestamp': current_time,
            'price': Decimal(str(price)),
            'volume': Decimal(str(volume))
        })
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)
            self._calculate_momentum()
    def _calculate_momentum(self) -> None:
        if len(self.price_history) < 2:
            return
        latest_price = self.price_history[-1]['price']
        previous_price = self.price_history[0]['price']
        roc = (latest_price - previous_price) / previous_price
        self.momentum_values.append(roc)
        if len(self.momentum_values) > self.lookback:
            self.momentum_values.pop(0)
    def get_momentum_acceleration(self) -> Decimal:
        if len(self.momentum_values) < 2:
            return Decimal('0')
        return self.momentum_values[-1] - self.momentum_values[-2]
    def is_surge_detected(self) -> bool:
        if not self.momentum_values:
            return False
        return abs(self.momentum_values[-1]) > self.surge_threshold
