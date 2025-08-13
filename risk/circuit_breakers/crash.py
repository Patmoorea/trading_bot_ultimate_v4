from decimal import Decimal
from datetime import datetime
class MarketCrashDetector:
    def __init__(self, price_drop_threshold: float = 0.10,
                 volume_surge_threshold: float = 3.0,
                 timeframe_minutes: int = 5):
        self.price_threshold = Decimal(str(price_drop_threshold))
        self.volume_threshold = Decimal(str(volume_surge_threshold))
        self.timeframe = timeframe_minutes
        self.price_history = []
        self.volume_history = []
    def update(self, price: Decimal, volume: Decimal) -> None:
        self.price_history.append(price)
        self.volume_history.append(volume)
    def is_crash_detected(self) -> bool:
        if len(self.price_history) < 2:
            return False
        price_change = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
        return price_change <= -self.price_threshold
    def get_crash_magnitude(self) -> Decimal:
        if len(self.price_history) < 2:
            return Decimal('0')
        return abs((self.price_history[-1] - self.price_history[0]) / self.price_history[0])
