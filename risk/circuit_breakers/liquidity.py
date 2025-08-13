from decimal import Decimal
from datetime import datetime
class LiquidityShockDetector:
    def __init__(self, min_depth_btc: float = 10.0, depth_drop_threshold: float = 0.5):
        self.min_depth = Decimal(str(min_depth_btc))
        self.threshold = Decimal(str(depth_drop_threshold))
        self.status = "NORMAL"
    def is_shock_detected(self, orderbook: dict) -> bool:
        total_depth = sum(amount for _, amount in orderbook['bids'][:3])
        if total_depth < self.min_depth:
            self.status = "SHOCK"
            return True
        return False
    def get_current_status(self) -> str:
        return self.status
