from decimal import Decimal
from typing import Dict
from src.config.settings import Settings
class RiskManager:
    def __init__(self):
        self.settings = Settings
        self.open_positions: Dict = {}
    def can_open_position(self, symbol: str, amount: Decimal, 
                         current_price: Decimal) -> bool:
        position_value = amount * current_price
        total_exposure = sum(pos['value'] for pos in self.open_positions.values())
        return (position_value / total_exposure <= self.settings.MAX_POSITION_SIZE 
                if total_exposure > 0 else True)
    def calculate_position_size(self, account_balance: Decimal, 
                              entry_price: Decimal, stop_loss: Decimal) -> Decimal:
        risk_amount = account_balance * Decimal(str(self.settings.RISK_PERCENTAGE))
        price_diff = abs(entry_price - stop_loss)
        return (risk_amount / price_diff).quantize(Decimal('0.00001'))
