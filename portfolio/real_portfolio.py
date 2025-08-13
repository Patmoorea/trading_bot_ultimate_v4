import logging
from typing import Dict, Optional
logger = logging.getLogger(__name__)
class RealPortfolio:
    def __init__(self, initial_balance: float = 0.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trade_history = []
        logger.info(f"Portfolio initialized with balance: {initial_balance}")
    def get_balance(self) -> float:
        return self.current_balance
    def get_position(self, symbol: str) -> Dict:
        return self.positions.get(symbol, {
            'amount': 0.0,
            'entry_price': 0.0
        })
    def update_balance(self, new_balance: float) -> None:
        self.current_balance = new_balance
        logger.info(f"Balance updated to: {new_balance}")
    def update_position(self, symbol: str, amount: float, price: float) -> None:
        if symbol not in self.positions:
            self.positions[symbol] = {
                'amount': amount,
                'entry_price': price
            }
        else:
            self.positions[symbol]['amount'] += amount
            # Mettre à jour le prix d'entrée moyen si on ajoute à la position
            if amount > 0:
                total_amount = self.positions[symbol]['amount']
                old_cost = (total_amount - amount) * self.positions[symbol]['entry_price']
                new_cost = amount * price
                self.positions[symbol]['entry_price'] = (old_cost + new_cost) / total_amount
        logger.info(f"Position updated for {symbol}: Amount={amount}, Price={price}")
