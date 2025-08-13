from datetime import datetime
from typing import Dict, Optional, Tuple
class RiskManager:
    """
    Core risk management module for trading operations.
    """
    def __init__(self, 
                 max_position_size: float = 0.1,
                 stop_loss_pct: float = 0.02,
                 max_drawdown: float = 0.1,
                 risk_per_trade: float = 0.01):
        """Initialize risk management parameters"""
        self.max_position_size = max_position_size  # 10% max of portfolio
        self.stop_loss_pct = stop_loss_pct         # 2% stop loss
        self.max_drawdown = max_drawdown           # 10% max drawdown
        self.risk_per_trade = risk_per_trade       # 1% risk per trade
        self.current_drawdown = 0.0
        self.positions: Dict[str, float] = {}
    def calculate_position_size(self, 
                              capital: float, 
                              price: float, 
                              volatility: Optional[float] = None) -> float:
        """Calculate safe position size based on capital and risk parameters"""
        # Base position size calculation
        max_position = capital * self.max_position_size
        risk_based_size = (capital * self.risk_per_trade) / self.stop_loss_pct
        # Determine initial position size
        position_size = min(max_position, risk_based_size)
        if volatility is not None:
            # Pour garantir que la taille soit toujours réduite avec la volatilité
            # On utilise un facteur de réduction minimum de 0.95 même pour une faible volatilité
            volatility_factor = min(0.95, 1.0 / (1.0 + volatility))
            position_size = position_size * volatility_factor
            # Réduction supplémentaire pour haute volatilité
            if volatility > 0.5:
                position_size *= 0.9  # 10% de réduction supplémentaire
        return position_size
    def check_risk_limits(self, 
                         unrealized_pnl: float, 
                         capital: float) -> Tuple[bool, str]:
        """Check if current risk levels are within acceptable limits"""
        # Update drawdown
        self.current_drawdown = max(self.current_drawdown, -unrealized_pnl / capital)
        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown:
            return False, f"Max drawdown exceeded: {self.current_drawdown:.2%}"
        # Check total position exposure
        total_exposure = sum(self.positions.values())
        if total_exposure > self.max_position_size:
            return False, f"Max position size exceeded: {total_exposure:.2%}"
        return True, "Risk levels acceptable"
    def add_position(self, symbol: str, size: float) -> None:
        """Track new position"""
        self.positions[symbol] = self.positions.get(symbol, 0) + size
    def remove_position(self, symbol: str) -> None:
        """Remove closed position"""
        if symbol in self.positions:
            del self.positions[symbol]
    def get_stop_loss_price(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss price for a position"""
        if is_long:
            return entry_price * (1 - self.stop_loss_pct)
        return entry_price * (1 + self.stop_loss_pct)
    def reset_drawdown(self) -> None:
        """Reset drawdown tracking (e.g., at start of new period)"""
        self.current_drawdown = 0.0
