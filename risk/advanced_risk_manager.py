from datetime import datetime
class AdvancedRiskMonitor:
    """Extension de la gestion des risques sans modifier l'existant"""
    def __init__(self, arbitrage_module):
        self.arbitrage = arbitrage_module
        self.daily_loss_limit = 0.02  # 2%
        self.today_loss = 0.0
    def check_trade_approval(self, amount: float, pair: str) -> bool:
        """Vérifie si le trade est approuvé selon les règles de risque"""
        self._reset_daily_counter()
        spread = self.arbitrage._calculate_spread(pair)
        potential_loss = amount * spread
        return self.today_loss + potential_loss < self.daily_loss_limit
    def _reset_daily_counter(self):
        """Réinitialise le compteur quotidien"""
            self.today_loss = 0.0
