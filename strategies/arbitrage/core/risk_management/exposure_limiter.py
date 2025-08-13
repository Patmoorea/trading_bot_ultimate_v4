import logging
from decimal import Decimal
from typing import Dict, Set

logger = logging.getLogger(__name__)

class ExposureLimiter:
    def __init__(self, max_exposure: float = 50.0,
                 max_trades: int = 5,
                 max_concurrent_exchanges: int = 3):
        self.max_exposure = Decimal(str(max_exposure))
        self.max_trades = max_trades
        self.max_concurrent_exchanges = max_concurrent_exchanges
        
        self.current_exposure = Decimal('0')
        self.active_trades: Dict[str, Dict] = {}
        self.active_exchanges: Set[str] = set()
        self.account_balance = Decimal('0')

    def update_account_balance(self, balance: float) -> None:
        """Met à jour le solde du compte"""
        self.account_balance = Decimal(str(balance))

    def can_take_new_trade(self, exchange_a: str, exchange_b: str) -> bool:
        """Vérifie si un nouveau trade est possible"""
        # Vérifier le nombre de trades actifs
        if len(self.active_trades) >= self.max_trades:
            return False
            
        # Vérifier l'exposition totale
        max_exposure_value = self.account_balance * self.max_exposure / Decimal('100')
        if self.current_exposure >= max_exposure_value:
            return False
            
        # Vérifier le nombre d'exchanges
        potential_exchanges = self.active_exchanges | {exchange_a, exchange_b}
        if len(potential_exchanges) > self.max_concurrent_exchanges:
            return False
            
        return True

    def register_trade(self, trade_id: str, exchange_a: str, 
                      exchange_b: str, value: Decimal) -> None:
        """Enregistre un nouveau trade"""
        self.active_trades[trade_id] = {
            'exchange_a': exchange_a,
            'exchange_b': exchange_b,
            'value': value
        }
        self.active_exchanges.add(exchange_a)
        self.active_exchanges.add(exchange_b)
        self.current_exposure += value

    def close_trade(self, trade_id: str) -> None:
        """Clôture un trade"""
        if trade_id in self.active_trades:
            trade = self.active_trades[trade_id]
            self.current_exposure -= trade['value']
            del self.active_trades[trade_id]
            
            # Nettoyer les exchanges inactifs
            self._cleanup_exchanges()

    def _cleanup_exchanges(self) -> None:
        """Nettoie la liste des exchanges actifs"""
        active = set()
        for trade in self.active_trades.values():
            active.add(trade['exchange_a'])
            active.add(trade['exchange_b'])
        self.active_exchanges = active

    def reduce_limits(self, factor: Decimal) -> None:
        """Réduit les limites d'exposition"""
        self.max_exposure *= factor
        logger.info(f"Limites d'exposition réduites à {float(self.max_exposure):.2f}%")

    def get_status(self) -> Dict:
        """Retourne l'état actuel des limites d'exposition"""
        return {
            'current_exposure': float(self.current_exposure),
            'max_exposure': float(self.max_exposure),
            'active_trades': len(self.active_trades),
            'active_exchanges': len(self.active_exchanges)
        }
