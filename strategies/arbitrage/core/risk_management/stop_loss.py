import logging
from decimal import Decimal
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class StopLossManager:
    def __init__(self, global_stop_loss: float = 10.0, 
                 per_trade_stop_loss: float = 2.0,
                 trailing_stop: float = 1.0):
        self.global_stop_loss = Decimal(str(global_stop_loss))
        self.per_trade_stop_loss = Decimal(str(per_trade_stop_loss))
        self.trailing_stop = Decimal(str(trailing_stop))
        
        self.trades = {}
        self.global_stop_triggered = False
        self.total_loss = Decimal('0')

    def register_trade(self, trade_id: str, pair: str, exchange_a: str, 
                      exchange_b: str, value: Decimal) -> None:
        """Enregistre un nouveau trade pour le suivi des stop loss"""
        self.trades[trade_id] = {
            'pair': pair,
            'exchange_a': exchange_a,
            'exchange_b': exchange_b,
            'value': value,
            'high_water_mark': value
        }

    def update_trade_result(self, trade_id: str, profit_loss: float) -> None:
        """Met à jour le résultat d'un trade"""
        if trade_id in self.trades:
            pl_decimal = Decimal(str(profit_loss))
            if pl_decimal < Decimal('0'):
                self.total_loss -= pl_decimal
            del self.trades[trade_id]

    def is_trade_allowed(self, pair: str, exchange_a: str, exchange_b: str) -> bool:
        """Vérifie si un trade est autorisé selon les stop loss"""
        if self.global_stop_triggered:
            return False
            
        # Vérifier le stop loss global
        if self.total_loss > self.global_stop_loss:
            self.global_stop_triggered = True
            logger.warning("Stop loss global déclenché")
            return False
            
        return True

    def force_global_stop(self) -> None:
        """Force l'activation du stop loss global"""
        self.global_stop_triggered = True
        logger.warning("Stop loss global forcé")

    def get_status(self) -> Dict:
        """Retourne l'état actuel des stop loss"""
        return {
            'global_stop_triggered': self.global_stop_triggered,
            'total_loss': float(self.total_loss),
            'active_trades': len(self.trades)
        }
