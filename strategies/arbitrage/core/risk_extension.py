"""
OFFICIAL RISK EXTENSION - 100% compatible avec votre structure
"""
from typing import Tuple
import logging
class RiskExtension:
    def __init__(self, arbitrage_core):
        self.core = arbitrage_core  # Votre USDCArbitrage existant
        self.log = logging.getLogger('RiskExt')
    def validate(self, pair: str) -> Tuple[bool, str]:
        """Validation non intrusive"""
        try:
            spread = getattr(self.core, '_calculate_spread')(pair)
            liquidity = self._get_liquidity(pair)
            return (spread >= 0.002 and liquidity > 1000), ""
        except Exception as e:
            self.log.error(f"Validation error: {str(e)}")
            return False, "error"
    def _get_liquidity(self, pair: str) -> float:
        """Utilise UNIQUEMENT l'API publique"""
        ob = self.core.fetch_order_book(pair)
        return sum(bid[1] for bid in ob['bids'][:3])
