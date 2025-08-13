"""
"""
from typing import Dict, List
import numpy as np
from datetime import datetime
class MarketUtils:
    @staticmethod
    def calculate_optimal_amount(orderbook: Dict, min_profit: float) -> float:
        """Calcule le montant optimal pour un trade"""
        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])
        # Calcul du spread moyen
        spread = (bids[:, 0].mean() / asks[:, 0].mean() - 1) * 100
        if spread < min_profit:
            return 0
        # Calcul du volume disponible
        available_volume = min(
            bids[:, 1].sum(),
            asks[:, 1].sum()
        )
        # Ajustement basé sur le spread
        return available_volume * (spread / min_profit)
    @staticmethod
    def validate_opportunity(opp: Dict, min_profit: float, min_volume: float) -> bool:
        """Valide une opportunité d'arbitrage"""
        return (
            opp['spread'] >= min_profit and
            opp['volume'] >= min_volume and
            opp.get('timestamp') and
            isinstance(opp['timestamp'], datetime)
        )
