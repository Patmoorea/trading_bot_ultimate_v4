from typing import Dict, List
import numpy as np
class OrderExecutionOptimizer:
    """Optimiseur d'exécution qui travaille AVEC le module existant"""
    def __init__(self, arbitrage_module):
        self.arbitrage = arbitrage_module
        self.liquidity_cache: Dict[str, float] = {}
    def calculate_optimal_size(self, pair: str, spread: float) -> float:
        """Calcule la taille d'ordre optimale basée sur la liquidité"""
        liquidity = self._get_liquidity(pair)
        volatility = self._get_volatility(pair)
        return min(liquidity * 0.1, 1000)  # Ne dépasse pas 10% de la liquidité
    def _get_liquidity(self, pair: str) -> float:
        """Récupère la liquidité du cache ou la calcule"""
        if pair not in self.liquidity_cache:
            self.liquidity_cache[pair] = self._fetch_liquidity(pair)
        return self.liquidity_cache[pair]
