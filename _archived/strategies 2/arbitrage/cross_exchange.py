from dataclasses import dataclass
from typing import List, Dict
@dataclass
class ArbitrageOpportunity:
    pair: str
    exchanges: Dict[str, float]
    spread: float
class CrossExchangeArbitrage:
    def __init__(self, exchanges: List[str]):
        self.exchanges = exchanges
    def get_prices(self, pair: str) -> Dict[str, float]:
        """Simule la récupération des prix sur différents exchanges"""
        return {exchange: 50000 * (1 + i*0.001) for i, exchange in enumerate(self.exchanges)}
    def find_opportunities(self, pair: str = "BTC/USDT") -> List[ArbitrageOpportunity]:
        prices = self.get_prices(pair)
        avg_price = sum(prices.values()) / len(prices)
        opportunities = []
        for exchange, price in prices.items():
            spread = abs(price - avg_price) / avg_price
            if spread > 0.002:  # Seuil de 0.2%
                opportunities.append(
                    ArbitrageOpportunity(
                        pair=pair,
                        exchanges={exchange: price},
                        spread=spread
                    )
                )
        return opportunities
