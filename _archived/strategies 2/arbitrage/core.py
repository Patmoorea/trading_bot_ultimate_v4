import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from ..base import BaseStrategy
@dataclass
class ArbitrageOpportunity:
    pair: str
    exchange_a: str
    exchange_b: str
    spread: float
    volume: float
    timestamp: float
class ArbitrageEngine(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.min_spread = config.get('min_spread', 0.005)  # 0.5%
        self.max_volume = config.get('max_volume', 10.0)  # 10 BTC
    def find_opportunities(self, orderbooks: Dict) -> List[ArbitrageOpportunity]:
        """
        Détecte les opportunités d'arbitrage entre exchanges
        """
        opportunities = []
        for pair, books in orderbooks.items():
            for exchange_a, book_a in books.items():
                for exchange_b, book_b in books.items():
                    if exchange_a != exchange_b:
                        bid_a = book_a['bids'][0][0]
                        ask_b = book_b['asks'][0][0]
                        spread = (bid_a - ask_b) / ask_b
                        if spread > self.min_spread:
                            volume = min(book_a['bids'][0][1], book_b['asks'][0][1])
                            if volume >= self.max_volume:
                                opp = ArbitrageOpportunity(
                                    pair=pair,
                                    exchange_a=exchange_a,
                                    exchange_b=exchange_b,
                                    spread=spread,
                                    volume=volume,
                                    timestamp=time.time()
                                )
                                opportunities.append(opp)
        return opportunities
