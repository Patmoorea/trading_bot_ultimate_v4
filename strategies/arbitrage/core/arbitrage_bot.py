import time
import sys
import os
from pathlib import Path

from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner
from src.utils.notifications import send_arbitrage_alert

class ArbitrageBot:
    def __init__(self):
        self.scanner = ArbitrageScanner(
            min_profit=0.002,
            max_price_deviation=0.05,
            pairs=["BTC/USDT", "ETH/USDT"],
            max_trade_size=1000,
            timeout=5,
            volume_filter=1000,
            price_check=True,
            max_slippage=0.0005
        )
        
    async def get_best_spread(self):
        """Trouve la meilleure opportunité d'arbitrage"""
        try:
            opportunities = await self.scanner.find_opportunities()
            if opportunities:
                # Trier par profit décroissant
                return max(opportunities, key=lambda x: x['profit_pct'])
            return None
        except Exception as e:
            print(f"Erreur lors de la recherche d'opportunités: {e}")
            return None
