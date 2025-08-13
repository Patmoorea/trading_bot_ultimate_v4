#!/usr/bin/env python3
from .core import ArbitrageEngine
from .pairs_config import get_symbol, PAIRS_CONFIG
import numpy as np
class AdvancedArbitrageEngine(ArbitrageEngine):
    def find_arbitrage_opportunity(self, base_asset: str) -> dict:
        """Version corrigée avec support Blofin"""
        opportunities = []
        for broker_name, broker in self.brokers.items():
            if broker_name not in PAIRS_CONFIG:
                continue
            try:
                symbol = get_symbol(broker_name, base_asset)
                orderbook = broker.fetch_order_book(symbol)
                opportunities.append({
                    'broker': broker_name,
                    'bid': orderbook['bids'][0][0],
                    'ask': orderbook['asks'][0][0],
                    'symbol': symbol
                })
            except Exception as e:
                print(f"[SKIP] {broker_name} {base_asset}: {str(e)}")
                continue
        if len(opportunities) < 2:
            return {}
        best_buy = min(opportunities, key=lambda x: x['ask'])
        best_sell = max(opportunities, key=lambda x: x['bid'])
        spread = best_sell['bid'] - best_buy['ask']
        spread_pct = spread / best_buy['ask']
        return {
            'best_buy': best_buy,
            'best_sell': best_sell, 
            'spread': spread,
            'spread_pct': spread_pct,
            'timestamp': np.datetime64('now')
        }
