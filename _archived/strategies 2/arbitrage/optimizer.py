from typing import Dict, List
from dataclasses import dataclass
import numpy as np
@dataclass
class BrokerConstraint:
    quote_asset: str
    fee: float
    min_volume: float
    adjustment_factor: float = 1.0
class CrossExchangeOptimizer:
    def __init__(self, brokers):
        self.brokers = brokers
        self.constraints = {
            'binance': BrokerConstraint(
                quote_asset='USDC',
                fee=0.001,
                min_volume=0.001
            ),
            'okx': BrokerConstraint(
                quote_asset='USDT', 
                fee=0.0008,
                min_volume=0.002,
                adjustment_factor=1.0002
            ),
            'blofin': BrokerConstraint(
                quote_asset='USD',
                fee=0.0005,
                min_volume=0.005
            )
        }
    def get_optimal_pair(self, base_asset: str) -> Dict:
        """Trouve la meilleure paire cross-exchange"""
        opportunities = []
        for broker_name, broker in self.brokers.items():
            if broker_name not in self.constraints:
                continue
            constraint = self.constraints[broker_name]
            symbol = f"{base_asset}/{constraint.quote_asset}"
            try:
                orderbook = broker.fetch_order_book(symbol)
                effective_bid = orderbook['bids'][0][0] * constraint.adjustment_factor * (1 - constraint.fee)
                effective_ask = orderbook['asks'][0][0] * (1 + constraint.fee)
                opportunities.append({
                    'broker': broker_name,
                    'bid': effective_bid,
                    'ask': effective_ask,
                    'volume': min(orderbook['bids'][0][1], orderbook['asks'][0][1]),
                    'symbol': symbol
                })
            except Exception as e:
                print(f"Error processing {broker_name}: {str(e)}")
                continue
        if not opportunities:
            return {}
        best_buy = min(opportunities, key=lambda x: x['ask'])
        best_sell = max(opportunities, key=lambda x: x['bid'])
        return {
            'spread': best_sell['bid'] - best_buy['ask'],
            'buy': best_buy,
            'sell': best_sell,
            'timestamp': np.datetime64('now')
        }
