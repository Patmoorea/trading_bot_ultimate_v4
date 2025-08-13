import logging
from typing import Dict
from ccxt import Exchange
class ArbitrageEngine:
    def __init__(self, exchange=None):
        self.exchange = exchange or self._create_exchange()
        self.logger = logging.getLogger(__name__)
    def _create_exchange(self):
        from ccxt import binance
        return binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    async def check_opportunities(self, symbol: str) -> Dict:
        """Check arbitrage opportunities for given symbol"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol)
            return {
                'best_ask': orderbook['asks'][0][0],
                'best_bid': orderbook['bids'][0][0],
                'spread': orderbook['asks'][0][0] - orderbook['bids'][0][0],
                'timestamp': self.exchange.milliseconds()
            }
        except Exception as e:
            self.logger.error(f"Arbitrage check error: {str(e)}")
            return {}
    async def check_opportunities_v2(self, symbol: str) -> Dict:
        """Enhanced version with spread percentage and liquidity"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol)
            spread = orderbook['asks'][0][0] - orderbook['bids'][0][0]
            bid_price = orderbook['bids'][0][0]
            return {
                'spread': spread,
                'spread_pct': (spread / bid_price) * 100 if bid_price else 0,
                'liquidity': sum(b[1] for b in orderbook['bids'][:5]),
                'timestamp': self.exchange.milliseconds()
            }
        except Exception as e:
            self.logger.error(f"Arbitrage v2 check error: {str(e)}")
            return {}
