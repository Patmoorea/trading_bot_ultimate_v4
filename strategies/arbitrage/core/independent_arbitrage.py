import ccxt
import os
from time import sleep
class IndependentUSDCArbitrage:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True
        })
        self.threshold = float(os.getenv('ARBITRAGE_THRESHOLD', 0.5))
    def check(self):
        try:
            usdc = self.exchange.fetch_order_book('BTC/USDC')
            usdt = self.exchange.fetch_order_book('BTC/USDT')
            spread = (usdc['bids'][0][0] / usdt['asks'][0][0] - 1) * 100
            return spread if spread > self.threshold else None
        except Exception as e:
            print(f"Erreur: {str(e)}")
            return None
