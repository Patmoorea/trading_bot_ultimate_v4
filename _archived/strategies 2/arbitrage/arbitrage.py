import ccxt
import os
import time
from dotenv import load_dotenv
load_dotenv()
class BaseUSDCArbitrage:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        if not all([self.api_key, self.api_secret]):
            raise ValueError("Configuration API manquante dans .env")
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.threshold = float(os.getenv('ARBITRAGE_THRESHOLD', 0.3))
    def get_spread(self):
        try:
            usdc = self.exchange.fetch_order_book('BTC/USDC')
            usdt = self.exchange.fetch_order_book('BTC/USDT')
            best_bid_usdc = usdc['bids'][0][0]
            best_ask_usdt = usdt['asks'][0][0]
            return (best_bid_usdc / best_ask_usdt - 1) * 100
        except Exception as e:
            print(f"Erreur de marché: {str(e)}")
            return None
    def monitor(self, interval=10):
        print("\n=== Surveillance Arbitrage USDC/USDT ===")
        print(f"Seuil configuré: {self.threshold}%")
        print("Appuyez sur Ctrl+C pour arrêter\n")
        while True:
            try:
                spread = self.get_spread()
                if spread is not None:
                    if spread > self.threshold:
                        print(f"🚨 Opportunité: {spread:.4f}%")
                    else:
                        print(f"Spread actuel: {spread:.4f}%", end='\r')
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nArrêt du monitoring")
                break
