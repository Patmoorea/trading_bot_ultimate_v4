import ccxt
import os
import time
from dotenv import load_dotenv
load_dotenv()
class EnhancedArbitrage:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.threshold = float(os.getenv('ARBITRAGE_THRESHOLD', 0.3))
        if not all([self.api_key, self.api_secret]):
            raise ValueError("Configuration API manquante dans .env")
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True
        })
    def get_current_spread(self):
        """Nouvelle méthode pour récupérer le spread actuel"""
        usdc = self.exchange.fetch_order_book('BTC/USDC')
        usdt = self.exchange.fetch_order_book('BTC/USDT')
        return (usdc['bids'][0][0] / usdt['asks'][0][0] - 1) * 100
    def check_opportunity(self):
        """Vérifie si le spread dépasse le seuil"""
        spread = self.get_current_spread()
        return spread if spread > self.threshold else None
    def monitor(self, interval=10):
        """Surveillance continue avec gestion des erreurs"""
        while True:
            try:
                spread = self.get_current_spread()
                if spread > self.threshold:
                    print(f"\033[92mARBITRAGE: {spread:.4f}%\033[0m")
                else:
                    print(f"Spread actuel: {spread:.4f}%", end='\r')
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nMonitoring arrêté")
                break
            except Exception as e:
                print(f"\033[91mErreur: {str(e)}\033[0m")
                time.sleep(30)
def enhanced_check(self):
    """Compare avec 3 exchanges (Binance, Gate.io, OKX)"""
    spreads = []
    for exchange in [self.exchange, ccxt.gateio(), ccxt.okx()]:
        try:
            usdc = exchange.fetch_order_book('BTC/USDC')
            usdt = exchange.fetch_order_book('BTC/USDT')
            spreads.append((usdc['bids'][0][0] / usdt['asks'][0][0] - 1) * 100)
        except:
            continue
    return max(spreads) if spreads else None
