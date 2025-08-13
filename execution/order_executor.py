import ccxt
import asyncio
import os
from dotenv import load_dotenv
from typing import Dict, Any
load_dotenv()
class OrderExecutor:
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError("Configurez vos clés API dans .env")
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })
    def execute_order(self, order: Dict[str, Any]):
        """Version synchrone pour simplifier"""
        try:
            symbol = order['symbol'].replace('/', '')
            return self.exchange.create_order(
                symbol,
                'limit',
                order['action'],
                order['amount'],
                order['price'],
                params={'test': True}  # Mode test d'abord
            )
        except Exception as e:
            print(f"Erreur d'exécution: {e}")
            return None
def main():
    executor = OrderExecutor()
    test_order = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'amount': 0.01,
        'price': 50000
    }
    result = executor.execute_order(test_order)
    print("Résultat:", result)
if __name__ == '__main__':
    main()
