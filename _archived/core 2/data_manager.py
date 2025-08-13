"""
Gestion des données market
"""
from pandas import DataFrame
from binance.spot import Spot
from config.paths import Paths
class DataManager:
    def __init__(self, client: Spot):
        self.client = client
        self.cache = {}
    def get_klines(self, symbol: str, interval: str, limit: int = 500):
        try:
            klines = self.client.klines(symbol, interval, limit=limit)
            df = DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
