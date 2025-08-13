from typing import Dict, List
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
from ..exchanges.base_exchange import BaseExchange
class MarketDataCollector:
    def __init__(self, exchange: BaseExchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
    async def get_ticker_data(self, symbol: str) -> Dict:
        cache_key = f"ticker_{symbol}"
        # Vérification du cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
                return cached_data
        try:
            ticker_data = await asyncio.to_thread(self.exchange.get_ticker, symbol)
            return ticker_data
        except Exception as e:
            self.logger.error(f"Erreur get_ticker_data: {str(e)}")
            raise
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start_time: datetime, end_time: datetime) -> pd.DataFrame:
        try:
            # Conversion des paramètres pour l'API
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'start': int(start_time.timestamp() * 1000),
                'end': int(end_time.timestamp() * 1000)
            }
            # Récupération des données historiques
            historical_data = await asyncio.to_thread(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe,
                params['start'],
                params['end']
            )
            # Conversion en DataFrame
            df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Erreur get_historical_data: {str(e)}")
            raise
    def save_to_csv(self, data: pd.DataFrame, filename: str):
        try:
            data.to_csv(filename)
            self.logger.info(f"Données sauvegardées dans {filename}")
        except Exception as e:
            self.logger.error(f"Erreur save_to_csv: {str(e)}")
            raise
