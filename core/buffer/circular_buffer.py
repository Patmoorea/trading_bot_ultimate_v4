import time
import logging
logger = logging.getLogger(__name__)
class CircularBuffer:
    """Buffer circulaire optimisé"""
    def __init__(self, maxlen=1000, compress=True):
        self.maxlen = maxlen
        self.compress = compress
        self.data = {}
        self.timestamps = {}
        logger.info(f"Buffer circulaire initialisé (taille max: {maxlen}, compression: {compress})")
    def update_data(self, symbol: str, data: dict):
        """Mise à jour des données avec compression optionnelle"""
        try:
            if symbol not in self.data:
                self.data[symbol] = []
                self.timestamps[symbol] = []
            if self.compress:
                # Compression des données avant stockage
                compressed_data = self._compress_data(data)
                self.data[symbol].append(compressed_data)
            else:
                self.data[symbol].append(data)
            self.timestamps[symbol].append(time.time())
            # Respect de la taille maximale
            while len(self.data[symbol]) > self.maxlen:
                self.data[symbol].pop(0)
                self.timestamps[symbol].pop(0)
        except Exception as e:
            logger.error(f"Erreur mise à jour buffer: {e}")
    def _compress_data(self, data: dict) -> dict:
        """Compression des données"""
        try:
            compressed = {}
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    compressed[key] = round(value, 6)  # Réduction précision
                elif isinstance(value, dict):
                    compressed[key] = self._compress_data(value)
                else:
                    compressed[key] = value
            return compressed
        except Exception as e:
            logger.error(f"Erreur compression: {e}")
            return data
    def get_latest_data(self, symbol: str = None):
        """Récupère les dernières données pour un symbole"""
        try:
            if symbol:
                return self.data.get(symbol, [])[-1] if self.data.get(symbol) else None
            return {sym: data[-1] if data else None for sym, data in self.data.items()}
        except Exception as e:
            logger.error(f"Erreur récupération données: {e}")
            return None
    def get_all_data(self, symbol: str = None):
        """Récupère toutes les données stockées"""
        try:
            if symbol:
                return self.data.get(symbol, [])
            return self.data
        except Exception as e:
            logger.error(f"Erreur récupération données: {e}")
            return {}
    def get_latest_ohlcv(self, symbol: str):
        """Récupère les dernières données OHLCV pour un symbole"""
        try:
            data = self.get_latest_data(symbol)
            if data and all(k in data for k in ['open', 'high', 'low', 'close', 'volume']):
                return {
                    'timestamp': data.get('timestamp', time.time()),
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume']
                }
            return None
        except Exception as e:
            logger.error(f"Erreur récupération OHLCV: {e}")
            return None
    def get_orderbook(self, symbol: str):
        """Récupère le dernier orderbook pour un symbole"""
        try:
            data = self.get_latest_data(symbol)
            if data and 'orderbook' in data:
                return data['orderbook']
            return None
        except Exception as e:
            logger.error(f"Erreur récupération orderbook: {e}")
            return None
    def get_volume_profile(self, symbol: str = None):
        """Récupère le profil de volume"""
        try:
            data = self.get_all_data(symbol) if symbol else self.data
            volume_profile = {}
            for sym, sym_data in data.items():
                if sym_data:
                    prices = [d.get('close', 0) for d in sym_data if 'close' in d]
                    volumes = [d.get('volume', 0) for d in sym_data if 'volume' in d]
                    for price, vol in zip(prices, volumes):
                        if price > 0:
                            price_level = round(price, 2)
                            if price_level not in volume_profile:
                                volume_profile[price_level] = 0
                            volume_profile[price_level] += vol
            return volume_profile
        except Exception as e:
            logger.error(f"Erreur calcul profil volume: {e}")
            return {}
