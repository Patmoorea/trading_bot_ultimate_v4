import logging
import numpy as np
logger = logging.getLogger(__name__)
class CircularBuffer:
    def __init__(self, maxlen=1000, compress=True):
        """
        Initialise un buffer circulaire pour stocker les données
        Args:
            maxlen (int): Taille maximale du buffer
            compress (bool): Active la compression des données
        """
        self.buffer = {}
        self.trades = {}
        self.orderbooks = {}
        self.klines = {}
        self.maxlen = maxlen
        self.compress = compress
        logger.info(f"Buffer circulaire initialisé (taille max: {maxlen}, compression: {compress})")
    def update_data(self, symbol: str, data: dict):
        """Met à jour les données pour un symbole"""
        if symbol not in self.buffer:
            self.buffer[symbol] = []
        self.buffer[symbol].append(data)
        # Maintien de la taille maximale
        if len(self.buffer[symbol]) > self.maxlen:
            self.buffer[symbol].pop(0)
    def update_trades(self, trade_data: dict):
        """Met à jour les données de trades"""
        symbol = trade_data['symbol']
        if symbol not in self.trades:
            self.trades[symbol] = []
        self.trades[symbol].append(trade_data)
        # Maintien de la taille maximale
        if len(self.trades[symbol]) > self.maxlen:
            self.trades[symbol].pop(0)
    def update_orderbook(self, orderbook_data: dict):
        """Met à jour les données d'orderbook"""
        symbol = orderbook_data['symbol']
        self.orderbooks[symbol] = orderbook_data
    def update_klines(self, kline_data: dict):
        """Met à jour les données de klines"""
        symbol = kline_data['symbol']
        if symbol not in self.klines:
            self.klines[symbol] = []
        self.klines[symbol].append(kline_data)
        # Maintien de la taille maximale
        if len(self.klines[symbol]) > self.maxlen:
            self.klines[symbol].pop(0)
    def get_latest(self, symbol: str) -> dict:
        """Récupère les dernières données pour un symbole"""
        if symbol not in self.buffer or not self.buffer[symbol]:
            return None
        return self.buffer[symbol][-1]
    def get_orderbook(self, symbol: str) -> dict:
        """Récupère le dernier orderbook pour un symbole"""
        return self.orderbooks.get(symbol)
    def get_latest_trades(self, symbol: str, n: int = None) -> list:
        """Récupère les n derniers trades pour un symbole"""
        if symbol not in self.trades:
            return []
        if n is None:
            return self.trades[symbol]
        return self.trades[symbol][-n:]
    def get_latest_klines(self, symbol: str, n: int = None) -> list:
        """Récupère les n dernières klines pour un symbole"""
        if symbol not in self.klines:
            return []
        if n is None:
            return self.klines[symbol]
        return self.klines[symbol][-n:]
