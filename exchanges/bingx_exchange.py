"""
BingX Exchange Module
Handles all interactions with BingX API (Futures Trading)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import ccxt.async_support as ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BingXExchange:
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize BingXExchange
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self._exchange = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the exchange connection"""
        try:
            self._exchange = ccxt.bingx(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap", "adjustForTimeDifference": True},
                }
            )
            await self._exchange.load_markets()
            self._initialized = True
            logger.info("BingXExchange initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BingXExchange: {e}")
            raise

    async def close(self) -> None:
        """Close exchange connection"""
        if self._exchange:
            await self._exchange.close()
            self._initialized = False

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
        Returns:
            Dict containing ticker data
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return {
                "last": str(ticker.get("last", 0)),
                "bid": str(ticker.get("bid", 0)),
                "ask": str(ticker.get("ask", 0)),
                "high": str(ticker.get("high", 0)),
                "low": str(ticker.get("low", 0)),
                "baseVolume": str(ticker.get("baseVolume", 0)),
                "quoteVolume": str(ticker.get("quoteVolume", 0)),
                "timestamp": ticker.get("timestamp", 0),
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Alias direct pour compatibilité (ex: scan d'arbitrage multi-broker)
        """
        return await self.get_ticker(symbol)

    async def get_klines(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[List[float]]:
        """
        Get OHLCV klines/candlesticks
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to fetch
        Returns:
            List of OHLCV data
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            klines = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return klines
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            raise

    async def get_orderbook(self, symbol: str) -> Dict[str, List[List[float]]]:
        """
        Get current orderbook
        Args:
            symbol: Trading pair symbol
        Returns:
            Dict containing bids and asks
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            orderbook = await self._exchange.fetch_order_book(symbol)
            return {"bids": orderbook["bids"], "asks": orderbook["asks"]}
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            raise

    async def fetch_order_book(self, symbol: str) -> Dict[str, List[List[float]]]:
        """
        Alias direct pour compatibilité (ex: scan d'arbitrage multi-broker)
        """
        return await self.get_orderbook(symbol)

    async def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance
        Returns:
            Dict containing balance information
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            balance = await self._exchange.fetch_balance()
            return {
                currency: data
                for currency, data in balance.items()
                if isinstance(data, dict) and float(data.get("total", 0)) > 0
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get open positions
        Returns:
            List of open positions
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            positions = await self._exchange.fetch_positions()
            return [pos for pos in positions if float(pos.get("contracts", 0)) != 0]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    async def get_my_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get trade history
        Args:
            symbol: Trading pair symbol
        Returns:
            List of trades
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            trades = await self._exchange.fetch_my_trades(symbol)
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise

    async def set_leverage(
        self, symbol: str, leverage: int, position_side: str = "LONG"
    ) -> bool:
        """
        Set leverage for a symbol
        Args:
            symbol: Trading pair symbol
            leverage: Leverage value (e.g., 1-125)
            position_side: Position side ('LONG', 'SHORT', or 'BOTH')
        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            # S'assurer que position_side est en majuscules et valide
            position_side = position_side.upper()
            if position_side not in ["LONG", "SHORT", "BOTH"]:
                raise ValueError("position_side must be one of: LONG, SHORT, BOTH")
            await self._exchange.set_leverage(
                leverage, symbol, params={"side": position_side}
            )
            return True
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            raise

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Dict = None,
    ) -> Dict[str, Any]:
        """
        Create a new order (compatible spot/futures selon l'API CCXT BingX)
        Args:
            symbol: Trading pair symbol
            order_type: Type of order ('market' or 'limit')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (required for limit orders)
            params: Additional parameters for the order
        Returns:
            Order information
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            params = params or {}
            # Ajouter le positionSide par défaut pour les futures
            if "positionSide" not in params:
                position_side = "LONG" if side.lower() == "buy" else "SHORT"
                params["positionSide"] = position_side
            if float(amount) <= 0:
                raise ValueError("Amount must be positive")
            if order_type == "limit":
                if not price or float(price) <= 0:
                    raise ValueError("Valid price required for limit orders")
                order = await self._exchange.create_limit_order(
                    symbol, side, float(amount), float(price), params
                )
            else:
                order = await self._exchange.create_market_order(
                    symbol, side, float(amount), None, params
                )
            return order
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an order
        Args:
            symbol: Trading pair symbol
            order_id: ID of the order to cancel
        Returns:
            True if successful
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            raise

    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Get order information
        Args:
            symbol: Trading pair symbol
            order_id: Order ID
        Returns:
            Order information
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            order = await self._exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
            raise

    async def get_historical_data(
        self, pairs: List[str], timeframes: List[str], period: str
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Récupère les données OHLCV historiques pour chaque paire/timeframe sur la période demandée.
        Retourne {timeframe: {pair: pd.DataFrame}}
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        result = {}
        try:
            # Parse period (ex: "7d" -> 7 jours)
            if period.endswith("d"):
                days = int(period.replace("d", ""))
                since = int(
                    (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
                )
            else:
                # Fallback: 1 jour
                since = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000)
            for tf in timeframes:
                tf_result = {}
                for pair in pairs:
                    try:
                        ohlcv = await self._exchange.fetch_ohlcv(pair, tf, since=since)
                        df = pd.DataFrame(
                            ohlcv,
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ],
                        )
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        tf_result[pair] = df
                    except Exception as e:
                        logger.error(f"Erreur historique {pair} {tf}: {e}")
                        tf_result[pair] = pd.DataFrame()
                result[tf] = tf_result
            return result
        except Exception as e:
            logger.error(f"Erreur get_historical_data: {e}")
            raise

    # -------------------- AJOUT : Méthodes pour arbitrage cross-exchange --------------------
    async def withdraw(
        self,
        code: str,
        amount: float,
        address: str,
        tag: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Effectue un retrait d'actif via l'API BingX.
        code: Asset code (ex: 'USDT')
        amount: Montant à retirer
        address: Adresse de destination
        tag: Tag/Memo (optionnel)
        params: Paramètres additionnels (optionnel)
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            params = params or {}
            result = await self._exchange.withdraw(code, amount, address, tag, params)
            return result
        except Exception as e:
            logger.error(f"Error withdrawing {amount} {code} to {address}: {e}")
            raise

    async def get_deposit_address(
        self, asset: str, params: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Récupère l'adresse de dépôt pour un actif (ex: 'USDT').
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            params = params or {}
            result = await self._exchange.fetch_deposit_address(asset, params)
            return result
        except Exception as e:
            logger.error(f"Error fetching deposit address for {asset}: {e}")
            raise

    async def get_deposit_history(
        self, asset: Optional[str] = None, params: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Suivi des dépôts pour un actif (optionnel: asset).
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            params = params or {}
            result = await self._exchange.fetch_deposits(code=asset, params=params)
            return result
        except Exception as e:
            logger.error(f"Error fetching deposit history for {asset}: {e}")
            raise
