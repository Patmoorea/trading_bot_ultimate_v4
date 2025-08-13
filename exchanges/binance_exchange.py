"""
Binance Exchange Module
Handles all interactions with Binance API (Spot Trading)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import ccxt.async_support as ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceExchange:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._exchange = None
        self._initialized = False

    async def initialize(self):
        try:
            self._exchange = ccxt.binance(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "spot",
                        "adjustForTimeDifference": True,
                        "testnet": self.testnet,
                    },
                }
            )
            logger.info(
                f"BinanceExchange: testnet={self.testnet}, defaultType={self._exchange.options.get('defaultType')}"
            )
            if self.testnet:
                logger.warning("Binance testnet: PAS de load_markets !")
                self._exchange.urls["api"] = {
                    "web": "https://testnet.binance.vision",
                    "rest": "https://testnet.binance.vision",
                }
                self._exchange.load_markets = lambda *a, **k: None
            else:
                logger.info(">>> AVANT load_markets")
                await self._exchange.load_markets()
                logger.info(">>> APRES load_markets")

            self._initialized = True
            logger.info("BinanceExchange initialized successfully")

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to initialize BinanceExchange: {e}\n{traceback.format_exc()}"
            )
            raise

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
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
        Alias for compatibility with multi-broker scanner.
        """
        return await self.get_ticker(symbol)

    async def get_balance(self) -> Dict[str, Any]:
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

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._initialized = False

    async def get_klines(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[List[float]]:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            klines = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return klines
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            raise

    async def get_orderbook(self, symbol: str) -> Dict[str, List[List[float]]]:
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
        Alias for compatibility with multi-broker scanner.
        """
        return await self.get_orderbook(symbol)

    async def get_my_trades(self, symbol: str) -> List[Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            trades = await self._exchange.fetch_my_trades(symbol)
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Crée un ordre spot réel sur Binance.
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            params = {}
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
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            raise

    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
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
        PATCH: Ne retourne jamais None, toujours un DataFrame (éventuellement vide), jamais d'objet non-attendu.
        PATCH: Garantit que fetch_ohlcv est toujours awaitable, même en testnet ou mock.
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        result = {}
        try:
            if period.endswith("d"):
                days = int(period.replace("d", ""))
                since = int(
                    (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
                )
            else:
                since = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000)
            for tf in timeframes:
                tf_result = {}
                for pair in pairs:
                    try:
                        fetch_ohlcv = getattr(self._exchange, "fetch_ohlcv", None)
                        klines = []
                        if fetch_ohlcv is not None:
                            try:
                                if asyncio.iscoroutinefunction(fetch_ohlcv):
                                    klines = await fetch_ohlcv(pair, tf, since=since)
                                else:
                                    klines = fetch_ohlcv(pair, tf, since=since)
                            except Exception as e:
                                logger.error(
                                    f"[{pair}][{tf}] fetch_ohlcv exception: {e}"
                                )
                                klines = []
                        else:
                            logger.warning(
                                f"[{pair}][{tf}] fetch_ohlcv absent (probablement testnet)"
                            )
                            klines = []
                        logger.info(
                            f"[DEBUG][{pair}][{tf}] klines type: {type(klines)} len: {len(klines) if klines else 0}"
                        )
                        if klines and len(klines) > 0:
                            logger.info(f"[DEBUG][{pair}][{tf}] klines[0]: {klines[0]}")
                        if not klines or len(klines) == 0:
                            logger.error(f"Aucune donnée historique pour {pair} {tf}")
                            tf_result[pair] = pd.DataFrame(
                                columns=[
                                    "timestamp",
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                ]
                            )
                            continue

                        valid_klines = [
                            row
                            for row in klines
                            if isinstance(row, (list, tuple)) and len(row) == 6
                        ]
                        if len(valid_klines) < len(klines):
                            logger.warning(
                                f"[{pair}][{tf}] {len(klines) - len(valid_klines)} bougies invalides ignorées"
                            )

                        df = pd.DataFrame(
                            valid_klines,
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
                        logger.info(
                            f"[DEBUG][{pair}][{tf}] df.columns: {list(df.columns)} shape: {df.shape}"
                        )
                    except Exception as e:
                        logger.error(f"Erreur historique {pair} {tf}: {e}")
                        tf_result[pair] = pd.DataFrame(
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]
                        )
                result[tf] = tf_result
            return result
        except Exception as e:
            logger.error(f"Erreur get_historical_data: {e}")
            for tf in timeframes:
                if tf not in result:
                    result[tf] = {}
                for pair in pairs:
                    if pair not in result[tf]:
                        result[tf][pair] = pd.DataFrame(
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]
                        )
            return result

    # -------------- AJOUT : Méthodes pour arbitrage cross-exchange --------------
    async def withdraw(
        self,
        code: str,
        amount: float,
        address: str,
        tag: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Effectue un retrait d'actif via l'API Binance.
        code: Asset code (ex: 'BTC')
        amount: Montant à retirer
        address: Adresse de destination
        tag: Tag/Memo (optionnel)
        params: Paramètres additionnels (optionnel)
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            # CCXT: withdraw
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
        Récupère l'adresse de dépôt pour un actif (ex: 'BTC').
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
            # CCXT: fetch_deposits
            result = await self._exchange.fetch_deposits(code=asset, params=params)
            return result
        except Exception as e:
            logger.error(f"Error fetching deposit history for {asset}: {e}")
            raise

    # create_order déjà présent plus haut et compatible spot réel
    # (utilise order_type "market"/"limit", side "buy"/"sell", amount, price)
