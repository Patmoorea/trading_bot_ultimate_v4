"""
Gate.io Exchange Module optimized for Spot Arbitrage
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import ccxt.async_support as ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class GateIOExchange:
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize GateIOExchange
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
        """
        self.api_key = api_key or os.getenv("GATEIO_API_KEY")
        self.api_secret = api_secret or os.getenv("GATEIO_API_SECRET")
        self._exchange = None
        self._initialized = False
        self._markets_info = {}
        self._min_trade_amounts = {}
        self._trading_fees = {}
        if not self.api_key or not self.api_secret:
            raise ValueError("Gate.io API credentials not properly configured")

    async def initialize(self) -> None:
        """Initialize the exchange connection"""
        try:
            self._exchange = ccxt.gateio(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "spot",
                        "adjustForTimeDifference": True,
                        "createMarketBuyOrderRequiresPrice": True,
                        "fetchMinOrderAmounts": True,
                        "fetchTradingFees": True,
                        "recvWindow": 60000,
                    },
                }
            )
            await self._exchange.load_markets()
            await self._load_trading_fees()
            await self._load_min_amounts()
            self._initialized = True
            logger.info("GateIOExchange initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GateIOExchange: {e}")
            raise

    async def _load_trading_fees(self) -> None:
        """Charge les frais de trading pour chaque paire"""
        try:
            self._trading_fees = await self._exchange.fetch_trading_fees()
            logger.info(f"Loaded trading fees for {len(self._trading_fees)} pairs")
        except Exception as e:
            logger.error(f"Error loading trading fees: {e}")
            self._trading_fees = {}

    async def _load_min_amounts(self) -> None:
        """Charge les montants minimums de trading"""
        for symbol, market in self._exchange.markets.items():
            self._min_trade_amounts[symbol] = (
                market.get("limits", {}).get("amount", {}).get("min")
            )

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
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
        Alias for compatibility with multi-broker arbitrage scanner
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
        Alias for compatibility with multi-broker arbitrage scanner
        """
        return await self.get_orderbook(symbol)

    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance"""
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

    async def get_my_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get trade history
        Args:
            symbol: Trading pair symbol
        """
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
        Create a new order
        Args:
            symbol: Trading pair symbol
            order_type: Type of order ('market' or 'limit')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (required for limit orders)
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            if float(amount) <= 0:
                raise ValueError("Amount must be positive")
            if order_type == "limit":
                if not price or float(price) <= 0:
                    raise ValueError("Valid price required for limit orders")
                order = await self._exchange.create_limit_order(
                    symbol, side, float(amount), float(price)
                )
            else:
                order = await self._exchange.create_market_order(
                    symbol, side, float(amount)
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
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            order = await self._exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
            raise

    async def get_arbitrage_opportunities(
        self, symbols: List[str], min_profit_percent: float = 0.1
    ) -> List[Dict]:
        """
        Trouve les opportunités d'arbitrage entre les paires données
        Args:
            symbols: Liste des paires à analyser
            min_profit_percent: Profit minimum recherché en pourcentage
        """
        opportunities = []
        orderbooks = {}
        for symbol in symbols:
            try:
                orderbooks[symbol] = await self.get_orderbook(symbol)
            except Exception as e:
                logger.error(f"Error fetching orderbook for {symbol}: {e}")
                continue
        for symbol in symbols:
            try:
                ticker = await self.get_ticker(symbol)
                best_ask = float(ticker["ask"])
                best_bid = float(ticker["bid"])
                # Calculer le spread et les frais
                spread = (best_bid - best_ask) / best_ask * 100
                fees = self._get_total_fees(symbol)
                potential_profit = spread - fees
                if potential_profit > min_profit_percent:
                    opportunities.append(
                        {
                            "symbol": symbol,
                            "ask_price": best_ask,
                            "bid_price": best_bid,
                            "spread": spread,
                            "fees": fees,
                            "potential_profit": potential_profit,
                        }
                    )
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        return opportunities

    def _get_total_fees(self, symbol: str) -> float:
        """
        Calcule les frais totaux pour une paire
        Args:
            symbol: Trading pair symbol
        """
        if symbol in self._trading_fees:
            maker_fee = self._trading_fees[symbol].get("maker", 0.002)
            taker_fee = self._trading_fees[symbol].get("taker", 0.002)
            return (maker_fee + taker_fee) * 100  # Convertir en pourcentage
        return 0.4  # Frais par défaut si non disponible (0.2% + 0.2%)

    async def execute_arbitrage(self, opportunity: Dict) -> Dict:
        """
        Exécute une opportunité d'arbitrage
        Args:
            opportunity: Dictionnaire contenant les détails de l'opportunité
        """
        symbol = opportunity["symbol"]
        try:
            # Vérifier le solde avant l'exécution
            balance = await self.get_balance()
            usdt_balance = float(balance.get("USDT", {}).get("free", 0))
            if usdt_balance < 10:  # Minimum 10 USDT pour l'arbitrage
                raise ValueError("Insufficient USDT balance for arbitrage")
            # Calculer le montant optimal pour l'arbitrage
            min_amount = self._min_trade_amounts.get(symbol, 0)
            amount = max(
                min_amount, (usdt_balance * 0.1) / opportunity["ask_price"]
            )  # Utilise 10% du solde
            # Exécuter les ordres d'arbitrage
            buy_order = await self.create_order(
                symbol=symbol,
                order_type="limit",
                side="buy",
                amount=amount,
                price=opportunity["ask_price"] * 1.001,
            )
            sell_order = await self.create_order(
                symbol=symbol,
                order_type="limit",
                side="sell",
                amount=amount,
                price=opportunity["bid_price"] * 0.999,
            )
            return {
                "status": "success",
                "buy_order": buy_order,
                "sell_order": sell_order,
            }
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def close(self) -> None:
        """Close exchange connection"""
        if self._exchange:
            await self._exchange.close()
            self._initialized = False

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
        Effectue un retrait d'actif via l'API Gate.io.
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
