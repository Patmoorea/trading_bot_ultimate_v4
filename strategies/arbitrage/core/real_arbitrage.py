from typing import Dict, List, Tuple, Union
import ccxt.async_support as ccxt
import asyncio
import logging
from ..base import BaseStrategy


class USDCArbitrage(BaseStrategy):
    """
    Version finale ultra-robuste avec gestion complète des API Binance (async)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_spread = float(config.get("min_spread", 0.002))
        self.timeout = int(config.get("timeout", 30000))  # 30s par défaut
        self.logger = logging.getLogger(__name__)
        self.exchanges = asyncio.get_event_loop().run_until_complete(
            self._init_exchanges(config.get("exchanges", ["binance"]))
        )
        self.exchange = next(iter(self.exchanges.values())) if self.exchanges else None

    async def _init_exchanges(
        self, exchange_names: List[str]
    ) -> Dict[str, ccxt.Exchange]:
        exchanges = {}
        for name in exchange_names:
            try:
                exchange = getattr(ccxt, name)(
                    {
                        "enableRateLimit": True,
                        "timeout": self.timeout,
                        "options": {
                            "defaultType": "spot",
                            "adjustForTimeDifference": True,
                        },
                    }
                )
                exchanges[name] = exchange
                self.logger.info(f"Exchange {name} initialisé avec succès")
            except Exception as e:
                self.logger.error(f"Échec initialisation {name}: {str(e)}")
        return exchanges

    def _extract_price(self, price_data) -> float:
        """Extrait le prix de n'importe quel format de données"""
        try:
            if isinstance(price_data, (list, tuple)) and len(price_data) >= 1:
                return float(price_data[0])
            elif isinstance(price_data, dict):
                for key in ["price", "bid", "ask", "last"]:
                    if key in price_data:
                        return float(price_data[key])
            return float(price_data)
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Erreur extraction prix: {str(e)}")
            return 0.0

    async def _safe_fetch_prices(
        self, exchange, symbol: str, retries: int = 3
    ) -> tuple:
        """Récupération ultra-robuste des prix avec fallback"""
        methods = [
            self._fetch_via_order_book,
            self._fetch_via_ticker,
            self._fetch_via_trades,
        ]
        for attempt in range(retries):
            for method in methods:
                try:
                    return await method(exchange, symbol)
                except Exception as e:
                    self.logger.warning(
                        f"Tentative {attempt+1}: Méthode {method.__name__} échouée pour {symbol}: {str(e)}"
                    )
                    await asyncio.sleep(1)
        raise ConnectionError(f"Échec après {retries} tentatives pour {symbol}")

    async def _fetch_via_order_book(self, exchange, symbol: str) -> tuple:
        data = await exchange.fetch_order_book(symbol, {"limit": 1})
        bid = self._extract_price(data["bids"][0]) if data.get("bids") else 0
        ask = self._extract_price(data["asks"][0]) if data.get("asks") else 0
        if bid <= 0 or ask <= 0 or bid >= ask:
            raise ValueError("Prix invalides dans l'order book")
        return bid, ask

    async def _fetch_via_ticker(self, exchange, symbol: str) -> tuple:
        ticker = await exchange.fetch_ticker(symbol)
        bid = self._extract_price(ticker["bid"])
        ask = self._extract_price(ticker["ask"])
        if bid <= 0 or ask <= 0 or bid >= ask:
            raise ValueError("Prix invalides dans le ticker")
        return bid, ask

    async def _fetch_via_trades(self, exchange, symbol: str) -> tuple:
        trades = await exchange.fetch_trades(symbol, limit=10)
        if not trades:
            raise ValueError("Aucun trade disponible")
        prices = [self._extract_price(t["price"]) for t in trades]
        mid_price = sum(prices) / len(prices)
        spread = mid_price * 0.001  # Estimation du spread
        return mid_price - spread / 2, mid_price + spread / 2

    async def scan_all_pairs(self) -> Dict[str, float]:
        """Scan principal avec gestion d'erreur complète"""
        opportunities = {}
        for name, exchange in self.exchanges.items():
            try:
                markets = await exchange.load_markets()
                for symbol in markets:
                    if symbol.endswith("/USDC") and markets[symbol].get("active"):
                        try:
                            bid, ask = await self._safe_fetch_prices(exchange, symbol)
                            spread = (ask - bid) / ask
                            if spread > self.min_spread:
                                opportunities[f"{name}:{symbol}"] = spread
                        except Exception as e:
                            self.logger.warning(f"Erreur traitement {symbol}: {str(e)}")
                            continue
            except Exception as e:
                self.logger.error(f"Erreur exchange {name}: {str(e)}")
                continue
        return opportunities

    async def get_opportunities(self) -> List[Tuple[str, float]]:
        return [
            (pair, spread)
            for pair, spread in (await self.scan_all_pairs()).items()
            if spread > self.min_spread
        ]

    def switch_broker(self, broker: str):
        if broker in self.exchanges:
            self.exchange = self.exchanges[broker]
            self.logger.info(f"Exchange changé pour {broker}")
