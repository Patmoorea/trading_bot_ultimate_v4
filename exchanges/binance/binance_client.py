from decimal import Decimal
from typing import Dict, Optional
from ..base_exchange import BaseExchange
import ccxt
import logging
from typing import List


class BinanceClient(BaseExchange):
    def __init__(self, api_key: str, api_secret: str):
        super().__init__(api_key, api_secret)
        self.logger = logging.getLogger(__name__)
        self._initialize_exchange()

    def get_24h_ticker(self, symbol):
        # symbol ex: "BTC/USDC"
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "volume": ticker.get("baseVolume", 0),
                "priceChangePercent": ticker.get("percentage", 0),
            }
        except Exception as e:
            self.logger.warning(f"get_24h_ticker error for {symbol}: {e}")
            raise NotImplementedError("No method for 24h ticker")

    def get_ticker_price(self, symbol: str):
        # symbol ex: "BTC/USDC"
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            self.logger.warning(f"get_ticker_price error for {symbol}: {e}")
            raise NotImplementedError("No method to get latest price")

    def _initialize_exchange(self):
        self.exchange = ccxt.binance(
            {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        self.exchange.load_markets()

    def get_ticker(self, symbol: str) -> Dict:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "bid": Decimal(str(ticker["bid"])),
                "ask": Decimal(str(ticker["ask"])),
                "last": Decimal(str(ticker["last"])),
                "volume": Decimal(str(ticker["baseVolume"])),
            }
        except Exception as e:
            self.logger.error(f"Erreur Binance get_ticker: {str(e)}")
            raise

    def get_balance(self) -> Dict:
        try:
            balance = self.exchange.fetch_balance()
            if not balance or "total" not in balance:
                raise ValueError("Balance invalide ou vide")
            # Convertir en format API Binance standard
            balances = []
            for currency, info in balance.items():
                if isinstance(info, dict) and info.get("total", 0) > 0:
                    balances.append(
                        {
                            "asset": currency,
                            "free": str(info.get("free", 0)),
                            "locked": str(info.get("used", 0)),
                        }
                    )
            return {"balances": balances}
        except Exception as e:
            self.logger.error(f"Erreur Binance get_balance: {str(e)}")
            raise

    def place_order(
        self, symbol: str, side: str, amount: Decimal, price: Optional[Decimal] = None
    ) -> Dict:
        try:
            order_type = "market" if price is None else "limit"
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=float(amount),
                price=float(price) if price else None,
            )
            return order
        except Exception as e:
            self.logger.error(f"Erreur Binance place_order: {str(e)}")
            raise

    def get_order_book(self, symbol: str) -> Dict:
        try:
            book = self.exchange.fetch_order_book(symbol)
            return {
                "bids": [
                    [Decimal(str(price)), Decimal(str(amount))]
                    for price, amount in book["bids"]
                ],
                "asks": [
                    [Decimal(str(price)), Decimal(str(amount))]
                    for price, amount in book["asks"]
                ],
            }
        except Exception as e:
            self.logger.error(f"Erreur Binance get_order_book: {str(e)}")
            raise

    def get_open_orders(self, symbol: str = None) -> List:
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Erreur get_open_orders: {e}")
            return []
