import os
import asyncio
from decimal import Decimal
from ccxt.async_support import binance
from dotenv import load_dotenv

load_dotenv()


class BinanceConnector:
    def _format_quantity(self, value, precision=8):
        """
        Format a float or Decimal value as a decimal string, avoiding scientific notation, with the given precision.
        """
        if value is None:
            return None
        fmt = f"{{0:.{precision}f}}"
        s = fmt.format(float(value))
        s = s.rstrip("0").rstrip(".") if "." in s else s
        return s

    def __init__(self):
        self.exchange = binance(
            {
                "apiKey": os.getenv("BINANCE_API_KEY"),
                "secret": os.getenv("BINANCE_API_SECRET"),
                "options": {"defaultType": "spot", "adjustForTimeDifference": True},
                "enableRateLimit": True,
            }
        )

    async def get_order_book(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Récupère le carnet d'ordres de Binance"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol)
            bid = (
                Decimal(str(orderbook["bids"][0][0]))
                if len(orderbook["bids"]) > 0
                else Decimal(0)
            )
            ask = (
                Decimal(str(orderbook["asks"][0][0]))
                if len(orderbook["asks"]) > 0
                else Decimal("Infinity")
            )
            return bid, ask
        except Exception as e:
            raise Exception(f"Binance error: {str(e)}")

    async def create_order(
        self,
        symbol: str,
        side: str,
        usdc_amount: Decimal = None,
        price: Decimal = None,
        use_usdc: bool = True,
    ):
        """
        Crée un ordre sur Binance
        Si use_usdc=True, achète pour un montant en USDC (quoteOrderQty).
        Sinon, achète pour une quantité en BTC (amount).
        """
        try:
            params = {
                "type": "market" if not price else "limit",
            }
            amount = None
            if use_usdc and usdc_amount is not None:
                params["quoteOrderQty"] = self._format_quantity(
                    usdc_amount, precision=8
                )
            elif usdc_amount is not None:
                params["amount"] = self._format_quantity(usdc_amount, precision=8)
                amount = self._format_quantity(usdc_amount, precision=8)
            if price:
                params["price"] = self._format_quantity(price, precision=8)

            return await self.exchange.create_order(
                symbol,
                "market" if not price else "limit",
                side,
                amount,
                float(params["price"]) if price else None,
                params,
            )
        except Exception as e:
            raise Exception(f"Binance order error: {str(e)}")

    async def fetch_ticker(self, symbol: str):
        """Ajoute fetch_ticker pour compatibilité arbitrage"""
        return await self.exchange.fetch_ticker(symbol)

    async def get_ticker(self, symbol: str):
        """Retourne un ticker au format attendu"""
        ticker = await self.exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "bid": ticker.get("bid", 0),
            "ask": ticker.get("ask", 0),
            "last": ticker.get("last", 0),
            "timestamp": ticker.get("timestamp"),
        }

    async def close(self):
        """Ferme la connexion"""
        await self.exchange.close()
