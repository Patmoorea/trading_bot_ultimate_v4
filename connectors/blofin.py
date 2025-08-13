import os
import hmac
import hashlib
import time
import requests
from typing import Tuple
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()


class BlofinConnector:
    def __init__(self, api_key=None, api_secret=None):
        self.base_url = "https://openapi.blofin.com"
        self.api_key = api_key or os.getenv("BLOFIN_API_KEY")
        self.api_secret = api_secret or os.getenv("BLOFIN_API_SECRET")

    def _generate_signature(self, timestamp, method, endpoint, body=""):
        message = str(timestamp) + method.upper() + endpoint + str(body)
        return hmac.new(
            self.api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    async def get_order_book(self, symbol: str) -> Tuple[Decimal, Decimal]:
        endpoint = f"/api/v1/market/orderbook?instId={symbol}&sz=1"
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, "GET", endpoint)

        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(self.base_url + endpoint, headers=headers)
            data = response.json()
            if data["code"] == "0":
                bids = data["data"]["bids"]
                asks = data["data"]["asks"]
                return Decimal(bids[0][0]), Decimal(asks[0][0])
            return Decimal(0), Decimal("Infinity")
        except Exception:
            return Decimal(0), Decimal("Infinity")

    async def create_order(
        self, symbol: str, side: str, amount: Decimal, price: Decimal = None
    ):
        endpoint = "/api/v1/trade/order"
        timestamp = str(int(time.time() * 1000))

        order_data = {
            "instId": symbol,
            "tdMode": "cash",
            "side": side.lower(),
            "ordType": "market" if not price else "limit",
            "sz": str(amount),
        }

        if price:
            order_data["px"] = str(price)

        signature = self._generate_signature(timestamp, "POST", endpoint, order_data)

        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.base_url + endpoint, json=order_data, headers=headers
        )
        return response.json()

    # 🟢 Ajoute cette méthode pour compatibilité arbitrage async
    async def get_ticker(self, symbol: str):
        """
        Retourne le ticker du symbole (bid/ask moyen, simulé pour compatibilité).
        """
        bid, ask = await self.get_order_book(symbol)
        # Simule un format à la CCXT/async exchanges
        return {
            "symbol": symbol,
            "bid": float(bid),
            "ask": float(ask),
            "last": (
                float((bid + ask) / 2) if bid > 0 and ask < Decimal("Infinity") else 0
            ),
            "timestamp": int(time.time() * 1000),
        }

    async def fetch_ticker(self, symbol: str):
        """
        Ajoute la méthode fetch_ticker pour compatibilité avec l'arbitrage multi-broker.
        """
        return await self.get_ticker(symbol)
