import ccxt
import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict

load_dotenv()
logger = logging.getLogger(__name__)

BROKER_SETTINGS = {
    "binance": {
        "quote": "USDC",
        "key_var": "BINANCE_API_KEY",
        "secret_var": "BINANCE_API_SECRET",
        "params": {"options": {"adjustForTimeDifference": True}},
    },
    "gateio": {
        "quote": "USDT",
        "key_var": "GATEIO_API_KEY",
        "secret_var": "GATEIO_API_SECRET",
    },
    "bingx": {
        "quote": "USDT",
        "key_var": "BINGX_API_KEY",
        "secret_var": "BINGX_API_SECRET",
    },
    "blofin": {
        "quote": "USDT",
        "key_var": "BLOFIN_API_KEY",
        "secret_var": "BLOFIN_API_SECRET",
    },
    "okx": {
        "quote": "USDT",
        "key_var": "OKX_API_KEY",
        "secret_var": "OKX_API_SECRET",
        "passphrase_var": "OKX_PASSPHRASE",
    },
}


class BrokerManager:
    def __init__(self):
        self.brokers: Dict[str, ccxt.Exchange] = {}
        self._initialize_all()

    def _initialize_all(self):
        for name, config in BROKER_SETTINGS.items():
            self._init_broker(name, config)

    def _init_broker(self, name: str, config: dict):
        try:
            params = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
            if "params" in config:
                params.update(config["params"])
            api_key = os.getenv(config["key_var"])
            api_secret = os.getenv(config["secret_var"])
            if api_key and api_secret:
                params.update({"apiKey": api_key, "secret": api_secret})
                if "passphrase_var" in config:
                    params["password"] = os.getenv(config["passphrase_var"])
            self.brokers[name] = getattr(ccxt, name)(params)
        except Exception as e:
            logger.warning(f"Initialisation {name} échouée: {str(e)}")


class USDCArbitrage:
    """Gestion professionnelle de l'arbitrage USDC multi-broker"""

    def __init__(self, pairs: List[str], broker_name: str = "binance"):
        self.broker_manager = BrokerManager()
        self.set_broker(broker_name)
        self.pairs = self._validate_pairs(pairs)
        self.min_spread = 0.0005  # 0.05%
        self.timeout = 10

    def set_broker(self, broker_name: str):
        """Change de broker dynamiquement"""
        if broker_name in self.broker_manager.brokers:
            self.exchange = self.broker_manager.brokers[broker_name]
            self.quote_asset = BROKER_SETTINGS[broker_name]["quote"]
        else:
            raise ValueError(f"Broker {broker_name} non configuré")

    def available_brokers(self) -> list:
        """Liste les brokers disponibles"""
        return list(BROKER_SETTINGS.keys())

    def _validate_pairs(self, pairs: List[str]) -> List[str]:
        """Validation des paires avec gestion d'erreur"""
        try:
            markets = self.exchange.load_markets()
            return [p for p in pairs if p in markets and markets[p]["active"]]
        except Exception as e:
            logger.error(f"Erreur chargement markets: {str(e)}")
            return []

    async def _fetch_order_book(self, pair: str):
        """Récupération robuste du carnet d'ordres"""
        try:
            return await asyncio.wait_for(
                self.exchange.fetch_order_book(pair, limit=5), timeout=self.timeout
            )
        except Exception as e:
            logger.debug(f"Erreur sur {pair}: {str(e)}")
            return None

    async def _scan_pair(self, pair: str) -> Tuple[bool, float]:
        """Calcul précis du spread"""
        ob = await self._fetch_order_book(pair)
        if ob and ob["bids"] and ob["asks"]:
            bid = ob["bids"][0][0]
            ask = ob["asks"][0][0]
            spread = (ask - bid) / bid
            return spread > self.min_spread, spread
        return False, 0.0

    async def scan_async(self) -> List[Tuple[str, float]]:
        """Scan asynchrone optimisé"""
        tasks = [self._scan_pair(pair) for pair in self.pairs]
        results = await asyncio.gather(*tasks)
        return [
            (pair, round(spread, 6))
            for pair, (found, spread) in zip(self.pairs, results)
            if found
        ]

    def scan_all_pairs(self) -> List[Tuple[str, float]]:
        """Interface synchrone"""
        if not self.pairs:
            return []
        return asyncio.run(self.scan_async())

    def get_balance(self, currency: str = "USDC") -> float:
        """Récupération sécurisée du solde"""
        try:
            balance = self.exchange.fetch_balance()
            return balance.get(currency, {}).get("free", 0)
        except Exception as e:
            logger.error(f"Erreur balance: {str(e)}")
            return 0.0
