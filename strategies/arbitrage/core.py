#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import logging
import asyncio
from dotenv import load_dotenv

# Importe tes wrappers custom
from connectors.binance_exchange import BinanceExchange
from connectors.okx_exchange import OKXExchange
from connectors.blofin import BlofinConnector
from connectors.gateio_exchange import GateIOExchange
from connectors.bingx_exchange import BingXExchange

# Si tu n’as pas de wrapper custom pour un exchange, utilise ccxt.async_support directement
# import ccxt.async_support as ccxt

# Chargement des variables d'environnement
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))


class ExchangeManager:
    def __init__(self, brokers):
        self.exchanges = brokers


class ArbitrageEngine:
    def __init__(self):
        self._init_logger()
        self.exchange_manager = None  # sera initialisé en async

    def _init_logger(self):
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def async_init(self):
        """Initialise tous les brokers en mode asynchrone avec wrappers custom"""
        brokers = {
            "binance": BinanceExchange(
                os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
            ),
            "okx": OKXExchange(
                os.getenv("OKX_API_KEY"),
                os.getenv("OKX_API_SECRET"),
                os.getenv("OKX_PASSPHRASE"),
            ),
            "blofin": BlofinConnector(
                os.getenv("BLOFIN_API_KEY"), os.getenv("BLOFIN_API_SECRET")
            ),
            "gateio": GateIOExchange(
                os.getenv("GATEIO_API_KEY"), os.getenv("GATEIO_API_SECRET")
            ),
            "bingx": BingXExchange(
                os.getenv("BINGX_API_KEY"), os.getenv("BINGX_API_SECRET")
            ),
        }
        for name, broker in brokers.items():
            try:
                await broker.initialize()
                self.logger.info(f"Broker {name} initialisé avec succès")
            except Exception as e:
                self.logger.error(f"Erreur initialisation {name}: {str(e)}")
                brokers[name] = None
        self.exchange_manager = ExchangeManager(brokers)


# Méthodes existantes à conserver...
