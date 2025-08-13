#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import logging
import ccxt
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
# Chargement des variables d'environnement
load_dotenv(os.path.join(os.path.dirname(__file__), '../../../.env'))
class ArbitrageEngine:
    def __init__(self):
        """Initialise le moteur d'arbitrage avec tous les brokers configurés"""
        self._init_logger()
        self.brokers = self._init_all_brokers()
        self.logger.info("ArbitrageEngine initialisé avec succès")
    def _init_logger(self):
        """Configure le système de logging"""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    def _init_all_brokers(self):
        """Initialise tous les brokers avec leurs configurations spécifiques"""
        brokers_config = {
            'binance': {
                'class': ccxt.binance,
                'params': {
                    'apiKey': os.getenv('BINANCE_API_KEY'),
                    'secret': os.getenv('BINANCE_API_SECRET'),
                    'options': {'defaultType': 'spot'}
                }
            },
            'okx': {
                'class': ccxt.okx,
                'params': {
                    'apiKey': os.getenv('OKX_API_KEY'),
                    'secret': os.getenv('OKX_API_SECRET'),
                    'password': os.getenv('OKX_PASSPHRASE')
                }
            },
            'blofin': {
                'class': ccxt.blofin,
                'params': {
                    'apiKey': os.getenv('BLOFIN_API_KEY'),
                    'secret': os.getenv('BLOFIN_API_SECRET')
                }
            },
            'gateio': {
                'class': ccxt.gateio,
                'params': {
                    'apiKey': os.getenv('GATEIO_API_KEY'),
                    'secret': os.getenv('GATEIO_API_SECRET')
                }
            },
            'bingx': {
                'class': ccxt.bingx,
                'params': {
                    'apiKey': os.getenv('BINGX_API_KEY'),
                    'secret': os.getenv('BINGX_API_SECRET')
                }
            }
        }
        brokers = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._init_broker, name, config)
                for name, config in brokers_config.items()
            }
            for future in futures:
                name, broker = future.result()
                brokers[name] = broker
        return brokers
    def _init_broker(self, name, config):
        """Initialise un broker individuel avec gestion d'erreur"""
        try:
            broker = config['class'](config['params'])
            broker.load_markets()
            self.logger.info(f"Broker {name} initialisé avec succès")
            return name, broker
        except Exception as e:
            self.logger.error(f"Erreur initialisation {name}: {str(e)}")
            return name, None
# Méthodes existantes à conserver...
