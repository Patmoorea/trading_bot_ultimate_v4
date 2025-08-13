import ccxt
import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import List, Tuple
# Charge les variables d'environnement
load_dotenv()
logger = logging.getLogger(__name__)
class USDCArbitrage:
    """Gestion professionnelle de l'arbitrage USDC"""
    def __init__(self, pairs: List[str], exchange_name: str = 'binance'):
        self.exchange = self._init_exchange(exchange_name)
        self.pairs = self._validate_pairs(pairs)
        self.min_spread = 0.0005  # 0.05%
        self.timeout = 10
    def _init_exchange(self, name: str):
        """Initialisation sécurisée de l'exchange"""
        exchange_class = getattr(ccxt, name.lower())
        # Configuration de base
        config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        # Récupération sécurisée des clés
        api_key = os.getenv(f'{name.upper()}_API_KEY')
        api_secret = os.getenv(f'{name.upper()}_API_SECRET')
        if not api_key or not api_secret:
            logger.warning(f"Clés API manquantes pour {name}, mode public seulement")
        else:
            config.update({
                'apiKey': api_key,
                'secret': api_secret,
            })
            # Configurations spécifiques
            if name.lower() == 'binance':
                if os.getenv('BINANCE_TESTNET', '').lower() == 'true':
                    config.update({
                        'apiKey': os.getenv('BINANCE_TESTNET_KEY'),
                        'secret': os.getenv('BINANCE_TESTNET_SECRET'),
                    })
                config['options']['adjustForTimeDifference'] = True
            elif name.lower() == 'okx':
                config['password'] = os.getenv('OKX_PASSPHRASE')
        return exchange_class(config)
    def _validate_pairs(self, pairs: List[str]) -> List[str]:
        """Validation des paires avec gestion d'erreur"""
        try:
            markets = self.exchange.load_markets()
            return [
                p for p in pairs 
                if p in markets 
                and markets[p]['active']
        except Exception as e:
            logger.error(f"Erreur chargement markets: {str(e)}")
            return []
    async def _fetch_order_book(self, pair: str):
        """Récupération robuste du carnet d'ordres"""
        try:
            return await asyncio.wait_for(
                self.exchange.fetch_order_book(pair, limit=5),
                timeout=self.timeout
            )
        except Exception as e:
            logger.debug(f"Erreur sur {pair}: {str(e)}")
            return None
    async def _scan_pair(self, pair: str) -> Tuple[bool, float]:
        """Calcul précis du spread"""
        ob = await self._fetch_order_book(pair)
        if ob and ob['bids'] and ob['asks']:
            bid = ob['bids'][0][0]
            ask = ob['asks'][0][0]
            spread = (ask - bid) / bid  # Formule standard
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
    def get_balance(self, currency: str = 'USDC') -> float:
        """Récupération sécurisée du solde"""
        try:
            balance = self.exchange.fetch_balance()
            return balance.get(currency, {}).get('free', 0)
        except Exception as e:
            logger.error(f"Erreur balance: {str(e)}")
            return 0.0
# ============ DÉBUT AJOUT MULTI-BROKER ============
config.brokers_config.BROKER_CONFIG = {
    'binance': {
        'quote_asset': 'USDC',
        'api_key_env': 'BINANCE_API_KEY',
        'api_secret_env': 'BINANCE_API_SECRET',
        'special_params': {'options': {'adjustForTimeDifference': True}}
    },
    'gateio': {
        'quote_asset': 'USDT',
        'api_key_env': 'GATEIO_API_KEY',
        'api_secret_env': 'GATEIO_API_SECRET'
    },
    'bingx': {
        'quote_asset': 'USDT',
        'api_key_env': 'BINGX_API_KEY',
        'api_secret_env': 'BINGX_API_SECRET'
    },
    'blofin': {
        'quote_asset': 'USDT',
        'api_key_env': 'BLOFIN_API_KEY',
        'api_secret_env': 'BLOFIN_API_SECRET'
    },
    'okx': {
        'quote_asset': 'USDT',
        'api_key_env': 'OKX_API_KEY',
        'api_secret_env': 'OKX_API_SECRET',
        'requires_passphrase': True,
        'passphrase_env': 'OKX_PASSPHRASE'
    }
}
class MultiBrokerManager:
    def __init__(self):
        self.brokers = {}
        self._init_all_brokers()
    def _init_all_brokers(self):
        for broker_name, config in config.brokers_config.BROKER_CONFIG.items():
            self._init_broker(broker_name, config)
    def _init_broker(self, broker_name, config):
        try:
            exchange = getattr(ccxt, broker_name)({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            # Configuration des clés
            api_key = os.getenv(config['api_key_env'])
            api_secret = os.getenv(config['api_secret_env'])
            if api_key and api_secret:
                exchange.apiKey = api_key
                exchange.secret = api_secret
                if config.get('requires_passphrase'):
                    exchange.password = os.getenv(config['passphrase_env'])
            self.brokers[broker_name] = exchange
        except Exception as e:
            logger.warning(f"Erreur initialisation {broker_name}: {str(e)}")
# ============ FIN AJOUT MULTI-BROKER ============
# Ajout à la classe USDCArbitrage existante
def set_broker(self, broker_name):
    """Change de broker dynamiquement"""
    if broker_name in config.brokers_config.BROKER_CONFIG:
        self.exchange = MultiBrokerManager().brokers.get(broker_name)
        self.quote_asset = config.brokers_config.BROKER_CONFIG[broker_name]['quote_asset']
    else:
        raise ValueError(f"Broker {broker_name} non configuré")
def get_available_brokers(self):
    """Liste tous les brokers configurés"""
    return list(config.brokers_config.BROKER_CONFIG.keys())
# ============ AJOUT MULTI-BROKER ============
config.brokers_config.BROKER_CONFIG = {
    'binance': {
        'name': 'binance',
        'quote': 'USDC',
        'key_env': 'BINANCE_API_KEY',
        'secret_env': 'BINANCE_API_SECRET',
        'params': {'options': {'adjustForTimeDifference': True}}
    },
    'gateio': {
        'name': 'gateio',
        'quote': 'USDT',
        'key_env': 'GATEIO_API_KEY', 
        'secret_env': 'GATEIO_API_SECRET'
    },
    'bingx': {
        'name': 'bingx',
        'quote': 'USDT',
        'key_env': 'BINGX_API_KEY',
        'secret_env': 'BINGX_API_SECRET'
    },
    'blofin': {
        'name': 'blofin',
        'quote': 'USDT',
        'key_env': 'BLOFIN_API_KEY',
        'secret_env': 'BLOFIN_API_SECRET'
    },
    'okx': {
        'name': 'okx',
        'quote': 'USDT',
        'key_env': 'OKX_API_KEY',
        'secret_env': 'OKX_API_SECRET',
        'passphrase_env': 'OKX_PASSPHRASE'
    }
}
class MultiBroker:
    def __init__(self):
        self.brokers = {}
        self._init_all()
    def _init_all(self):
        for name, config in config.brokers_config.BROKER_CONFIG.items():
            self._init_broker(name, config)
    def _init_broker(self, name, config):
        try:
            params = {
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }
            # Ajout des paramètres spécifiques
            if 'params' in config:
                params.update(config['params'])
            # Configuration des clés
            api_key = os.getenv(config['key_env'])
            api_secret = os.getenv(config['secret_env'])
            if api_key and api_secret:
                params['apiKey'] = api_key
                params['secret'] = api_secret
                if 'passphrase_env' in config:
                    params['password'] = os.getenv(config['passphrase_env'])
            self.brokers[name] = getattr(ccxt, name)(params)
        except Exception as e:
            logger.warning(f"Init {name} failed: {str(e)}")
# Ajout méthode à USDCArbitrage
def set_broker(self, name):
    if name in self.brokers:
        self.exchange = self.brokers[name]
        self.quote_asset = config.brokers_config.BROKER_CONFIG[name]['quote']
    else:
        raise ValueError(f"Broker {name} not configured")
# ============ CONFIGURATION MULTI-BROKER ============
import os
from typing import Dict
BROKER_SETTINGS = {
    'binance': {
        'base_params': {'options': {'adjustForTimeDifference': True}},
        'quote': 'USDC',
        'key_var': 'BINANCE_API_KEY',
        'secret_var': 'BINANCE_API_SECRET'
    },
    'gateio': {
        'quote': 'USDT',
        'key_var': 'GATEIO_API_KEY',
        'secret_var': 'GATEIO_API_SECRET'
    },
    'bingx': {
        'quote': 'USDT',
        'key_var': 'BINGX_API_KEY',
        'secret_var': 'BINGX_API_SECRET'
    },
    'blofin': {
        'quote': 'USDT',
        'key_var': 'BLOFIN_API_KEY',
        'secret_var': 'BLOFIN_API_SECRET'
    },
    'okx': {
        'quote': 'USDT',
        'key_var': 'OKX_API_KEY',
        'secret_var': 'OKX_API_SECRET',
        'passphrase_var': 'OKX_PASSPHRASE'
    }
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
            params = {
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }
            # Paramètres spécifiques
            if 'base_params' in config:
                params.update(config['base_params'])
            # Authentification
            api_key = os.getenv(config['key_var'])
            api_secret = os.getenv(config['secret_var'])
            if api_key and api_secret:
                params.update({
                    'apiKey': api_key,
                    'secret': api_secret
                })
                if 'passphrase_var' in config:
                    params['password'] = os.getenv(config['passphrase_var'])
            self.brokers[name] = getattr(ccxt, name)(params)
        except Exception as e:
            logger.warning(f"Initialisation {name} échouée: {str(e)}")
# Extension de USDCArbitrage
def switch_broker(self, broker_name: str):
    """Change de broker dynamiquement"""
    if broker_name in BrokerManager().brokers:
        self.exchange = BrokerManager().brokers[broker_name]
        self.quote_asset = BROKER_SETTINGS[broker_name]['quote']
    else:
        raise ValueError(f"Broker {broker_name} non configuré")
def available_brokers(self) -> list:
    """Liste les brokers disponibles"""
    return list(BROKER_SETTINGS.keys())
