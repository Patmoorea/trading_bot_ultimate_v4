"""
Chargement centralisé des exchanges CCXT à partir des variables .env
Adapté à l'environnement existant sans modifier les noms de variables.
"""
import os
from dotenv import load_dotenv
import ccxt
load_dotenv()
def load_binance():
    return ccxt.binance({
        'apiKey': os.getenv("BINANCE_API_KEY"),
        'secret': os.getenv("BINANCE_API_SECRET"),
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {
            'defaultType': 'spot'
        }
    })
def load_gateio():
    return ccxt.gateio({
        'apiKey': os.getenv("GATEIO_API_KEY"),
        'secret': os.getenv("GATEIO_API_SECRET"),
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {
            'defaultType': 'spot'
        }
    })
def load_bingx():
    return ccxt.bingx({
        'apiKey': os.getenv("BINGX_API_KEY"),
        'secret': os.getenv("BINGX_API_SECRET"),
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {
            'defaultType': 'spot'
        }
    })
def load_blofin():
    return ccxt.blofin({
        'apiKey': os.getenv("BLOFIN_API_KEY"),
        'secret': os.getenv("BLOFIN_API_SECRET"),
        'enableRateLimit': True,
        'timeout': 15000
    })
def load_okx():
    return ccxt.okx({
        'apiKey': os.getenv("OKX_API_KEY"),
        'secret': os.getenv("OKX_API_SECRET"),
        'password': os.getenv("OKX_PASSPHRASE"),
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {
            'defaultType': 'spot'
        }
    })
