import os
from decimal import Decimal
from dotenv import load_dotenv
load_dotenv()
# Configuration des paires SPOT (corrigée)
PAIRS = {
    'BTC': {
        'binance': 'BTC/USDC',
        'gateio': 'BTC_USDT',  # Format correct pour Gate.io
        'bingx': 'BTC-USDT',   # Format correct pour BingX
        'okx': 'BTC/USDT',     # Format correct pour OKX
        'blofin': 'BTC-USDT'   # Format correct pour Blofin
    },
    'ETH': {
        'binance': 'ETH/USDC',
        'gateio': 'ETH_USDT',
        'bingx': 'ETH-USDT',
        'okx': 'ETH/USDT',
        'blofin': 'ETH-USDT'
    }
}
SETTINGS = {
    'profit_threshold': Decimal('0.01'),  # Seuil à 1% pour plus de résultats
    'max_order_value': Decimal('500'),    # Montant réduit pour tests
    'min_liquidity': Decimal('1000'),     # Seuil de liquidité réduit
    'fee_adjustment': Decimal('1.5'),     # Marge de sécurité augmentée
    'price_expiry': 10,                   # Intervalle augmenté
    'max_slippage': Decimal('0.01')       # Slippage augmenté
}
FEES = {
    'binance': {'maker': Decimal('0.001'), 'taker': Decimal('0.001')},
    'gateio': {'maker': Decimal('0.002'), 'taker': Decimal('0.002')},
    'bingx': {'maker': Decimal('0.0002'), 'taker': Decimal('0.0005')},
    'okx': {'maker': Decimal('0.0002'), 'taker': Decimal('0.0005')},
    'blofin': {'maker': Decimal('0.0001'), 'taker': Decimal('0.0004')}
}
