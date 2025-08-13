import os
from decimal import Decimal
from dotenv import load_dotenv
load_dotenv()
PAIRS = {
    'BTC': {
        'binance': 'BTC/USDT',
        'gateio': 'BTC_USDT',
        'bingx': 'BTC-USDT',
        'okx': 'BTC/USDT:USDT',
        'blofin': 'BTC-USDT'
    },
    'ETH': {
        'binance': 'ETH/USDT',
        'gateio': 'ETH_USDT',
        'bingx': 'ETH-USDT',
        'okx': 'ETH/USDT:USDT',
        'blofin': 'ETH-USDT'
    },
    'SOL': {
        'binance': 'SOL/USDT',
        'gateio': 'SOL_USDT',
        'bingx': 'SOL-USDT',
        'okx': 'SOL/USDT:USDT',
        'blofin': 'SOL-USDT'
    }
}
SETTINGS = {
    'profit_threshold': Decimal('0.005'),
    'max_order_value': Decimal('100'),
    'min_liquidity': Decimal('1000'),
    'fee_adjustment': Decimal('1.3'),
    'price_expiry': 5,
    'max_slippage': Decimal('0.008')
}
FEES = {
    'binance': {'maker': Decimal('0.001'), 'taker': Decimal('0.001')},
    'gateio': {'maker': Decimal('0.002'), 'taker': Decimal('0.002')},
    'bingx': {'maker': Decimal('0.0002'), 'taker': Decimal('0.0005')},
    'okx': {'maker': Decimal('0.0002'), 'taker': Decimal('0.0005')},
    'blofin': {'maker': Decimal('0.0001'), 'taker': Decimal('0.0004')}
}
