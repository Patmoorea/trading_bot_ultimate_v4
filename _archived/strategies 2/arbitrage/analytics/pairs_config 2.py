#!/usr/bin/env python3
# Configuration précise par exchange
PAIRS_CONFIG = {
    'binance': {
        'format': '{base}/USDC',  # Binance en USDC
        'assets': ['BTC', 'ETH', 'SOL']
    },
    'okx': {
        'format': '{base}/USDT',  
        'assets': ['BTC', 'ETH', 'SOL']
    },
    'blofin': {
        'format': '{base}USDT',  # Format spécifique Blofin (sans slash)
        'assets': ['BTC', 'ETH', 'SOL']
    },
    'gateio': {
        'format': '{base}_USDT',  # Format Gate.io
        'assets': ['BTC', 'ETH']
    }
}
def get_symbol(exchange: str, base_asset: str) -> str:
    """Génère le symbol exact pour chaque exchange"""
    if exchange not in PAIRS_CONFIG:
        raise ValueError(f"Exchange {exchange} non configuré")
    if base_asset not in PAIRS_CONFIG[exchange]['assets']:
        raise ValueError(f"Asset {base_asset} non supporté sur {exchange}")
    return PAIRS_CONFIG[exchange]['format'].format(base=base_asset)
