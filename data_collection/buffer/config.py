BUFFER_CONFIG = {
    'max_size': 1000,
    'compression': 'lz4',
    'latency_target': 15  # ms,
    'enabled_pairs': [
        'BTC/USDC',
        'ETH/USDC',
        'SOL/USDC',
        'AVAX/USDC',
        'MATIC/USDC'
    ],
    'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d']
}
STORAGE_CONFIG = {
    'format': 'arrow',
    'compression_ratio': 0.4,
    'auto_cleanup': True,
    'max_age_days': 30
}
