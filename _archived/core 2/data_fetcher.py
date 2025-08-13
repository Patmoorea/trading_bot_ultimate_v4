def get_market_data():
    """Récupère les données de marché actuelles"""
    from ccxt import binance
    exchange = binance()
    return exchange.fetch_ohlcv(
        'BTC/USDT', '1m')[-100:]  # 100 dernières minutes
