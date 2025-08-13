# ===== ARBITRAGE USDC =====
def check_usdc_arbitrage():
    """Ajout à l'executor existant"""
    btc_usdc = exchange.fetch_order_book('BTC/USDC')
    btc_usdt = exchange.fetch_order_book('BTC/USDT')
    return (btc_usdc['bids'][0][0] / btc_usdt['asks'][0][0] - 1) > 0.005
def check_usdc_arbitrage():
    """Analyse les opportunités d'arbitrage USDC/USDT"""
    import ccxt
    from decimal import Decimal
    from time import sleep
    # Configuration depuis .env
    exchange = ccxt.binance({
        'apiKey': '${BINANCE_API_KEY}',
        'secret': '${BINANCE_API_SECRET}',
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    while True:
        try:
            # Récupération des order books
            usdc_book = exchange.fetch_order_book('BTC/USDC')
            usdt_book = exchange.fetch_order_book('BTC/USDT')
            # Calcul du spread
            best_bid_usdc = Decimal(usdc_book['bids'][0][0])
            best_ask_usdt = Decimal(usdt_book['asks'][0][0])
            spread = (best_bid_usdc / best_ask_usdt - 1) * 100
            if spread > float('${ARBITRAGE_THRESHOLD:-0.5}'):  # Seuil par défaut 0.5%
                print(f"Opportunité d'arbitrage détectée: {spread:.2f}%")
                # Ici vous pourriez ajouter la logique d'exécution
            sleep(10)  # Vérification toutes les 10 secondes
        except Exception as e:
            print(f"Erreur d'arbitrage: {str(e)}")
            sleep(30)
