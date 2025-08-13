import ccxt
import os
import time
from dotenv import load_dotenv
load_dotenv()
class MultiExchangeArbitrage:
    def __init__(self):
        self.exchanges = {
            'binance': ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_API_SECRET'),
                'enableRateLimit': True
            }),
            'gateio': ccxt.gateio({
                'apiKey': os.getenv('GATEIO_API_KEY'),
                'secret': os.getenv('GATEIO_API_SECRET')
            }),
            'bingx': ccxt.bingx({
                'apiKey': os.getenv('BINGX_API_KEY'),
                'secret': os.getenv('BINGX_API_SECRET')
            }),
            'okx': ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_API_SECRET'),
                'password': os.getenv('OKX_PASSPHRASE')
            })
        }
        self.threshold = float(os.getenv('ARBITRAGE_THRESHOLD', 0.3))
    def check_arbitrage(self, base='BTC', quote1='USDC', quote2='USDT'):
        opportunities = []
        for name, exchange in self.exchanges.items():
            try:
                # Récupération des order books
                pair1 = f"{base}/{quote1}"
                pair2 = f"{base}/{quote2}"
                book1 = exchange.fetch_order_book(pair1)
                book2 = exchange.fetch_order_book(pair2)
                # Calcul du spread
                bid = book1['bids'][0][0]
                ask = book2['asks'][0][0]
                spread = (bid / ask - 1) * 100
                if spread > self.threshold:
                    opportunities.append({
                        'exchange': name,
                        'spread': spread,
                        'pair1': pair1,
                        'pair2': pair2,
                        'bid': bid,
                        'ask': ask
                    })
            except Exception as e:
                print(f"Erreur sur {name}: {str(e)}")
        return opportunities
    def monitor(self, interval=30):
        print("\n=== Surveillance Multi-Plateforme ===")
        print(f"Seuil: {self.threshold}% | Intervalle: {interval}s")
        print("Plateformes actives: Binance, Gate.io, BingX, OKX")
        print("Appuyez sur Ctrl+C pour quitter\n")
        while True:
            try:
                opportunities = self.check_arbitrage()
                if opportunities:
                    for opp in opportunities:
                        print(f"[{timestamp}] {opp['exchange'].upper()}:")
                        print(f"  {opp['pair1']} bid: {opp['bid']}")
                        print(f"  {opp['pair2']} ask: {opp['ask']}")
                        print(f"  SPREAD: {opp['spread']:.4f}%")
                        print("-"*40)
                else:
                    print(f"[{timestamp}] Aucune opportunité > {self.threshold}%", end='\r')
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nArrêt du monitoring")
                break
def send_arbitrage_alert(opportunity):
    """Nouvelle fonction pour les alertes Telegram"""
    from utils.telegram_notifications import notifier
    message = (
        f"🚨 <b>Opportunité d'Arbitrage</b>\n"
        f"• Exchange: {opportunity['exchange'].upper()}\n"
        f"• Spread: {opportunity['spread']:.2f}%\n"
        f"• {opportunity['pair1']} bid: {opportunity['bid']}\n"
        f"• {opportunity['pair2']} ask: {opportunity['ask']}"
    )
    return notifier.send(message)
# Modification de la méthode monitor (ajout à la fin existante)
def monitor(self, interval=30):
    """Version étendue avec notifications"""
    while True:
        opportunities = self.check_arbitrage()
        if opportunities:
            for opp in opportunities:
                if opp['spread'] > self.threshold:
                    send_arbitrage_alert(opp)
        time.sleep(interval)
