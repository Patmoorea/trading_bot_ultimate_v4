import ccxt
import os
import time
import logging
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class MultiExchangeArbitrage:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchanges = {
            "binance": ccxt.binance(
                {
                    "apiKey": os.getenv("BINANCE_API_KEY"),
                    "secret": os.getenv("BINANCE_API_SECRET"),
                    "enableRateLimit": True,
                }
            ),
            "gateio": ccxt.gateio(
                {
                    "apiKey": os.getenv("GATEIO_API_KEY"),
                    "secret": os.getenv("GATEIO_API_SECRET"),
                }
            ),
            "bingx": ccxt.bingx(
                {
                    "apiKey": os.getenv("BINGX_API_KEY"),
                    "secret": os.getenv("BINGX_API_SECRET"),
                }
            ),
            "okx": ccxt.okx(
                {
                    "apiKey": os.getenv("OKX_API_KEY"),
                    "secret": os.getenv("OKX_API_SECRET"),
                    "password": os.getenv("OKX_PASSPHRASE"),
                }
            ),
            "blofin": ccxt.blofin(
                {
                    "apiKey": os.getenv("BLOFIN_API_KEY"),
                    "secret": os.getenv("BLOFIN_API_SECRET"),
                    "password": os.getenv("BLOFIN_PASSPHRASE"),
                }
            ),
        }
        self.threshold = float(os.getenv("ARBITRAGE_THRESHOLD", 0.3))
        self.version = "2.0.0"

    def check_arbitrage(self, base="BTC", quote1="USDC", quote2="USDT") -> List[Dict]:
        opportunities = []
        for name, exchange in self.exchanges.items():
            try:
                pair1 = f"{base}/{quote1}"
                pair2 = f"{base}/{quote2}"
                book1 = exchange.fetch_order_book(pair1)
                book2 = exchange.fetch_order_book(pair2)
                bid = book1["bids"][0][0]
                ask = book2["asks"][0][0]
                spread = (bid / ask - 1) * 100
                if spread > self.threshold:
                    opportunities.append(
                        {
                            "exchange": name,
                            "spread": spread,
                            "pair1": pair1,
                            "pair2": pair2,
                            "bid": bid,
                            "ask": ask,
                            "timestamp": datetime.utcnow(),
                            "volume": min(book1["bids"][0][1], book2["asks"][0][1]),
                        }
                    )
            except Exception as e:
                self.logger.error(f"Erreur sur {name}: {str(e)}")
        return sorted(opportunities, key=lambda x: x["spread"], reverse=True)

    def monitor(self, interval: int = 30):
        print("\n=== Surveillance Multi-Plateforme ===")
        print(f"Seuil: {self.threshold}% | Intervalle: {interval}s")
        print("Plateformes actives: Binance, Gate.io, BingX, OKX, Blofin")
        print("Appuyez sur Ctrl+C pour quitter\n")
        while True:
            try:
                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                opportunities = self.check_arbitrage()
                if opportunities:
                    for opp in opportunities:
                        print(f"[{timestamp}] {opp['exchange'].upper()}:")
                        print(f"  {opp['pair1']} bid: {opp['bid']}")
                        print(f"  {opp['pair2']} ask: {opp['ask']}")
                        print(f"  SPREAD: {opp['spread']:.4f}%")
                        print(f"  VOLUME: {opp['volume']:.4f}")
                        print("-" * 40)
                else:
                    print(
                        f"[{timestamp}] Aucune opportunité > {self.threshold}%",
                        end="\r",
                    )
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nArrêt du monitoring")
                break

    def get_best_spread(self) -> Dict:
        opportunities = self.check_arbitrage()
        return opportunities[0] if opportunities else None

    def execute_arbitrage(self, opportunity: Dict) -> Dict:
        try:
            exchange = self.exchanges[opportunity["exchange"]]
            balance = exchange.fetch_balance()
            amount = min(
                float(balance["free"][opportunity["pair1"].split("/")[0]]),
                opportunity["volume"],
            )
            if amount <= 0:
                return {"success": False, "error": "Balance insuffisante"}
            # Exécution des ordres
            buy_order = exchange.create_market_buy_order(opportunity["pair1"], amount)
            sell_order = exchange.create_market_sell_order(opportunity["pair2"], amount)
            return {
                "success": True,
                "buy_order": buy_order,
                "sell_order": sell_order,
                "profit": opportunity["spread"],
                "amount": amount,
                "timestamp": datetime.utcnow(),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "timestamp": datetime.utcnow()}
