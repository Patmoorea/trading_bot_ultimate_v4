from decimal import Decimal
from datetime import datetime
import logging
from typing import List, Dict, Any

# Configuration du logging
logger = logging.getLogger(__name__)


class ArbitrageScanner:
    def __init__(
        self,
        exchanges=None,
        min_profit=0.002,
        max_price_deviation=0.05,
        pairs=None,
        max_trade_size=1000,
        timeout=5,
        volume_filter=1000,
        price_check=True,
        max_slippage=0.0005,
    ):
        self.exchanges = exchanges or []
        self.pairs = pairs or ["BTC/USDT", "ETH/USDT"]
        self.min_profit = min_profit
        self.max_price_deviation = max_price_deviation
        self.max_trade_size = max_trade_size
        self.timeout = timeout
        self.volume_filter = volume_filter
        self.price_check = price_check
        self.max_slippage = max_slippage
        self.quote_currency_map = {
            "binance": "USDT",
            "bybit": "USDT",
            "okx": "USDT",
            "gateio": "USDT",
        }

    async def scan_opportunities(self) -> list[dict]:
        """Scanne les opportunités d'arbitrage"""
        opportunities = []
        try:
            for pair in self.pairs:
                valid_exchanges = [
                    e for e in self.exchanges if self._validate_symbol(pair, e.name)
                ]
                for buy_ex in valid_exchanges:
                    for sell_ex in valid_exchanges:
                        if buy_ex.name == sell_ex.name:
                            continue
                        # Récupération des orderbooks
                        buy_ob = await buy_ex.fetch_order_book(pair)
                        sell_ob = await sell_ex.fetch_order_book(pair)
                        if not buy_ob or not sell_ob:
                            continue
                        # Vérification du volume
                        buy_volume = buy_ob["bids"][0][1]
                        sell_volume = sell_ob["asks"][0][1]
                        if min(buy_volume, sell_volume) < self.volume_filter:
                            continue
                        # Prix d'achat et de vente
                        buy_price = buy_ob["asks"][0][0]  # Meilleur ask pour acheter
                        sell_price = sell_ob["bids"][0][0]  # Meilleur bid pour vendre
                        # Calcul du profit potentiel
                        profit = (sell_price - buy_price) / buy_price
                        # Vérification du profit minimum
                        if profit > self.min_profit:
                            # Calcul de la taille maximale possible
                            max_size = min(
                                self.max_trade_size,
                                buy_volume * buy_price,
                                sell_volume * sell_price,
                            )
                            opportunities.append(
                                {
                                    "pair": pair,
                                    "buy_exchange": buy_ex.name,
                                    "sell_exchange": sell_ex.name,
                                    "buy_price": buy_price,
                                    "sell_price": sell_price,
                                    "profit_pct": profit * 100,
                                    "max_trade_size": max_size,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
        except Exception as e:
            logger.error(f"Erreur lors du scan des opportunités: {e}")
        return opportunities

    def _validate_symbol(self, symbol: str, exchange: str) -> bool:
        """Valide si un symbole est compatible avec l'exchange"""
        quote = self._get_exchange_quote_currency(exchange)
        return quote in symbol and "/" in symbol

    def _get_exchange_quote_currency(self, exchange: str) -> str:
        """Retourne la devise de cotation de l'exchange"""
        return self.quote_currency_map.get(exchange.lower(), "USDT")

    async def check_price_feasibility(self, opportunity: dict) -> bool:
        """Vérifie si les prix sont réalisables avec le slippage maximum"""
        try:
            if not self.price_check:
                return True
            buy_ex = next(
                e for e in self.exchanges if e.name == opportunity["buy_exchange"]
            )
            sell_ex = next(
                e for e in self.exchanges if e.name == opportunity["sell_exchange"]
            )
            # Vérification des prix actuels
            buy_ticker = await buy_ex.fetch_ticker(opportunity["pair"])
            sell_ticker = await sell_ex.fetch_ticker(opportunity["pair"])
            # Calcul du slippage
            buy_slippage = (
                abs(buy_ticker["ask"] - opportunity["buy_price"])
                / opportunity["buy_price"]
            )
            sell_slippage = (
                abs(sell_ticker["bid"] - opportunity["sell_price"])
                / opportunity["sell_price"]
            )
            return max(buy_slippage, sell_slippage) <= self.max_slippage
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des prix: {e}")
            return False

    async def find_opportunities(self) -> list[dict]:
        """Trouve et valide les opportunités d'arbitrage"""
        try:
            # Scan initial des opportunités
            opportunities = await self.scan_opportunities()
            # Filtrage des opportunités valides
            valid_opportunities = []
            for opp in opportunities:
                if await self.check_price_feasibility(opp):
                    valid_opportunities.append(opp)
            return valid_opportunities
        except Exception as e:
            logger.error(f"Erreur lors de la recherche d'opportunités: {e}")
            return []
