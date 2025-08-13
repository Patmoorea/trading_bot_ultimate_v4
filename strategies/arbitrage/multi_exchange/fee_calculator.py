"""
Calculateur de frais pour les opérations d'arbitrage
"""
from decimal import Decimal
from typing import Dict
from src.exchanges.base_exchange import BaseExchange
class FeeCalculator:
    """Calculateur de frais pour les opérations d'arbitrage"""
    def __init__(self):
        self.fee_cache = {}  # Cache des frais par exchange
    async def calculate_total_fees(self, 
                                 buy_exchange: BaseExchange,
                                 sell_exchange: BaseExchange,
                                 symbol: str,
                                 amount: Decimal,
                                 buy_price: Decimal,
                                 sell_price: Decimal) -> Dict:
        """
        Calcule les frais totaux pour une opération d'arbitrage
        Args:
            buy_exchange: Exchange pour l'achat
            sell_exchange: Exchange pour la vente
            symbol: Paire de trading (ex: BTC/USDT)
            amount: Quantité à trader
            buy_price: Prix d'achat
            sell_price: Prix de vente
        Returns:
            Dict contenant:
            - buy_fees: Frais d'achat
            - sell_fees: Frais de vente
            - transfer_fees: Frais de transfert estimés
            - total_fees: Total des frais
            - net_profit: Profit après frais
            - is_profitable: True si l'opération est profitable après frais
        """
        # Calcul des frais d'achat
        buy_fees = await self._get_trading_fees(buy_exchange, symbol, amount, buy_price)
        # Calcul des frais de vente
        sell_fees = await self._get_trading_fees(sell_exchange, symbol, amount, sell_price)
        # Estimation des frais de transfert
        transfer_fees = await self._estimate_transfer_fees(buy_exchange, sell_exchange, symbol, amount)
        # Calcul des frais totaux
        total_fees = buy_fees + sell_fees + transfer_fees
        # Calcul du profit brut
        gross_profit = (sell_price - buy_price) * amount
        # Calcul du profit net
        net_profit = gross_profit - total_fees
        return {
            'buy_fees': buy_fees,
            'sell_fees': sell_fees,
            'transfer_fees': transfer_fees,
            'total_fees': total_fees,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'is_profitable': net_profit > Decimal('0')
        }
    async def _get_trading_fees(self, 
                              exchange: BaseExchange, 
                              symbol: str,
                              amount: Decimal,
                              price: Decimal) -> Decimal:
        """Récupère les frais de trading pour un exchange"""
        if exchange not in self.fee_cache:
            self.fee_cache[exchange] = await self._fetch_trading_fees(exchange)
        fee_rate = self.fee_cache[exchange].get(symbol, Decimal('0.001'))  # 0.1% par défaut
        return amount * price * fee_rate
    async def _fetch_trading_fees(self, exchange: BaseExchange) -> Dict:
        """Récupère les frais de trading depuis l'API de l'exchange"""
        # À implémenter pour chaque exchange
        return {}
    async def _estimate_transfer_fees(self,
                                    from_exchange: BaseExchange,
                                    to_exchange: BaseExchange,
                                    symbol: str,
                                    amount: Decimal) -> Decimal:
        """Estime les frais de transfert entre exchanges"""
        # À implémenter : estimation basée sur les frais de réseau
        return Decimal('0')
