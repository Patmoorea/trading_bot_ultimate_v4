from ccxt import Exchange
from typing import Dict
from .core import ArbitrageOpportunity
class ArbitrageExecutor:
    def __init__(self, exchanges: Dict[str, Exchange]):
        self.exchanges = exchanges
    def execute_trade(self, opportunity: ArbitrageOpportunity):
        """
        Exécute une opération d'arbitrage
        """
        try:
            # Achat sur exchange B
            exch_b = self.exchanges[opportunity.exchange_b]
            buy_order = exch_b.create_order(
                symbol=opportunity.pair,
                type='limit',
                side='buy',
                amount=opportunity.volume,
                price=opportunity.ask_b
            )
            # Vente sur exchange A
            exch_a = self.exchanges[opportunity.exchange_a]
            sell_order = exch_a.create_order(
                symbol=opportunity.pair,
                type='limit',
                side='sell',
                amount=opportunity.volume,
                price=opportunity.bid_a
            )
            return {
                'buy_order': buy_order,
                'sell_order': sell_order,
                'profit': (opportunity.bid_a - opportunity.ask_b) * opportunity.volume
            }
        except Exception as e:
            self.logger.error(f"Erreur d'exécution: {str(e)}")
            self.cancel_all_orders(opportunity)
            raise
    def cancel_all_orders(self, opportunity):
        """Annule tous les ordres en cas d'erreur"""
        for exch_name in [opportunity.exchange_a, opportunity.exchange_b]:
            self.exchanges[exch_name].cancel_all_orders(opportunity.pair)
