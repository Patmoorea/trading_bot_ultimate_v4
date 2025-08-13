"""
Wrapper pour maintenir la compatibilité avec l'existant
"""
from .core import UnifiedArbitrage
class USDCArbitrage(UnifiedArbitrage):
    """Wrapper pour les anciens scripts"""
    def scan_all_pairs(self):
        return self.find_opportunities()
    def get_opportunities(self):
        return [opp.__dict__ for opp in self.find_opportunities()]
    def switch_broker(self, exchange_name):
        if exchange_name in self.exchanges:
            self.exchanges = {exchange_name: self.exchanges[exchange_name]}
class ArbitrageEngine(UnifiedArbitrage):
    """Compatibilité avec modules/arbitrage_engine.py"""
    def calculate(self):
        opps = self.find_opportunities()
        return {"status": "success", "count": len(opps)}
