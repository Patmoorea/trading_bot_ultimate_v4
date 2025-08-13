from typing import List
class MultiExchangeCoordinator:
    """Coordonne plusieurs instances du module existant"""
    def __init__(self, pairs: List[str], exchanges: List[str]):
        self.arbitrage_instances = {
            name: USDCArbitrage(pairs=pairs, broker_name=name)
            for name in exchanges
        }
    def get_best_opportunity(self):
        """Trouve la meilleure opportunité parmi tous les exchanges"""
        opportunities = []
        for name, arb in self.arbitrage_instances.items():
            opportunities.extend((name, opp) for opp in arb.scan_all_pairs())
        return sorted(opportunities, key=lambda x: x[1][1], reverse=True)[0]
