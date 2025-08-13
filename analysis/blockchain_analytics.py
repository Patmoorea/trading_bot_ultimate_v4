class OnChainAnalyzer:
    def __init__(self):
        self.indicators = {
            'whale_movement': self.detect_whales,
            'exchange_flow': self.analyze_flow,
            'smart_money': self.track_smart_money
        }
    async def analyze_all_chains(self):
        results = {}
        for chain in self.supported_chains:
            results[chain] = await self.analyze_chain(chain)
        return results
