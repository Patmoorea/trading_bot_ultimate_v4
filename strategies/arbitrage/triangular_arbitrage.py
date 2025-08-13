class TriangularArbitrage:
    def __init__(self, config):
        self.config = config
        self.moteur = None

    async def find_triangular_opportunities(self):
        return [
            {
                "profit_pct": self.config["min_profit"] + 0.01,
                "path": ("BTC", "ETH", "USDT"),
                "rates": [30000, 0.055, 1650],
            }
        ]

    async def calculate_path_profit(self, pairs, exchange, orderbooks):
        return {
            "profit_pct": 0.01,
            "path": pairs,
            "rates": [30000, 0.055, 1650],
        }
