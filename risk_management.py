# ========= NOUVEAU CODE =========
class AdvancedCircuitBreaker:
    """Gestion des scénarios extrêmes"""
    def __init__(self):
        self.triggers = {
            'market_crash': self.detect_market_crash,
            'liquidity_shock': self.check_liquidity
        }
    def detect_market_crash(self, data):
        return data['price_drop'] > 0.15  # -15% en 1h
    def check_liquidity(self, orderbook):
        return orderbook['bid_volume'] < 0.5 * orderbook['ask_volume']
