class AdvancedIndicators:
    def __init__(self):
        self.volatility_indicators = {
            'parkinson': self._parkinson_volatility,
            'yang_zhang': self._yang_zhang_volatility
        }
    def calculate_all(self, data):
        results = {}
        for name, func in self.volatility_indicators.items():
            results[name] = func(data)
        return results
