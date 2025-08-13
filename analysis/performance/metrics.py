class SharpeRatio:
    """Calculate Sharpe Ratio for strategy performance"""
    def __init__(self, risk_free_rate=0.0):
        self.risk_free_rate = risk_free_rate
    
    def calculate(self, returns):
        """Calculate Sharpe ratio from returns series"""
        import numpy as np
        excess_returns = returns - self.risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)


class MaxDrawdown:
    """Calculate Maximum Drawdown for strategy performance"""
    def calculate(self, equity_curve):
        """Calculate max drawdown from equity curve"""
        import numpy as np
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return np.max(drawdown)