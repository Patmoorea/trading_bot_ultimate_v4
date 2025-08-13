import numpy as np

def kelly_criterion(win_rate, payoff_ratio):
    if payoff_ratio > 0:
        return win_rate - (1 - win_rate) / payoff_ratio
    return 0.0

def calculate_var(returns, alpha=0.05):
    # returns doit être un array de rendements journaliers
    return np.percentile(returns, 100 * alpha)

def calculate_max_drawdown(equity_curve):
    cum_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / cum_max - 1
    return np.min(drawdown)