import numpy as np

def calculate_var(returns, alpha=0.05):
    """
    Calcule la Value at Risk (VaR) à un certain niveau alpha (ex: 0.05 pour 95%).
    returns: liste ou np.array des rendements.
    """
    if returns is None or len(returns) == 0:
        return 0
    return np.percentile(returns, 100 * alpha)
