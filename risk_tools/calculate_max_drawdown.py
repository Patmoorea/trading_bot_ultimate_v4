import numpy as np

def calculate_max_drawdown(equity_curve):
    """
    Calcule le maximum drawdown d'une courbe d'équité.
    equity_curve : liste ou np.array des valeurs de portefeuille (balance).
    """
    if equity_curve is None or len(equity_curve) == 0:
        return 0
    equity_curve = np.array(equity_curve)
    roll_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - roll_max) / roll_max
    return np.min(drawdowns)
