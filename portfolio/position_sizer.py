import numpy as np

def compute_drawdown(equity_curve):
    """Calcule le drawdown maximum réalisé sur une equity curve (tableau de float)"""
    roll_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()  # Ex: -0.25 pour -25%

def dynamic_position_size(
    base_risk,
    volatility,
    confidence,
    drawdown,
    perf_recent=0,
    max_risk=0.05,
    min_risk=0.01,
):
    # Facteur volatilité (plus c'est volatile, plus on réduit le risque)
    vol_factor = min(1.0, 0.02 / max(volatility, 1e-6))
    # Facteur confiance (plus la confiance est forte, plus on engage)
    conf_factor = max(0.5, abs(confidence))
    # Facteur drawdown (plus le drawdown est fort, plus on réduit)
    dd_factor = max(0.5, 1 + drawdown)  # drawdown négatif (ex: -0.2)
    # Facteur performance récente (bonus/malus)
    perf_factor = 1.0 + np.tanh(perf_recent)
    # Taille finale
    risk = base_risk * vol_factor * conf_factor * dd_factor * perf_factor
    # Clamp entre min et max
    return float(max(min_risk, min(max_risk, risk)))
