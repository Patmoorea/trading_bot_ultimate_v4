def kelly_criterion(win_rate, payoff_ratio):
    """
    Calcule la fraction de Kelly (gestion du risque optimale).
    win_rate : taux de réussite (ex : 0.55)
    payoff_ratio : ratio gain/perte moyen (ex : 1.5)
    """
    if payoff_ratio == 0:
        return 0
    edge = win_rate - (1 - win_rate) / payoff_ratio
    return max(0, edge)
