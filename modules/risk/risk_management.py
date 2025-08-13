# ===== NOUVEAU CODE KELLY =====
def calculate_kelly(win_rate, win_loss_ratio):
    """Ajout à la classe existante"""
    return (win_rate * (win_loss_ratio + 1) - 1) / win_loss_ratio
def calculate_kelly(win_rate, win_loss_ratio):
    """Position sizing selon Kelly"""
    return (win_rate * (win_loss_ratio + 1) - 1) / win_loss_ratio
def dynamic_stop_loss(volatility):
    """Nouveau calcul de stop-loss adaptatif"""
    return 0.02 + (volatility * 0.1)  # 2% de base + ajustement volatilité
