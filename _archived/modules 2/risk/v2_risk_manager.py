class KellyRiskManager:
    """Nouvelle implémentation côte-à-côte avec l'ancienne"""
    @staticmethod
    def calculate(wins, losses):
        win_rate = wins / (wins + losses)
        win_loss_ratio = sum(wins)/sum(losses)
        return (win_rate * (win_loss_ratio + 1) - 1) / win_loss_ratio
