import numpy as np


class ExitManager:
    def __init__(self, tp_levels=None, trailing_pct=0.03):
        """
        tp_levels: Liste de tuples (target_pct, fraction) ex: [(0.03, 0.3), (0.07, 0.3)]
        trailing_pct: Pourcentage du trailing stop (ex: 0.03 = 3%)
        """
        self.tp_levels = [
            {"level": 1.02, "size": 0.3},  # +2% -> vend 30%
            {"level": 1.035, "size": 0.3},  # +3.5% -> vend 30%
            {"level": 1.05, "size": 0.4},  # +5% -> vend 40%
        ]
        self.trailing_activation = 1.03  # Active trailing à +3%
        self.trailing_stop_distance = 0.015  # Stop suiveur à 1.5%
        self.max_loss = 0.02  # Stop loss à -2%

        # Par défaut : 30% à +3%, 30% à +7%, 40% trailing
        # self.tp_levels = tp_levels or [(0.03, 0.3), (0.07, 0.3)]
        self.trailing_pct = trailing_pct

    def get_tp_targets(self, entry_price):
        """
        Calcule les niveaux de take profit en fonction du prix d'entrée.
        Retourne une liste de tuples (prix_target, fraction)
        """
        return [(entry_price * (1 + pct), fraction) for pct, fraction in self.tp_levels]

    def check_tp_partial(self, entry_price, current_price, filled_targets):
        """
        Vérifie si des cibles de TP sont atteintes.
        Retourne la fraction à sortir et le nouveau filled_targets.
        filled_targets: liste de booléens indiquant si le TP a déjà été atteint.
        """
        targets = self.get_tp_targets(entry_price)
        to_exit = 0
        new_filled = filled_targets.copy()
        for i, (tp_price, fraction) in enumerate(targets):
            if not filled_targets[i] and current_price >= tp_price:
                to_exit += fraction
                new_filled[i] = True
        return to_exit, new_filled

    def check_trailing(self, entry_price, price_history, trailing_base=None):
        """
        price_history: liste des prix depuis l'entrée
        trailing_base: le plus haut atteint depuis l'entrée (pour le trailing)
        Retourne (should_exit, nouveau_max_price)
        should_exit: True si le trailing TP doit déclencher la vente
        nouveau_max_price: le plus haut atteint depuis l'entrée
        """
        if not price_history:
            return False, trailing_base
        max_price = max(trailing_base or price_history[0], max(price_history))
        trigger_price = max_price * (1 - self.trailing_pct)
        should_exit = price_history[-1] <= trigger_price
        return should_exit, max_price

    def is_tp_near(self, pos, threshold=0.9):
        """
        Détecte si le prix actuel est proche d'un niveau TP.
        pos: dictionnaire de position avec 'entry_price' et 'current_price'
        threshold: ex. 0.9 = 90% du TP atteint
        Retourne True si le prix actuel >= 90% du niveau TP le plus proche non atteint
        """
        entry = pos.get("entry_price")
        current = pos.get("current_price")
        if entry is None or current is None:
            return False
        # Vérifie le premier TP non encore rempli
        for i, (tp_pct, fraction) in enumerate(self.tp_levels):
            tp_price = entry * (1 + tp_pct)
            # Si le TP n'est pas encore rempli et le prix est proche du TP
            filled_targets = pos.get("filled_tp_targets", [False] * len(self.tp_levels))
            if not filled_targets[i]:
                if current >= entry + (tp_price - entry) * threshold:
                    return True
        return False
