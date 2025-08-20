import numpy as np


class ExitManager:
    def __init__(self, tp_levels=None, trailing_pct=0.03):
        """
        tp_levels: Liste de tuples (target_pct, fraction) ex: [(0.03, 0.3), (0.07, 0.3)]
        trailing_pct: Pourcentage du trailing stop (ex: 0.03 = 3%)
        """
        # CORRECTION : Utilisez soit des tuples, soit adaptez le code aux dicts
        self.tp_levels = tp_levels or [
            (0.02, 0.3),  # +2% -> vend 30%
            (0.035, 0.3),  # +3.5% -> vend 30%
            (0.05, 0.4),  # +5% -> vend 40%
        ]

        self.trailing_activation = 1.03  # Active trailing à +3%
        self.trailing_stop_distance = 0.015  # Stop suiveur à 1.5%
        self.max_loss = 0.02  # Stop loss à -2%
        self.trailing_pct = trailing_pct

    def get_tp_targets(self, entry_price):
        """
        Calcule les niveaux de take profit en fonction du prix d'entrée.
        Retourne une liste de tuples (prix_target, fraction)
        """
        try:
            entry = float(entry_price)
            return [(entry * (1 + pct), fraction) for pct, fraction in self.tp_levels]
        except Exception as e:
            print(f"[ERROR] Dans get_tp_targets: {e}")
            return []

    def check_tp_partial(self, entry_price, current_price, filled_targets):
        """
        Vérifie si des cibles de TP sont atteintes.
        Retourne la fraction à sortir et le nouveau filled_targets.
        filled_targets: liste de booléens indiquant si le TP a déjà été atteint.
        """
        try:
            entry = float(entry_price)
            current = float(current_price)

            targets = self.get_tp_targets(entry)
            to_exit = 0
            new_filled = (
                filled_targets.copy() if filled_targets else [False] * len(targets)
            )

            for i, (tp_price, fraction) in enumerate(targets):
                if i < len(new_filled) and not new_filled[i] and current >= tp_price:
                    to_exit += fraction
                    new_filled[i] = True
            return to_exit, new_filled

        except Exception as e:
            print(f"[ERROR] Dans check_tp_partial: {e}")
            return 0, filled_targets or [False] * len(self.tp_levels)

    def check_trailing(
        self, entry_price, price_history, trailing_base=None, trailing_pct=None
    ):
        """
        Version ultra-sécurisée de check_trailing
        Accepte maintenant le 4ème paramètre trailing_pct
        """
        try:
            # Conversion de tous les paramètres en float
            entry = float(entry_price) if entry_price is not None else 0.0

            # Utilise le trailing_pct passé en paramètre ou celui par défaut
            trailing_pct_to_use = (
                float(trailing_pct)
                if trailing_pct is not None
                else float(self.trailing_pct)
            )

            # Conversion de l'historique des prix
            numeric_history = []
            for price in price_history:
                try:
                    numeric_history.append(float(price))
                except (ValueError, TypeError):
                    numeric_history.append(0.0)

            # Conversion de trailing_base
            try:
                base = (
                    float(trailing_base)
                    if trailing_base is not None
                    else (max(numeric_history) if numeric_history else entry)
                )
            except:
                base = entry

            if not numeric_history:
                return False, base

            # Calcul du prix maximum
            current_max = max(numeric_history)
            max_price = max(base, current_max)

            # Calcul du prix de déclenchement
            trigger_price = max_price * (1 - trailing_pct_to_use)

            # Dernier prix
            last_price = numeric_history[-1] if numeric_history else 0.0

            should_exit = last_price <= trigger_price
            return should_exit, max_price

        except Exception as e:
            print(f"[CRITICAL] Erreur dans check_trailing: {e}")
            # Fallback total
            try:
                fallback_max = (
                    float(trailing_base)
                    if trailing_base is not None
                    else float(entry_price) if entry_price is not None else 0.0
                )
                return False, fallback_max
            except:
                return False, 0.0

    def is_tp_near(self, pos, threshold=0.9):
        """
        Détecte si le prix actuel est proche d'un niveau TP.
        pos: dictionnaire de position avec 'entry_price' et 'current_price'
        threshold: ex. 0.9 = 90% du TP atteint
        Retourne True si le prix actuel >= 90% du niveau TP le plus proche non atteint
        """
        try:
            entry = float(pos.get("entry_price", 0))
            current = float(pos.get("current_price", 0))

            if entry == 0 or current == 0:
                return False

            # Vérifie le premier TP non encore rempli
            filled_targets = pos.get("filled_tp_targets", [False] * len(self.tp_levels))

            for i, (tp_pct, fraction) in enumerate(self.tp_levels):
                if i < len(filled_targets) and not filled_targets[i]:
                    tp_price = entry * (1 + tp_pct)
                    if current >= entry + (tp_price - entry) * threshold:
                        return True
            return False

        except Exception as e:
            print(f"[ERROR] Dans is_tp_near: {e}")
            return False
