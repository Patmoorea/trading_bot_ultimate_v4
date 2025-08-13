from decimal import Decimal, ROUND_DOWN
import logging
# Configuration du logging
logger = logging.getLogger(__name__)
class PositionManager:
    def __init__(
        self,
        account_balance,
        max_positions=5,
        max_leverage=3.0,
        min_position_size=0.001,
        max_risk_per_trade=0.02
    ):
        """
        Initialise le gestionnaire de positions
        :param account_balance: Balance du compte
        :param max_positions: Nombre maximum de positions simultanées
        :param max_leverage: Levier maximum autorisé
        :param min_position_size: Taille minimum d'une position
        :param max_risk_per_trade: Risque maximum par trade (2% par défaut)
        """
        self.account_balance = Decimal(str(account_balance))
        self.max_positions = max_positions
        self.max_leverage = Decimal(str(max_leverage))
        self.min_position_size = Decimal(str(min_position_size))
        self.max_risk_per_trade = Decimal(str(max_risk_per_trade))
        self.current_positions = []
    def calculate_position_size(self, entry_price, stop_loss, leverage=1):
        """
        Calcule la taille de position optimale selon le risque défini
        """
        try:
            # Vérification du levier
            if Decimal(str(leverage)) > self.max_leverage:
                logger.warning(f"[{self.current_time}] Levier {leverage} supérieur au maximum autorisé {self.max_leverage}")
                leverage = float(self.max_leverage)
            risk_amount = self.account_balance * self.max_risk_per_trade
            price_diff = abs(Decimal(str(entry_price)) - Decimal(str(stop_loss)))
            position_size = (risk_amount / price_diff) * Decimal(str(leverage))
            # Application de la taille minimum
            if position_size < self.min_position_size:
                logger.warning(f"[{self.current_time}] Taille calculée {position_size} inférieure au minimum {self.min_position_size}")
                return self.min_position_size
            return position_size.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur calcul taille position: {e}")
            return self.min_position_size
    def validate_trade(self, position_size, current_positions=None):
        """
        Vérifie si le trade respecte les règles de gestion du risque
        """
        try:
            if current_positions is None:
                current_positions = self.current_positions
            # Vérification du nombre de positions
            if len(current_positions) >= self.max_positions:
                logger.warning(f"[{self.current_time}] Nombre maximum de positions atteint ({self.max_positions})")
                return False
            # Vérification de l'exposition totale
            total_exposure = sum(Decimal(str(pos['size'])) for pos in current_positions)
            new_exposure = total_exposure + Decimal(str(position_size))
            max_exposure = self.account_balance * Decimal('2.5')  # 250% max exposure
            if new_exposure > max_exposure:
                logger.warning(f"[{self.current_time}] Exposition maximale dépassée: {float(new_exposure):.2f} > {float(max_exposure):.2f}")
                return False
            return True
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur validation trade: {e}")
            return False
    def check_position_limits(self, position_size):
        """Vérifie si une nouvelle position respecte les limites"""
        try:
            # Vérification de la taille minimum
            if Decimal(str(position_size)) < self.min_position_size:
                logger.warning(f"[{self.current_time}] Taille {position_size} inférieure au minimum {self.min_position_size}")
                return False
            # Vérification du nombre de positions
            if len(self.current_positions) >= self.max_positions:
                logger.warning(f"[{self.current_time}] Limite de positions atteinte ({self.max_positions})")
                return False
            return True
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur vérification limites: {e}")
            return False
    def calculate_drawdown(self):
        """Calcule le drawdown actuel"""
        try:
            if not self.current_positions:
                return 0.0
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.current_positions)
            return float(total_pnl) / float(self.account_balance)
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur calcul drawdown: {e}")
            return 0.0
    def calculate_daily_loss(self):
        """Calcule la perte journalière"""
        try:
            daily_positions = [p for p in self.current_positions if p.get('opening_time', '').startswith(self.current_time[:10])]
            if not daily_positions:
                return 0.0
            daily_pnl = sum(pos.get('realized_pnl', 0) for pos in daily_positions)
            return abs(float(daily_pnl)) / float(self.account_balance)
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur calcul perte journalière: {e}")
            return 0.0
