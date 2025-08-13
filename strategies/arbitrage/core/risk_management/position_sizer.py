from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

class PositionSizer:
    def __init__(self, max_position: float = 1000.0, max_percentage: float = 5.0, initial_size: float = 100.0):
        self.max_position = Decimal(str(max_position))
        self.max_percentage = Decimal(str(max_percentage))
        self.initial_size = Decimal(str(initial_size))
        self.account_balance = Decimal('0')

    def update_account_balance(self, balance: float) -> None:
        """Met à jour le solde du compte"""
        self.account_balance = Decimal(str(balance))

    def calculate_position_size(self, spread: float, pair: str) -> Decimal:
        """Calcule la taille de position optimale"""
        if self.account_balance == Decimal('0'):
            return Decimal('0')

        # Calculer la limite basée sur le pourcentage du compte
        balance_limit = self.account_balance * self.max_percentage / Decimal('100')
        
        # Ajuster la taille en fonction du spread
        spread_decimal = Decimal(str(spread))
        if spread_decimal > Decimal('0.01'):  # Spread > 1%
            size_multiplier = Decimal('1.5')
        elif spread_decimal > Decimal('0.005'):  # Spread > 0.5%
            size_multiplier = Decimal('1.2')
        else:
            size_multiplier = Decimal('1.0')

        # Calculer la taille suggérée
        suggested_size = min(
            self.max_position,
            balance_limit,
            self.initial_size * size_multiplier
        )

        return suggested_size

    def reduce_limits(self, factor: Decimal) -> None:
        """Réduit les limites de position"""
        self.max_position *= factor
        logger.info(f"Limites de position réduites à {float(self.max_position):.2f}")

    def increase_limits(self, factor: Decimal) -> None:
        """Augmente les limites de position"""
        self.max_position *= factor
        logger.info(f"Limites de position augmentées à {float(self.max_position):.2f}")
