from decimal import Decimal
import asyncio
import time
class OrderManager:
    def __init__(self, exchange_client, logger):
        self.exchange = exchange_client
        self.logger = logger
        self.pending_orders = {}
    async def place_order(self, symbol, order_type, side, amount, price=None):
        """
        Place un ordre avec vérifications de sécurité
        """
        try:
            # Vérifications pre-trade
            self._validate_order_params(symbol, order_type, side, amount, price)
            # Création de l'ordre
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price
            )
            # Enregistrement de l'ordre
            self.pending_orders[order['id']] = {
                'order': order,
                'status': 'pending',
                'created_at': time.time()
            }
            # Log de l'ordre
            self.logger.log_trade({
                'action': 'place_order',
                'order_id': order['id'],
                'details': order
            })
            return order
        except Exception as e:
            self.logger.log_error(f"Erreur lors du placement de l'ordre: {str(e)}")
            raise
    def _validate_order_params(self, symbol, order_type, side, amount, price):
        """
        Vérifie la validité des paramètres de l'ordre
        """
        if not symbol or not order_type or not side:
            raise ValueError("Paramètres manquants")
        if amount <= 0:
            raise ValueError("Montant invalide")
        if order_type == 'limit' and (not price or price <= 0):
            raise ValueError("Prix invalide pour un ordre limit")
        if side not in ['buy', 'sell']:
            raise ValueError("Side invalide")
