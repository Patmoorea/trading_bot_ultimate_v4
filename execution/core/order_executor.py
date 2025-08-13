"""
"""
from typing import Dict, List
import asyncio
import logging
from datetime import datetime
class OrderExecutor:
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        self.active_orders: Dict[str, Dict] = {}
    async def execute_order(self, order: Dict) -> Dict:
        try:
            exchange = self.exchange_manager.get_exchange(order['exchange'])
            # Validation pré-exécution
            if not self._validate_order(order):
                raise ValueError("Invalid order parameters")
            # Exécution de l'ordre
            response = await self._place_order(exchange, order)
            # Suivi de l'ordre
            self.active_orders[response['id']] = {
                **response,
                'status': 'PENDING',
                'created_at': datetime.utcnow()
            }
            return response
        except Exception as e:
            self.logger.error(f"Order execution error: {str(e)}")
            raise
    def _validate_order(self, order: Dict) -> bool:
        required_fields = ['symbol', 'type', 'side', 'amount']
        return all(field in order for field in required_fields)
    async def _place_order(self, exchange, order: Dict) -> Dict:
        try:
            response = await exchange.create_order(
                symbol=order['symbol'],
                type=order['type'],
                side=order['side'],
                amount=order['amount'],
                price=order.get('price'),
                params=order.get('params', {})
            )
            return response
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise
    async def monitor_orders(self):
        while True:
            for order_id, order in list(self.active_orders.items()):
                try:
                    exchange = self.exchange_manager.get_exchange(order['exchange'])
                    status = await exchange.fetch_order_status(order_id)
                    if status in ['filled', 'cancelled']:
                        self.active_orders.pop(order_id)
                        self.logger.info(f"Order {order_id} {status}")
                except Exception as e:
                    self.logger.error(f"Error monitoring order {order_id}: {str(e)}")
            await asyncio.sleep(1)
