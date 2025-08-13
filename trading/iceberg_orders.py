from typing import Dict, Optional
import asyncio
from decimal import Decimal


class EnhancedIcebergOrder:
    def __init__(
        self,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        visible_size: Decimal,
        price: Optional[Decimal] = None,
    ):
        self.symbol = symbol
        self.side = side.upper()
        self.total_quantity = total_quantity
        self.visible_size = visible_size
        self.price = price
        self.executed_quantity = Decimal("0")
        self.active_orders: Dict[str, Dict] = {}

    async def execute(self, exchange):
        remaining = self.total_quantity
        while remaining > Decimal("0"):
            visible = min(remaining, self.visible_size)
            try:
                order = await exchange.create_order(
                    symbol=self.symbol.replace("/", ""),
                    type="limit",
                    side=self.side,
                    amount=float(visible),
                    price=float(self.price) if self.price else None,
                )
                self.active_orders[order["id"]] = order
                remaining -= visible
                # Attendre que l'ordre soit rempli
                await self._wait_for_fill(exchange, order["id"])
            except Exception as e:
                print(f"Erreur lors de l'exécution: {e}")
                break

    async def _wait_for_fill(self, exchange, order_id: str):
        while True:
            order = await exchange.fetch_order(order_id)
            if order["status"] == "closed":
                self.executed_quantity += Decimal(str(order["filled"]))
                break
            await asyncio.sleep(1)
