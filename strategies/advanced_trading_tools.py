import numpy as np


class GridTrading:
    """
    Simple grid trading logic for spot/futures.
    Usage:
        grid = GridTrading(symbol="BTCUSDT", min_price=25000, max_price=35000, grid_size=10, base_qty=0.01)
        orders = grid.generate_grid_orders()
    """

    def __init__(self, symbol, min_price, max_price, grid_size=10, base_qty=0.01):
        self.symbol = symbol
        self.min_price = min_price
        self.max_price = max_price
        self.grid_size = grid_size
        self.base_qty = base_qty
        self.grid_levels = np.linspace(min_price, max_price, grid_size)

    def generate_grid_orders(self):
        """
        Returns a list of grid levels with type (buy/sell) and quantity.
        """
        grid_orders = []
        for i, price in enumerate(self.grid_levels):
            if i % 2 == 0:
                order_type = "buy"
            else:
                order_type = "sell"
            grid_orders.append(
                {
                    "symbol": self.symbol,
                    "price": float(price),
                    "qty": self.base_qty,
                    "type": order_type,
                }
            )
        return grid_orders

    def check_rebalance(self, current_price, positions):
        """
        Determines if a grid order should be executed based on price and open positions.
        """
        for level in self.grid_levels:
            # Buy if price crosses below a grid level, Sell if above
            if current_price <= level and not positions.get(level, False):
                return {"action": "buy", "price": level, "qty": self.base_qty}
            elif current_price >= level and not positions.get(level, False):
                return {"action": "sell", "price": level, "qty": self.base_qty}
        return None


class DCA:
    """
    Dollar-Cost Averaging logic for spot/futures.
    Usage:
        dca = DCA(symbol="BTCUSDT", base_qty=0.01, max_steps=5)
        order = dca.next_step(current_step)
    """

    def __init__(self, symbol, base_qty=0.01, max_steps=5):
        self.symbol = symbol
        self.base_qty = base_qty
        self.max_steps = max_steps

    def next_step(self, current_step):
        """
        Returns the next DCA order (buy) for the current step.
        """
        if current_step < self.max_steps:
            qty = self.base_qty * (1.5**current_step)
            return {
                "symbol": self.symbol,
                "qty": qty,
                "type": "buy",
                "step": current_step,
            }
        else:
            return None


def anti_slippage(order_price, last_price, slippage_pct=0.2):
    """
    Checks if the slippage is acceptable before sending the order.
    Returns True if order can be sent, False otherwise.
    """
    max_slippage = last_price * slippage_pct / 100
    if abs(order_price - last_price) > max_slippage:
        return False
    return True


# === EXEMPLES D'UTILISATION ===
if __name__ == "__main__":
    # GRID EXAMPLE
    grid = GridTrading("BTCUSDT", 25000, 35000, grid_size=6, base_qty=0.001)
    print("Grid Orders:", grid.generate_grid_orders())
    # DCA EXAMPLE
    dca = DCA("BTCUSDT", base_qty=0.002, max_steps=3)
    for step in range(4):
        print("DCA Step", step, ":", dca.next_step(step))
    # ANTI-SLIPPAGE EXAMPLE
    print("Slippage OK?", anti_slippage(30100, 30000, 0.5))  # -> True
    print("Slippage OK?", anti_slippage(30500, 30000, 0.5))  # -> False
