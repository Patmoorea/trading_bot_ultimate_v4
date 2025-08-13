import numpy as np
from ccxt import binance
class IcebergEngine:
    def __init__(self, api_key: str):
        self.exchange = binance({'apiKey': api_key})
    def optimize_order(self, symbol: str, amount: float) -> dict:
        ob = self.exchange.fetch_order_book(symbol)
        best_bid = ob['bids'][0][0]
        liquidity = sum([bid[1] for bid in ob['bids'][:5]])
        slippage = np.log(amount + 1) / liquidity  
        price = best_bid * (1 - slippage)
        return {
            "price": round(price, 4),
            "size": amount,
            "slippage": float(slippage)
        }
