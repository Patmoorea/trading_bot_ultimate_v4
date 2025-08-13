def get_avg_entry_price_binance_spot(client, asset, quote="USDC"):
    symbol = f"{asset}{quote}"
    try:
        trades = client.get_my_trades(symbol=symbol)
    except Exception as e:
        print(f"Erreur API get_my_trades pour {symbol}: {e}")
        return None
    total_qty = 0
    total_cost = 0
    for t in trades:
        if t["isBuyer"]:
            qty = float(t["qty"])
            price = float(t["price"])
            total_qty += qty
            total_cost += qty * price
    if total_qty > 0:
        return total_cost / total_qty
    else:
        return None
