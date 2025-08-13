from prometheus_client import start_http_server, Gauge

CYCLE = Gauge("trading_cycle", "Cycle actuel du bot")
BALANCE = Gauge("trading_balance", "Balance USDT")
PNL = Gauge("trading_pnl", "Profit & Loss courant")
TRADES = Gauge("trades_total", "Nombre de trades effectués")
LATENCY = Gauge("bot_latency", "Latence cycle (secondes)")


def update_metrics(bot_status):
    CYCLE.set(bot_status.get("cycle", 0))
    BALANCE.set(bot_status.get("performance", {}).get("balance", 0))
    PNL.set(bot_status.get("performance", {}).get("total_profit", 0))
    TRADES.set(bot_status.get("performance", {}).get("total_trades", 0))
    LATENCY.set(bot_status.get("latency", 0))


def start_prometheus_server(port=9900):
    start_http_server(port)
