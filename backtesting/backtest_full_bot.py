import asyncio
import pandas as pd
from src.bot_runner import TradingBotM4, execute_trading_cycle, execute_trade_decisions


async def backtest_full_bot(
    historical_data: dict,  # {"BTC/USDC": df_ohlcv, ...}
    news_data: list = None,  # optionnel: liste de news simulées
    n_cycles: int = 500,  # nombre de cycles à simuler
    capital_start: float = 10000.0,  # capital initial
    timeframes: list = ["1h"],  # TF à utiliser pour la simulation
    pairs: list = None,  # Paires à backtester (par défaut toutes de historical_data)
    debug: bool = False,
):
    """
    Backtest fidèle du bot complet: cycle trading, risk manager, ML, sizing, news, pause, dashboard, etc.
    Utilise TradingBotM4 et appelle execute_trading_cycle/execute_trade_decisions à chaque cycle.
    """
    # 1. Initialisation du bot en mode simulé
    bot = TradingBotM4()
    bot.is_live_trading = False
    bot.ai_enabled = True
    bot.news_enabled = True
    bot.current_cycle = 0
    bot.positions = {}
    bot.positions_binance = {}
    bot.market_data = {}
    bot.indicators = {}
    bot.trade_decisions = {}
    bot.regime = "RANGING"
    bot.data_file = "src/shared_data_backtest.json"
    bot.save_shared_data()
    bot.pairs_valid = pairs if pairs else list(historical_data.keys())

    # 2. Injection des données historiques dans market_data
    for pair in bot.pairs_valid:
        df = historical_data[pair]
        pair_key = pair.replace("/", "").upper()
        bot.market_data[pair_key] = {}
        for tf in timeframes:
            bot.market_data[pair_key][tf] = {
                "open": df["open"].tolist(),
                "high": df["high"].tolist(),
                "low": df["low"].tolist(),
                "close": df["close"].tolist(),
                "volume": df["volume"].tolist(),
                "timestamp": df["timestamp"].tolist(),
                "signals": bot.add_indicators(df),
            }
        bot.market_data[pair_key]["sentiment"] = 0.0

    # 3. Boucle principale du backtest
    for cycle in range(n_cycles):
        bot.current_cycle = cycle
        if debug and cycle % 10 == 0:
            print(f"=== CYCLE {cycle} ===")
        # Injection news/sentiment si fourni
        if news_data and cycle < len(news_data):
            await bot._update_sentiment_data([news_data[cycle]])
        trade_decisions, regime = await execute_trading_cycle(bot, bot.pairs_valid)
        await execute_trade_decisions(bot, trade_decisions)
        # SAUVEGARDE MOINS FREQUENTE !
        if cycle % 50 == 0 or cycle == n_cycles - 1:
            bot.save_shared_data()
        if debug and cycle % 10 == 0:
            perf = bot.get_performance_metrics()
            print(
                f"CYCLE {cycle}: Balance = {perf.get('balance', 0):.2f}, Win Rate = {perf.get('win_rate', 0):.2%}"
            )

    # 4. Résultats de performance
    perf = bot.get_performance_metrics()
    return perf, bot.market_data, bot.trade_decisions


# Exemple d'utilisation:
if __name__ == "__main__":
    df_btc = pd.read_csv("data/historical/BTCUSDC_1h.csv")
    df_eth = pd.read_csv("data/historical/ETHUSDC_1h.csv")
    historical_data = {
        "BTC/USDC": df_btc,
        "ETH/USDC": df_eth,
    }
    news_data = []  # ou charger d'un fichier

    results, market_data, trade_decisions = asyncio.run(
        backtest_full_bot(
            historical_data,
            news_data,
            n_cycles=300,
            capital_start=10000.0,
            timeframes=["1h"],
            debug=True,
        )
    )
    print("Performance backtest :", results)
