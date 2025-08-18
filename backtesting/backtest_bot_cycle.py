import asyncio
import pandas as pd

async def backtest_bot_cycle(
    bot, 
    historical_data,   # dict: { "BTC/USDC": df, ... }
    news_data=None,    # optionnel: liste de news simulées
    n_cycles=1000,     # nombre de cycles à simuler
    timeframes=["1h", "4h"], 
    capital_start=10000.0  # capital initial
):
    # 1. Reset du bot
    bot.positions = {}
    bot.positions_binance = {}
    bot.current_cycle = 0
    bot.market_data = {}
    bot.indicators = {}
    bot.trade_decisions = {}
    bot.regime = "RANGING"
    bot.news_pause_manager.reset_pauses([])
    bot.is_live_trading = False
    bot.ai_enabled = True
    bot.news_enabled = True
    bot.data_file = "src/shared_data_backtest.json"
    bot.save_shared_data()

    # 2. Injection des données historiques dans bot.market_data
    for pair, df in historical_data.items():
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
                # Ajout des signaux calculés
                "signals": bot.add_indicators(df),
            }
        # Ajout du sentiment si dispo
        bot.market_data[pair_key]["sentiment"] = 0.0

    # 3. Boucle de cycles simulés
    for cycle in range(n_cycles):
        bot.current_cycle = cycle
        # Simule l'arrivée de news si besoin
        if news_data and cycle < len(news_data):
            # Ajoute la news au bot
            try:
                await bot._update_sentiment_data([news_data[cycle]])
            except Exception:
                pass

        # Exécute un cycle de trading
        trade_decisions, regime = await execute_trading_cycle(bot, bot.pairs_valid)

        # Exécute les trades (simu, pas d'ordres réels)
        await execute_trade_decisions(bot, trade_decisions)

        # Mise à jour des métriques
        bot.save_shared_data()

    # 4. Résumé de performance
    perf = bot.get_performance_metrics()
    return perf, bot.market_data, bot.trade_decisions