import random
import numpy as np
from src.backtesting.core.backtest_engine import BacktestEngine
from src.strategies import sma_strategy, breakout_strategy, arbitrage_strategy

# Liste d'indicateurs simples pour la génération auto
INDICATORS = [
    "sma",  # Moyenne mobile simple
    "ema",  # Moyenne mobile exponentielle
    "rsi",
    "macd",
    "bbands",
    "donchian",
]


# Génération d'une stratégie aléatoire simple
def random_strategy_config():
    config = {}
    n_indics = random.randint(1, 3)
    config["indicators"] = random.sample(INDICATORS, n_indics)
    for ind in config["indicators"]:
        if ind in ["sma", "ema"]:
            config[f"{ind}_window"] = random.choice([10, 20, 50, 100])
        if ind == "rsi":
            config["rsi_period"] = random.choice([7, 14, 21])
            config["rsi_buy"] = random.uniform(20, 40)
            config["rsi_sell"] = random.uniform(60, 80)
        if ind == "macd":
            config["macd_fast"] = random.choice([12, 26])
            config["macd_slow"] = random.choice([26, 50])
            config["macd_signal"] = random.choice([9, 12])
        if ind == "bbands":
            config["bbands_window"] = random.choice([14, 20])
            config["bbands_std"] = random.choice([1.5, 2, 2.5])
        if ind == "donchian":
            config["donchian_window"] = random.choice([10, 20, 50])
    return config


def backtest_generated_strategy(df, config):
    signals = np.zeros(len(df))
    if "sma" in config["indicators"]:
        sma = df["close"].rolling(config["sma_window"]).mean()
        signals += (df["close"] > sma).astype(int) - (df["close"] < sma).astype(int)
    if "ema" in config["indicators"]:
        ema = df["close"].ewm(span=config["ema_window"], adjust=False).mean()
        signals += (df["close"] > ema).astype(int) - (df["close"] < ema).astype(int)
    if "rsi" in config["indicators"]:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(config["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(config["rsi_period"]).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        signals += (rsi < config["rsi_buy"]).astype(int) - (
            rsi > config["rsi_sell"]
        ).astype(int)
    # Ajoute d'autres indicateurs ici...

    actions = np.where(signals > 0, "buy", np.where(signals < 0, "sell", "hold"))
    initial = df["close"].iloc[0]
    position = 0
    returns = []
    for i in range(1, len(df)):
        if actions[i - 1] == "buy":
            position = 1
        elif actions[i - 1] == "sell":
            position = -1
        returns.append(position * (df["close"].iloc[i] - df["close"].iloc[i - 1]))
    total_return = sum(returns)
    return total_return


def auto_generate_and_backtest(df, n_strats=20):
    best_score = None
    best_config = None
    for _ in range(n_strats):
        config = random_strategy_config()
        score = backtest_generated_strategy(df, config)
        if (best_score is None) or (score > best_score):
            best_score = score
            best_config = config
    return best_config, best_score


def appliquer_config_strategy(df, config):
    import numpy as np

    signals = np.zeros(len(df))
    if "sma" in config["indicators"]:
        sma = df["close"].rolling(config["sma_window"]).mean()
        signals += (df["close"] > sma).astype(int) - (df["close"] < sma).astype(int)
    if "ema" in config["indicators"]:
        ema = df["close"].ewm(span=config["ema_window"], adjust=False).mean()
        signals += (df["close"] > ema).astype(int) - (df["close"] < ema).astype(int)
    if "rsi" in config["indicators"]:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(config["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(config["rsi_period"]).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        signals += (rsi < config["rsi_buy"]).astype(int) - (
            rsi > config["rsi_sell"]
        ).astype(int)
    # ... autres indicateurs si ajoutés

    last_signal = signals[-1]
    if last_signal > 0:
        return "buy"
    elif last_signal < 0:
        return "sell"
    else:
        return "hold"
