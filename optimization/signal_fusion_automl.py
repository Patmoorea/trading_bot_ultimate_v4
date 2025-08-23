import os
import json
import numpy as np
import pandas as pd
import optuna
import time
import functools
import asyncio

from datetime import datetime
from src.backtesting.core.backtest_engine import BacktestEngine
from src.analysis.news.sentiment_analyzer import NewsSentimentAnalyzer
from src.ai.deep_learning_model import DeepLearningModel
from binance.client import Client
from dotenv import load_dotenv

BEST_PARAMS_PATH = "config/best_signal_params.json"
DATA_CACHE_DIR = "data_cache"

load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_INTERVAL_MAP = {
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
}


def get_all_pairs_from_bot_config(config_path="config/trading_pairs.json"):
    """Charge toutes les paires configurées pour le bot."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f).get("valid_pairs", ["BTC/USDC", "ETH/USDC"])
    return ["BTC/USDC", "ETH/USDC"]


class DummyBot:
    def __init__(self, config, dl_model):
        self.config = config
        self.dl_model = dl_model
        self.news_analyzer = NewsSentimentAnalyzer(config)

    def add_indicators(self, df):
        try:
            import pandas_ta as pta

            df = df.copy()
            df["rsi_14"] = pta.rsi(df["close"], length=14)
            macd = pta.macd(df["close"])
            if macd is not None and not macd.empty:
                df["macd"] = macd["MACD_12_26_9"]
            else:
                df["macd"] = 0.0
            indics = {
                "rsi_14": df["rsi_14"].iloc[-1] if "rsi_14" in df else 50,
                "macd": df["macd"].iloc[-1] if "macd" in df else 0.0,
            }
            return indics
        except Exception:
            return {"rsi_14": 50, "macd": 0.0}


def get_enriched_sentiment(bot, pair_key, news_list):
    import asyncio

    news_list = bot.news_analyzer.analyze_sentiment_batch(news_list)
    try:
        loop = asyncio.get_running_loop()
        sentiment_score = loop.run_until_complete(
            bot.news_analyzer.get_symbol_sentiment(pair_key, news_list=news_list)
        )
    except RuntimeError:
        sentiment_score = asyncio.run(
            bot.news_analyzer.get_symbol_sentiment(pair_key, news_list=news_list)
        )
    summary = bot.news_analyzer.get_sentiment_summary()
    sentiment_global = summary.get("sentiment_global", 0.0)
    impact_score = summary.get("impact_score", 0.0)
    n_news = summary.get("n_news", 0)
    impact_factor = min(2.0, 1.0 + impact_score) if n_news > 15 else 1.0
    if sentiment_score == 0:
        sentiment_score = sentiment_global * impact_factor
    major_events = summary.get("major_events", "")
    if major_events and sentiment_score != 0:
        sentiment_score *= 1.2
    sentiment_score = np.clip(sentiment_score, -1, 1)
    print(
        f"[Sentiment] {pair_key}: {sentiment_score} (global={sentiment_global}, impact={impact_score}, n_news={n_news})"
    )
    return sentiment_score


def enrich_signals_with_real_values(bot, df, pair_key, news_list=None, window=20):
    indics = bot.add_indicators(df)
    rsi = indics.get("rsi_14", 50)
    df["signal_tech"] = (rsi - 50) / 50

    for col, default in [("close", 0.0), ("high", 0.0), ("low", 0.0), ("volume", 0.0)]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    df["rsi"] = df["rsi_14"] if "rsi_14" in df.columns else 50.0
    if "macd" not in df.columns and "macd" in indics:
        df["macd"] = indics["macd"] if indics["macd"] is not None else 0.0
    elif "macd" not in df.columns:
        df["macd"] = 0.0
    df["macd"] = df["macd"].fillna(0.0)
    if "volatility" not in df.columns:
        df["volatility"] = 0.0
    df["volatility"] = df["volatility"].fillna(0.0)
    df["rsi"] = df["rsi"].fillna(50.0)

    def ia_predictor(row):
        idx = row.name
        if idx < window - 1:
            return 0.0
        window_df = df.iloc[max(0, idx - window + 1) : idx + 1].copy()
        features = {
            "close": np.array(window_df["close"]),
            "high": np.array(window_df["high"]),
            "low": np.array(window_df["low"]),
            "volume": np.array(window_df["volume"]),
            "rsi": window_df["rsi"].iloc[-1] if "rsi" in window_df else 50.0,
            "macd": window_df["macd"].iloc[-1] if "macd" in window_df else 0.0,
            "volatility": (
                window_df["volatility"].iloc[-1] if "volatility" in window_df else 0.0
            ),
        }
        try:
            return float(bot.dl_model.predict(features))
        except Exception as e:
            print(f"Error in DL prediction: {e}")
            return 0.0

    df["signal_ia"] = df.apply(ia_predictor, axis=1)

    if hasattr(bot, "news_analyzer") and bot.news_analyzer:
        import asyncio

        if news_list is None:
            try:
                loop = asyncio.get_running_loop()
                news_list = loop.run_until_complete(bot.news_analyzer.fetch_all_news())
            except RuntimeError:
                news_list = asyncio.run(bot.news_analyzer.fetch_all_news())
        sentiment_score = get_enriched_sentiment(bot, pair_key, news_list)
        df["signal_sentiment"] = sentiment_score
        print(
            f"[Signal Sentiment in DataFrame] {pair_key}: {df['signal_sentiment'].iloc[0]}"
        )
    else:
        df["signal_sentiment"] = 0.0
        print(f"[Signal Sentiment in DataFrame] {pair_key}: 0.0 (no analyzer)")
    return df


def fetch_binance_ohlcv(
    symbol, interval, start_str, end_str, api_key, api_secret, retries=5, timeout=30
):
    from time import sleep

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    cache_file = f"{DATA_CACHE_DIR}/{symbol}_{interval}_{start_str}_{end_str}.csv"
    if os.path.exists(cache_file):
        print(f"[CACHE] Lecture depuis: {cache_file}")
        df = pd.read_csv(cache_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    client = Client(api_key, api_secret)
    client.session.request = functools.partial(client.session.request, timeout=timeout)
    last_exception = None
    for attempt in range(retries):
        try:
            klines = client.get_historical_klines(symbol, interval, start_str, end_str)
            if klines and len(klines) > 0:
                break
        except Exception as e:
            print(
                f"[FETCH] Attempt {attempt+1}/{retries} failed for {symbol} {interval} (error: {e})"
            )
            last_exception = e
            sleep(5)
    else:
        print(f"[FETCH] All retries failed for {symbol} {interval}")
        return None
    if not klines or len(klines) == 0:
        print(f"[FETCH] No data for {symbol} {interval}")
        return None
    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    try:
        df.to_csv(cache_file, index=False)
        print(f"[CACHE] Généré: {cache_file}")
    except Exception as e:
        print(f"[CACHE SAVE ERROR] {e}")
    return df


def fusion_signal_series(df, fusion_params):
    print("Tech min/max:", df["signal_tech"].min(), df["signal_tech"].max())
    print("IA min/max:", df["signal_ia"].min(), df["signal_ia"].max())
    print(
        "Sentiment min/max:", df["signal_sentiment"].min(), df["signal_sentiment"].max()
    )
    scale = fusion_params.get("scale", 2)
    fusion_scores = df.apply(
        lambda row: scale
        * (
            fusion_params["tech_weight"] * row.get("signal_tech", 0)
            + fusion_params["ia_weight"] * row.get("signal_ia", 0)
            + fusion_params["sentiment_weight"] * row.get("signal_sentiment", 0)
        ),
        axis=1,
    )
    print("Fusion min/max:", fusion_scores.min(), fusion_scores.max())
    print("Fusion describe:", fusion_scores.describe())
    buy_threshold = fusion_params.get("buy_threshold", 0.8)
    sell_threshold = fusion_params.get("sell_threshold", -0.8)

    def fusion(row):
        score = scale * (
            fusion_params["tech_weight"] * row.get("signal_tech", 0)
            + fusion_params["ia_weight"] * row.get("signal_ia", 0)
            + fusion_params["sentiment_weight"] * row.get("signal_sentiment", 0)
        )
        if score >= buy_threshold:
            return 1
        elif score <= sell_threshold:
            return -1
        return 0

    signals = df.apply(fusion, axis=1)
    print("Signal counts:", pd.Series(signals).value_counts())
    return signals


def run_full_backtest(df, fusion_params, initial_capital=10000, verbose=False):
    signals = fusion_signal_series(df, fusion_params)
    print("[DEBUG] Signals distribution:", pd.Series(signals).value_counts())
    results = BacktestEngine(initial_capital=initial_capital).run_backtest(
        df, lambda df, idx=None, **_kwargs: signals if idx is None else signals[idx]
    )
    print("[DEBUG] Backtest results:", results)
    if verbose:
        print(f"[BACKTEST] Résultat: {results}")
    return results


def print_trial_progress(study, trial):
    print(
        f"[Optuna] Trial {trial.number} terminé : Value={trial.value} | "
        f"Params={trial.params} | Best Value={study.best_value:.4f} (Trial {study.best_trial.number})"
    )


def save_best_params(study, trial, path=BEST_PARAMS_PATH):
    best_params = study.best_trial.params
    best_params["score"] = study.best_value
    with open(path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"[Optuna] Best params sauvegardés: {best_params}")


def optuna_callback(study, trial):
    print_trial_progress(study, trial)
    save_best_params(study, trial)


async def analyze_pair_tf(pair, tf, bot, fusion_params, window):
    df = fetch_binance_ohlcv(
        symbol=pair.replace("/", ""),
        interval=BINANCE_INTERVAL_MAP[tf],
        start_str="1 Jan, 2023",
        end_str="now",
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
    )
    if df is None or len(df) < window + 10:
        return None
    import pandas_ta as pta

    df["rsi"] = pta.rsi(df["close"], length=14)
    macd = pta.macd(df["close"])
    if macd is not None and not macd.empty:
        df["macd"] = macd["MACD_12_26_9"]
    else:
        df["macd"] = 0.0
    returns = np.log(df["close"]).diff()
    df["volatility"] = returns.rolling(14).std()
    for col in ["rsi", "macd", "volatility"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill").fillna(0)
    df = enrich_signals_with_real_values(
        bot, df, pair_key=pair.replace("/", ""), window=window
    )
    results = run_full_backtest(df, fusion_params, initial_capital=10000)
    profit = results.get("final_capital", 0) - 10000 if results else -9999
    if profit is None or np.isnan(profit):
        profit = -99999
    return profit


def optimize_signal_fusion_and_mm(n_trials=50):
    print("=== [DIAG] OPTIMIZATION FUNCTION CALLED ===")
    config = {
        "TRADING": {
            "pairs": get_all_pairs_from_bot_config(),  # PATCH: Toutes les paires de la config bot !
            "timeframes": ["1h", "4h"],
        },
        "news": {
            "sentiment_weight": 0.15,
            "update_interval": 300,
            "storage_path": "data/news_analysis.json",
            "low_watermark_ratio": 0.75,
            "symbol_mapping": {
                "bitcoin": "BTC",
                "ethereum": "ETH",
                "cardano": "ADA",
                "solana": "SOL",
                "litecoin": "LTC",
                "xrp": "XRP",
                "dogecoin": "DOGE",
                "binancecoin": "BNB",
                "tron": "TRX",
                "sui": "SUI",
                "stablecoin": "USDT",
                "ink": "INK",
                "ena": "ENA",
                "ledger": "BTC",
                "tether": "USDT",
            },
        },
    }
    window = 20

    def objective(trial):
        print(f"=== [DIAG] Optuna trial {trial.number} started ===")
        lr = trial.suggest_float("lr", 1e-5, 2e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        n_epochs = trial.suggest_int("n_epochs", 3, 10)
        tech_weight = trial.suggest_float("tech_weight", 0.0, 1.0)
        ia_weight = trial.suggest_float("ia_weight", 0.0, 1.0 - tech_weight)
        sentiment_weight = 1.0 - tech_weight - ia_weight
        buy_threshold = trial.suggest_float("buy_threshold", 0.1, 0.5)
        sell_threshold = trial.suggest_float("sell_threshold", -0.5, -0.1)
        mm_risk = trial.suggest_float("mm_risk", 0.01, 0.2)
        fusion_params = {
            "tech_weight": tech_weight,
            "ia_weight": ia_weight,
            "sentiment_weight": sentiment_weight,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "mm_risk": mm_risk,
            "lr": lr,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        }

        dl_model = DeepLearningModel()
        dl_model.initialize()
        print("[DIAG] About to call dl_model.train()")
        pairs = config["TRADING"]["pairs"]
        timeframes = config["TRADING"]["timeframes"]
        train_dfs = []
        for pair in pairs:
            for tf in timeframes:
                df = fetch_binance_ohlcv(
                    symbol=pair.replace("/", ""),
                    interval=BINANCE_INTERVAL_MAP[tf],
                    start_str="1 Jan, 2023",
                    end_str="now",
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_API_SECRET,
                )
                if df is not None and len(df) >= window + 10:
                    df["target"] = (df["close"].shift(-5) > df["close"]).astype(float)
                    import pandas_ta as pta

                    df["rsi"] = pta.rsi(df["close"], length=14)
                    macd = pta.macd(df["close"])
                    if macd is not None and not macd.empty:
                        df["macd"] = macd["MACD_12_26_9"]
                    else:
                        df["macd"] = 0.0
                    returns = np.log(df["close"]).diff()
                    df["volatility"] = returns.rolling(14).std()
                    for col in ["rsi", "macd", "volatility"]:
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                        df[col] = (
                            df[col]
                            .fillna(method="ffill")
                            .fillna(method="bfill")
                            .fillna(0)
                        )
                    train_dfs.append(df)
        all_df = pd.concat(train_dfs) if train_dfs else None
        if all_df is not None:
            try:
                dl_model.train(
                    all_df,
                    lr=lr,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    window=window,
                )
            except Exception as e:
                print(f"[WARN TRAIN IA] {e}")
        else:
            print("[WARN] No training data for IA, using default weights.")

        bot = DummyBot(config, dl_model)
        fetch_result = bot.news_analyzer.fetch_all_news()
        if hasattr(fetch_result, "__await__"):
            try:
                loop = asyncio.get_running_loop()
                loop.run_until_complete(fetch_result)
            except RuntimeError:
                asyncio.run(fetch_result)

        # === PARALLELISATION ANALYSE ===
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        tasks = [
            analyze_pair_tf(pair, tf, bot, fusion_params, window)
            for pair in pairs
            for tf in timeframes
        ]
        all_scores = loop.run_until_complete(asyncio.gather(*tasks))
        all_scores = [s for s in all_scores if s is not None]
        avg_profit = np.mean(all_scores) if all_scores else -99999
        print(f"[OPTUNA] Params: {fusion_params} | Score: {avg_profit:.2f}")
        return avg_profit

    print("=== [DIAG] CREATING STUDY ===")
    study = optuna.create_study(direction="maximize")
    print("=== [DIAG] RUNNING OPTIMIZATION ===")
    study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
    print("=== [DIAG] OPTIMIZATION DONE ===")
    print("Best params:", study.best_params)
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_params


if __name__ == "__main__":
    print(
        "=== OPTIMISATION SIGNAL FUSION & MM (FinBERT NLP + IA réelle + Optuna tuning IA) ==="
    )
    best = optimize_signal_fusion_and_mm(n_trials=100)
    print("Meilleure configuration trouvée :", best)
