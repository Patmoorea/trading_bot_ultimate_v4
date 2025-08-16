import warnings

# Supprimer TOUS les warnings Python
# warnings.filterwarnings("ignore")
# warnings.simplefilter("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Réduit les logs TensorFlow
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"  # Supprime certains warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Optimisation M4
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback GPU
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.75"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
import json
import asyncio
import aiohttp
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import pytz
import argparse
import pandas as pd
import pandas_ta as pta
import pyarrow as pa
import pyarrow.parquet as pq
import lz4.frame
import shutil

from decimal import Decimal
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.analysis.news.cointelegraph_fetcher import fetch_cointelegraph_news
from src.analysis.news.sentiment_analyzer import NewsSentimentAnalyzer

# Obtenir le chemin racine du projet (un niveau au-dessus de l'emplacement du script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import des modules existants avec les bons chemins
from web_interface.app.services.order_execution import SmartOrderExecutor
from src.strategies.arbitrage.execution.execution import ArbitrageExecutor
from src.ai.enhanced_cnn_lstm import EnhancedCNNLSTM
from src.ai_models.hybrid.cnn_lstm_enhanced import EnhancedCNNLSTM
from src.ai.ppo_gtrxl import PPOGTrXL
from src.connectors.binance import BinanceConnector
from src.ai.deep_learning_model import DeepLearningModel
from src.ai.ppo_strategy import PPOStrategy
from src.bot.trading_env import TradingEnv

from src.strategies.arbitrage.core.arbitrage_bot import ArbitrageBot
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner
from src.strategies.arbitrage.core.risk_management.risk_manager import RiskManager
from src.strategies.arbitrage.service import ArbitrageEngine

from src.data.ws_buffered_collector import BufferedWSCollector

from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators

from src.optimization.optuna_wrapper import (
    tune_hyperparameters,
    optimize_hyperparameters_full,
)
from src.security.key_manager import KeyManager

from src.backtesting.core.backtest_engine import BacktestEngine

# Import dynamique des stratégies
from src.strategies import sma_strategy, breakout_strategy, arbitrage_strategy

from src.ai.auto_strategy_generator import auto_generate_and_backtest
from src.ai.auto_strategy_generator import appliquer_config_strategy
from src.ai.train_cnn_lstm import train_with_live_data
from src.ai.deep_learning_model import features_to_array

from collections import defaultdict

from deep_translator import GoogleTranslator
from src.ai.hybrid_model import HybridAI

from bingx_order_executor import BingXOrderExecutor
from src.exchanges.bingx_exchange import BingXExchange

from src.risk_tools import kelly_criterion, calculate_var, calculate_max_drawdown

from src.portfolio.position_sizer import dynamic_position_size, compute_drawdown

from src.portfolio.exit_manager import ExitManager

from src.analysis.filters.volatility_anomaly_filter import filter_market

from src.analysis.filters.correlation_filter import filter_uncorrelated_pairs

from src.risk_tools.news_pause_manager import NewsPauseManager

from src.portfolio.binance_utils import get_avg_entry_price_binance_spot

from cachetools import TTLCache

from src.utils.safe_json_utils import safe_load_shared_data

# from src.risk_tools.enhanced_risk_manager import EnhancedRiskManager

# Charger les variables d'environnement depuis .env
load_dotenv()

LOG_FILE = "src/bot_logs.txt"


class ExchangeConnector:
    """
    Abstraction pour gérer plusieurs exchanges facilement.
    Chaque exchange doit avoir un client Python (Binance, Kucoin, OKX...).
    Tu utilises cette classe pour faire les ordres et récupérer le portefeuille.
    """

    def __init__(self, name, client=None):
        self.name = name
        self.client = client

    def execute_order(self, symbol, side, amount, **kwargs):
        if self.name == "binance":
            # Exemple simplifié, adapte à ton SDK
            return self.client.create_order(
                symbol=symbol, side=side, quantity=amount, **kwargs
            )
        elif self.name == "kucoin":
            # Placeholder, à implémenter
            pass
        elif self.name == "okx":
            # Placeholder, à implémenter
            pass

    def get_portfolio(self):
        if self.name == "binance":
            return self.client.get_account()
        elif self.name == "kucoin":
            pass
        elif self.name == "okx":
            pass

    def get_orderbook(self, symbol):
        if self.name == "binance":
            return self.client.get_order_book(symbol=symbol)
        elif self.name == "kucoin":
            pass
        elif self.name == "okx":
            pass


def safe_float(value, default=0.0):
    """Convertit une valeur en float de manière sécurisée"""
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def deep_cast_floats(d):
    """Cast toutes les valeurs numériques (str/int/float) en float dans un dict ou une liste, récursivement."""
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) or isinstance(v, list):
                deep_cast_floats(v)
            elif isinstance(v, (str, int, float)):
                try:
                    d[k] = safe_float(v, v)
                except Exception:
                    pass
    elif isinstance(d, list):
        for idx, v in enumerate(d):
            if isinstance(v, dict) or isinstance(v, list):
                deep_cast_floats(v)
            elif isinstance(v, (str, int, float)):
                try:
                    d[idx] = safe_float(v, v)
                except Exception:
                    pass


def add_dl_features(df):
    """
    Ajoute les features 'rsi', 'macd', 'volatility' nécessaires à l'entraînement IA.
    Corrige intelligemment les NaN/inf au lieu de tout drop ou reset.
    """

    # Tri par timestamp pour éviter des NaN liés au mauvais ordre
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset="timestamp", keep="last")

    # RSI 14
    if "rsi" not in df or df["rsi"].isnull().all():
        try:
            if len(df) >= 15:
                df["rsi"] = pta.rsi(df["close"], length=14)
            else:
                df["rsi"] = np.nan
        except Exception:
            df["rsi"] = np.nan
    # MACD
    if "macd" not in df or df["macd"].isnull().all():
        try:
            if len(df) >= 27:
                macd = pta.macd(df["close"])
                df["macd"] = macd["MACD_12_26_9"] if "MACD_12_26_9" in macd else np.nan
            else:
                df["macd"] = np.nan
        except Exception:
            df["macd"] = np.nan
    # Volatility
    if "volatility" not in df or df["volatility"].isnull().all():
        try:
            if len(df) >= 15:
                returns = np.log(df["close"]).diff()
                df["volatility"] = returns.rolling(14).std()
            else:
                df["volatility"] = np.nan
        except Exception:
            df["volatility"] = np.nan

    # Nettoyage intelligent NaN/inf (ffill puis bfill puis 0)
    for col in ["rsi", "macd", "volatility"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(method="ffill").fillna(method="bfill").fillna(0)
    return df


def log_dashboard(message):
    print(message)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.utcnow().isoformat()} | {message}\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")


def _generate_analysis_report(
    indicators_analysis, regime, news_sentiment=None, trade_decisions=None
):

    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        "📊 Analyse complète du marché:",
        f"Date: {current_time} UTC",
        f"Régime: {regime}",
        "\nTendances principales:",
    ]
    # Analyse des news
    if news_sentiment:
        try:
            sentiment = float(news_sentiment.get("overall_sentiment", 0) or 0)
            impact = float(news_sentiment.get("impact_score", 0) or 0)
            major_events = news_sentiment.get("major_events", "Aucun")
            report.extend(
                [
                    "\n📰 Analyse des News:",
                    f"Sentiment: {sentiment:.2%}",
                    f"Impact estimé: {impact:.2%}",
                    f"Événements majeurs: {major_events}",
                ]
            )
        except Exception as e:
            report.append(f"\n📰 Erreur sur analyse news : {e}")
    else:
        report.append("\n📰 Analyse des News: Aucune donnée disponible.")

    # PATCH: Sécurise l'accès aux news pour éviter "list index out of range"
    major_news = news_sentiment.get("latest_news", []) if news_sentiment else []
    if major_news and isinstance(major_news, list) and len(major_news) > 0:
        report.append("Dernières news :")
        for news in major_news[:3]:
            report.append(f"- {news}")
    else:
        report.append("Aucune news disponible.")

    for timeframe, analysis in indicators_analysis.items():
        try:
            report.append(f"\n⏰ {timeframe}:")
            trend_strength = float(
                analysis.get("trend", {}).get("trend_strength", 0) or 0
            )
            volatility = float(
                analysis.get("volatility", {}).get("current_volatility", 0) or 0
            )
            volume_profile = analysis.get("volume", {}).get("volume_profile", {})
            # Cohérence volume (float ou dict)
            if isinstance(volume_profile, dict):
                volume_strength = volume_profile.get("strength", "N/A")
            else:
                volume_strength = volume_profile
            report.extend(
                [
                    f"- Force de la tendance: {trend_strength:.2%}",
                    f"- Volatilité: {volatility:.2%}",
                    f"- Volume: {volume_strength}",
                    f"- Signal dominant: {analysis.get('dominant_signal', 'Neutre')}",
                ]
            )
            if trade_decisions and timeframe in trade_decisions:
                dec = trade_decisions[timeframe]
                try:
                    confidence = float(dec.get("confidence", 0))
                    tech = float(dec.get("tech", 0))
                    ia = float(dec.get("ai", 0))
                    sentiment_trade = float(dec.get("sentiment", 0))
                except Exception:
                    confidence = tech = ia = sentiment_trade = 0.0
                report.append(
                    f"└─ 🎯 Décision de trade: {dec['action'].upper()} "
                    f"(Conf: {confidence:.2f}, "
                    f"Tech: {tech:.2f}, "
                    f"IA: {ia:.2f}, "
                    f"Sentiment: {sentiment_trade:.2f})"
                )
        except Exception as e:
            report.extend(
                [
                    f"\n⏰ {timeframe}:",
                    "- Données non disponibles",
                    "- Analyse en cours...",
                ]
            )
    return "\n".join(report)


def fetch_binance_ohlcv(
    symbol, interval, start_str, end_str=None, api_key=None, api_secret=None
):
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    if not klines or len(klines) == 0:
        print(f"Aucune donnée récupérée pour {symbol}")
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
    # 🔷 Tri systématique par timestamp après chargement OHLCV
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtest", action="store_true", help="Lancer un backtest quantitatif"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/historical/BTCUSDT_1h.csv",
        help="Chemin du CSV market data",
    )
    parser.add_argument("--capital", type=float, default=0, help="Capital initial")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sma",
        choices=["sma", "breakout", "arbitrage"],
        help="Stratégie à utiliser",
    )
    # Ajoute ici d'autres paramètres si besoin (fast_window, slow_window, lookback, etc.)

    args = parser.parse_args()

    if args.backtest:
        log_dashboard("=== Lancement du backtesting quantitatif ===")
        df = pd.read_csv(args.data)
        # 🔷 Tri par timestamp pour cohérence des indicateurs
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Choix de la stratégie
        strategy_map = {
            "sma": sma_strategy,
            "breakout": breakout_strategy,
            "arbitrage": arbitrage_strategy,
        }
        strategy_func = strategy_map[args.strategy]

        # Exemple : utilise des paramètres par défaut, ou récupère-les via argparse
        results = BacktestEngine(initial_capital=args.capital).run_backtest(
            df, strategy_func
        )
        log_dashboard("Résultats backtest :")
        print(results)
        exit(0)


def debug_market_data_structure(market_data, pairs_valid, timeframes):
    for pair in pairs_valid:
        pair_key = pair.replace("/", "").upper()
        if pair_key not in market_data:
            # print(f"  ❌ ABSENT de market_data")
            continue
        for tf in timeframes:
            tf_data = market_data[pair_key].get(tf)
            if tf_data is None:
                # print(f"  - {tf}: ❌ ABSENT")
                pass
            elif isinstance(tf_data, dict):
                # print(f"  - {tf}: OK, keys: {list(tf_data.keys())}")
                pass
            else:
                # print(f"  - {tf}: Type inattendu: {type(tf_data)}")
                pass


# Charger les tokens Telegram depuis .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️ Attention: Variables Telegram non trouvées dans .env")

# Configuration ULTRA-stricte pour éliminer TOUS les warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["STREAMLIT_HIDE_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configuration logging pour ne montrer que nos messages
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Constantes de marché
MARKET_REGIMES = {
    "TRENDING_UP": "Tendance Haussière",
    "TRENDING_DOWN": "Tendance Baissière",
    "RANGING": "Range/Scalping",
    "VOLATILE": "Haute Volatilité",
}


def get_current_time():
    tahiti = pytz.timezone("Pacific/Tahiti")
    now_tahiti = datetime.now(tahiti)
    return now_tahiti.strftime("%Y-%m-%d %H:%M:%S")


def get_current_time_tahiti():
    tahiti = pytz.timezone("Pacific/Tahiti")
    now_tahiti = datetime.now(tahiti)
    return now_tahiti.strftime("%Y-%m-%d %H:%M:%S")


# Constantes
CURRENT_TIME = get_current_time_tahiti()
CURRENT_USER = "Patmoorea"
CONFIG_PATH = "config/trading_pairs.json"
SHARED_DATA_PATH = "src/shared_data.json"


def safe(val, default="N/A", fmt=None):
    """Sécurise l'affichage d'une valeur (None => défaut, format optionnel)"""
    try:
        if val is None:
            return default
        if fmt:
            return fmt.format(val)
        return val
    except Exception:
        return default


class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        if not bot_token or not chat_id:
            print("⚠️ Configuration Telegram incomplète")
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.N_FEATURES = 8
        self.N_STEPS = 63

    def get_input_dim(self):
        return self.N_FEATURES * self.N_STEPS * len(getattr(self, "pairs_valid", []))

    async def send_message(self, message):
        """Envoie un message sur Telegram via une queue non bloquante"""
        if not self.bot_token or not self.chat_id:
            print("⚠️ Message non envoyé: Configuration Telegram manquante")
            return

        header = (
            f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"👤 Patmoorea\n"
            "------------------------\n"
        )
        full_message = header + message

        MAX_TELEGRAM_LENGTH = 4000
        if len(full_message) > MAX_TELEGRAM_LENGTH:
            full_message = (
                full_message[: MAX_TELEGRAM_LENGTH - 20]
                + "\n... (troncature automatique)"
            )

        # utilisation d'une file asyncio pour déléguer l'envoi
        if not hasattr(self, "_queue"):
            self._queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._telegram_worker())

        await self._queue.put(full_message)

    async def _telegram_worker(self):
        """Worker qui envoie les messages Telegram en arrière-plan"""
        url = f"{self.base_url}/sendMessage"
        TIMEOUT = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            while True:
                msg = await self._queue.get()
                data = {"chat_id": self.chat_id, "text": msg, "parse_mode": "HTML"}
                try:
                    async with session.post(url, json=data) as response:
                        result = await response.json()
                        if not result.get("ok"):
                            print(f"⚠️ Erreur Telegram: {result.get('description')}")
                except Exception as e:
                    print(f"⚠️ Erreur envoi Telegram: {e}")
                    self._log_to_file(msg)
                finally:
                    self._queue.task_done()

    def _log_to_file(self, message):
        """Fallback: log le message dans un fichier local si l'envoi échoue"""
        try:
            with open("telegram_fallback.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.utcnow()}] {message}\n")
        except Exception as e:
            print(f"⚠️ Impossible d'écrire dans telegram_fallback.log: {e}")

    async def send_performance_update(self, performance_data):
        message = (
            "🤖 <b>Trading Bot Status Update</b>\n\n"
            f"💰 Balance: ${performance_data.get('balance')}\n"
            f"📊 Win Rate: {performance_data.get('win_rate', 0)*100:.1f}%\n"
            f"📈 Profit Factor: {performance_data.get('profit_factor')}\n"
            f"🔄 Total Trades: {performance_data.get('total_trades','N/A')}\n"
        )
        await self.send_message(message)

    async def send_cycle_update(self, cycle, regime, duration):
        message = (
            "🔄 <b>Cycle Update</b>\n\n"
            f"📊 Cycle: {cycle}\n"
            f"🎯 Régime: {regime}\n"
            f"⏱️ Durée: {duration:.1f}s\n"
        )
        await self.send_message(message)

    async def send_trade_alert(self, trade_data):
        emoji = (
            "🟢"
            if trade_data.get("side", "").upper() == "BUY"
            else "🔴" if trade_data.get("side", "").upper() == "SELL" else "⚪️"
        )
        message = (
            f"{emoji} <b>TRADE EXÉCUTÉ</b>\n\n"
            f"📊 Paire : {trade_data.get('symbol','?')}\n"
            f"Action : <b>{trade_data.get('side','?')}</b>\n"
            f"Montant : {trade_data.get('amount','?')}\n"
            f"Prix : {trade_data.get('price','?')}\n"
            f"Total : {trade_data.get('total', 'N/A')}\n"
            f"Confiance : {trade_data.get('confidence', 'N/A')}\n"
            f"Signaux : Tech {trade_data.get('tech', 'N/A')} | IA {trade_data.get('ia', 'N/A')} | Sentiment {trade_data.get('sentiment', 'N/A')}\n"
            f"Raison : {trade_data.get('reason', 'Signal de trading')}\n"
        )
        await self.send_message(message)

    async def send_arbitrage_alert(self, opportunity):
        message = (
            f"🔄 <b>Opportunité d'Arbitrage</b>\n\n"
            f"📊 Paire: {opportunity['pair']}\n"
            f"💹 Différence: {opportunity['diff_percent']:.2f}%\n"
            f"📈 {opportunity['exchange1']}: {opportunity['price1']}\n"
            f"📉 {opportunity['exchange2']}: {opportunity['price2']}\n"
            f"💰 Profit potentiel: {(opportunity['diff_percent'] - 0.2):.2f}% (après frais)"
        )
        await self.send_message(message)

    async def send_news_summary(
        self,
        news_data,
        market_data=None,
        ai_summary: str = None,
        filter_symbols=None,
        filter_volatility=None,
    ):
        if not news_data or len(news_data) == 0:
            await self.send_message(
                "📰 <b>Dernières Nouvelles Importantes</b>\n\nAucune news significative détectée récemment."
            )
            return

        source_emoji = {
            "CoinDesk": "📰",
            "Cointelegraph": "🟣",
            "Decrypt": "🟦",
            "Binance": "🟡",
            "Twitter": "🐦",
            "default": "🗞️",
        }

        filtered_news = []
        for news in news_data:
            # Filtrage par symbole
            if filter_symbols:
                news_symbols = [s.upper() for s in news.get("symbols", [])]
                if not any(sym in news_symbols for sym in filter_symbols):
                    continue
            # Filtrage par volatilité
            if filter_volatility and market_data and news.get("symbols"):
                symbol = news["symbols"][0].replace("/", "")
                vol = market_data.get(symbol, {}).get("1h", {}).get("volatility", 0)
                if vol is not None and vol < filter_volatility:
                    continue
            filtered_news.append(news)

        if not filtered_news:
            filtered_news = news_data

        message = "📰 <b>Dernières Nouvelles Importantes</b>\n\n"
        if ai_summary:
            message += f"🤖 <b>Résumé IA:</b>\n{ai_summary}\n\n"

        def real_translate_title(title):
            try:
                return GoogleTranslator(source="auto", target="fr").translate(title)
            except Exception:
                return title

        for news in filtered_news:
            src = news.get("source", "default")
            emoji = source_emoji.get(src, source_emoji["default"])
            title = news.get("title", "NO_TITLE")
            url = news.get("url", "")
            fr_title = real_translate_title(title)
            if url:
                title_line = f'{emoji} <a href="{url}">{fr_title}</a>'
            else:
                title_line = f"{emoji} {fr_title}"
            title_line += f" <i>({src})</i>\n"
            message += title_line

        await self.send_message(message)


class WarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        # motifs à ignorer (lowercase)
        self.noisy = (
            "scriptruncontext",
            "streamlit",
            "not available",
            "skipping",
            "deprecated",
        )

    def write(self, message):
        try:
            m = message.lower()
            # si le message correspond à du "bruit", on l'ignore
            if any(pat in m for pat in self.noisy):
                return
            # sinon forward normal
            self.original_stderr.write(message)
        except Exception:
            # fallback safe: write original
            try:
                self.original_stderr.write(message)
            except Exception:
                pass

    def flush(self):
        try:
            self.original_stderr.flush()
        except Exception:
            pass


# n'affecte sys.stderr qu'une fois
sys.stderr = WarningFilter(sys.stderr)


def get_sentiment_summary_from_batch(sentiment_scores, top_n=5):

    # Filtre les news avec score
    valid = [
        item
        for item in sentiment_scores
        if "sentiment" in item and item["sentiment"] is not None
    ]
    if not valid:
        return {
            "sentiment_global": 0.0,
            "n_news": 0,
            "top_symbols": [],
            "top_news": [],
        }
    # Calcul de la moyenne pondérée
    sentiments = [safe_float(item["sentiment"], 0.0) for item in valid]
    sentiment_global = float(np.mean(sentiments)) if sentiments else 0.0

    # Top news (par score absolu)
    top_news = sorted(
        valid, key=lambda x: abs(safe_float(x["sentiment"], 0.0)), reverse=True
    )[:top_n]
    top_news_titles = [news["title"] for news in top_news if "title" in news]

    # Top symbols (fréquence + score fort)
    symbol_scores = {}
    for item in valid:
        for s in item.get("symbols", []):
            symbol_scores.setdefault(s, []).append(safe_float(item["sentiment"], 0.0))
    top_symbols = sorted(
        symbol_scores.items(),
        key=lambda kv: abs(np.mean(kv[1])) if kv[1] else 0.0,
        reverse=True,
    )
    top_symbols = [s for s, scores in top_symbols[:top_n]]
    return {
        "sentiment_global": sentiment_global,
        "n_news": len(valid),
        "top_symbols": top_symbols,
        "top_news": top_news_titles,
    }


def merge_news_processed(old_scores, new_scores):
    """Merge les news existantes (ayant 'processed') avec les nouvelles, en préservant ce flag par titre."""
    old_map = {
        n.get("title"): n.get("processed", False) for n in old_scores if "title" in n
    }
    for n in new_scores:
        if n.get("title") in old_map:
            n["processed"] = old_map[n.get("title")]
    return new_scores


class APIRequestOptimizer:
    """Gestionnaire optimisé des requêtes API"""

    def __init__(self):
        self.rate_limits = {}
        self.cache = TTLCache(maxsize=100, ttl=60)
        self.backup_endpoints = []

    async def execute_with_retry(self, request_func, max_retries=3):
        for i in range(max_retries):
            try:
                return await request_func()
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                await asyncio.sleep(2**i)


class DataBackupManager:
    """Gestionnaire de sauvegarde des données"""

    def __init__(self, backup_dir="backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def backup_trade_data(self, data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_data_{timestamp}.parquet"
        self.save_parquet(data, os.path.join(self.backup_dir, filename))


class PerformanceMonitor:
    """Système avancé de monitoring des performances"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []

    def track_metric(self, name, value):
        self.metrics[name].append({"timestamp": datetime.now(), "value": value})
        self.check_alerts(name, value)

    def check_alerts(self, metric_name, value):
        """Vérifie les déviations de performance"""
        if metric_name == "win_rate" and value < 0.55:
            self.add_alert("Win rate below threshold", severity="high")
        elif metric_name == "drawdown" and value < -0.15:
            self.add_alert("Excessive drawdown", severity="critical")


class RiskManager:
    def __init__(self):
        self.max_drawdown_limit = 0.15
        self.position_limits = {"max_per_trade": 0.05, "max_total_exposure": 0.25}
        self.min_confidence = 0.8
        self.validation_thresholds = {
            "technical": 0.3,
            "momentum": 0.2,
            "orderflow": 0.2,
            "liquidity": 0.7,
            "pressure": 0.8,
        }

    # Dans RiskManager

    def validate_trade(self, signals):
        """Valide si un trade respecte les critères de risque"""
        try:
            if not signals or not isinstance(signals, dict):
                print("[RISK] Signaux invalides")
                return False

            # Extraction des composantes
            technical = signals.get("technical", {})
            momentum = signals.get("momentum", {})
            orderflow = signals.get("orderflow", {})

            # Validation technique
            tech_score = float(technical.get("score", 0))
            if abs(tech_score) < self.validation_thresholds["technical"]:
                print(f"[RISK] Score technique insuffisant: {tech_score:.2f}")
                return False

            # Validation momentum
            mom_score = float(momentum.get("score", 0))
            if abs(mom_score) < self.validation_thresholds["momentum"]:
                print(f"[RISK] Momentum insuffisant: {mom_score:.2f}")
                return False

            # Validation orderflow
            flow_score = float(orderflow.get("score", 0))
            if abs(flow_score) < self.validation_thresholds["orderflow"]:
                print(f"[RISK] Orderflow insuffisant: {flow_score:.2f}")
                return False

            # Score global
            total_score = (tech_score + mom_score + flow_score) / 3
            if abs(total_score) < 0.25:
                print(f"[RISK] Score global insuffisant: {total_score:.2f}")
                return False

            print(f"[RISK] ✅ Trade validé - Score global: {total_score:.2f}")
            return True

        except Exception as e:
            print(f"[RISK] Erreur validation: {e}")
            return False


class TradingBotM4:
    def __init__(self):
        # Configuration de base existante...
        self.config = {
            "TRADING": {
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "pairs": [
                    "BTC/USDC",
                    "ETH/USDC",
                    "LTC/USDC",
                    "XRP/USDC",
                    "DOGE/USDC",
                    "BNB/USDC",
                    "ADA/USDC",
                    "SOL/USDC",
                    "TRX/USDC",
                    "SUI/USDC",
                    "LINK/USDC",
                ],
            },
            "AI": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "n_epochs": 10,
                "verbose": 1,
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
        # Initialisation explicite du risk manager
        try:
            self.risk_manager = RiskManager()
            if hasattr(self.risk_manager, "validate_trade"):
                print("✅ Risk Manager initialisé avec succès")
            else:
                print("❌ Risk Manager mal initialisé")
        except Exception as e:
            print(f"❌ Erreur initialisation risk manager: {e}")
            # Fallback sur un gestionnaire minimal
            self.risk_manager = type(
                "DummyRiskManager",
                (),
                {
                    "validate_trade": lambda x: True,
                    "calculate_position_size": lambda *args: 0.0,
                },
            )()

        if hasattr(self.risk_manager, "validate_trade"):
            print("✅ Risk Manager initialisé avec succès")
        else:
            print("❌ Risk Manager mal initialisé")
        print("✅ Risk Manager initialisé")

        self.data_file = SHARED_DATA_PATH

        self.last_correlation_check = 0
        self.correlation_cache = {}
        self.correlation_cache_ttl = 300  # 5 minutes

        self.system_metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "api_latency": [],
            "ws_status": True,
        }

        self.news_pause_manager = NewsPauseManager(default_pause_cycles=6)
        try:
            shared_data = safe_load_shared_data(self.data_file)
            deep_cast_floats(shared_data)
            # PATCH: Restore ONLY from active_pauses, ignore pause_status
            active_pauses = shared_data.get("active_pauses", [])
            self.news_pause_manager.reset_pauses(active_pauses)
        except Exception as e:
            print(f"[INIT PAUSE] Impossible de restaurer l’état des pauses : {e}")

        self.exit_manager = ExitManager(
            tp_levels=[(0.03, 0.3), (0.07, 0.3)], trailing_pct=0.03
        )

        self.trade_decisions = {}

        self.signal_fusion_params = self.load_signal_fusion_params()

        self.positions = {}  # Ajouté : gestion des positions spot par paire
        self.stop_loss_pct = 0.03  # 3% stop-loss, modifiable

        bingx_api_key = os.getenv("BINGX_API_KEY")
        bingx_api_secret = os.getenv("BINGX_API_SECRET")

        self.bingx_client = BingXExchange(
            bingx_api_key, bingx_api_secret
        )  # adapte selon ton code
        self.bingx_executor = BingXOrderExecutor(self.bingx_client)

        # --- SYNCHRONISATION AUTO DES PAIRS ---
        self.pairs_valid = self.config["TRADING"]["pairs"]

        # --- WS COLLECTOR --- (toujours synchro avec la config)
        self.ws_collector = BufferedWSCollector(
            symbols=[s.replace("/", "").upper() for s in self.pairs_valid],
            timeframes=self.config["TRADING"]["timeframes"],
            maxlen=2000,
        )
        # Initialize basic attributes...

        self.current_cycle = 0
        self.regime = MARKET_REGIMES["RANGING"]
        self.market_data = {}
        self.indicators = {}
        self.news_analyzer = NewsSentimentAnalyzer(self.config)
        self.news_enabled = True
        self.dl_model_last_mtime = None

        self.news_weight = 0.15
        self.ai_weight = 0.5
        self.ensure_float = lambda x: (
            float(x) if isinstance(x, (int, float, str)) else 0.0
        )
        self.technical_weight = 0.6  # Poids des signaux techniques (60%)
        self.ai_enabled = False
        self.pairs_valid = self.config["TRADING"]["pairs"]

        # Initialisation de l'arbitrage engine
        try:
            self.arbitrage_engine = ArbitrageEngine()
            self.brokers = self.arbitrage_engine.brokers
            log_dashboard("✅ ArbitrageEngine initialisé avec succès")
        except Exception as e:
            log_dashboard(f"⚠️ Erreur initialisation ArbitrageEngine: {e}")
            self.arbitrage_engine = None
            self.brokers = {}

        self.arbitrage_executor = ArbitrageExecutor(self.brokers)

        # Initialisation de l'environnement (une seule fois)
        print("Configuration de l'environnement...")

        # --- ENVIRONNEMENT TRADING ---
        self.env = TradingEnv(
            trading_pairs=self.pairs_valid,
            timeframes=self.config["TRADING"]["timeframes"],
        )
        print("✅ Environnement initialisé avec succès")

        # Initialisation de l'IA (modèle réel uniquement)
        self._initialize_ai()

        # Initialise les données partagées
        self.initialize_shared_data()

        print(f"Trading pairs: {self.pairs_valid}")
        print(f"Environment initialized: {hasattr(self, 'env')}")
        if hasattr(self, "env"):
            print(
                f"Environment methods: reset={hasattr(self.env, 'reset')}, step={hasattr(self.env, 'step')}"
            )
        # Initialisation des composants d'arbitrage
        self.arbitrage_bot = ArbitrageBot()
        self.arbitrage_scanner = ArbitrageScanner()

        # Configuration de l'arbitrage
        self.arbitrage_config = {
            "min_profit": 0.5,
            "max_exposure": 1000,
            "enabled_exchanges": ["binance", "kucoin", "huobi"],
        }
        # Sécurité avancée: gestion de clé cold wallet
        # Ajoute cette option (True = utilisation automatique, False = ignorée)
        self.use_cold_wallet_key = False  # ou False selon besoin

        self.key_manager = KeyManager()
        if self.use_cold_wallet_key:
            if not self.key_manager.has_key():
                print(
                    "Aucune clé cold wallet détectée, génération d'une nouvelle clé sécurisée…"
                )
                pk = self.key_manager.generate_private_key()
                self.key_manager.save_private_key()
                print("Clé cold wallet générée et sauvegardée de manière chiffrée.")
            else:
                try:
                    # Si tu veux demander le mot de passe à chaque fois (optionnel):
                    # password = self.ask_wallet_password()
                    # self.key_manager.load_private_key(password=password)
                    self.key_manager.load_private_key()
                    print("Clé cold wallet chargée avec succès.")
                except Exception as e:
                    print(f"Erreur de chargement de la clé cold wallet: {e}")
        else:
            print("⚠️ Utilisation de la clé cold wallet désactivée.")

        self.auto_strategy_config = None
        if os.path.exists("config/auto_strategy.json"):
            with open("config/auto_strategy.json", "r") as f:
                self.auto_strategy_config = json.load(f)
            log_dashboard("✅ Auto-stratégie chargée :", self.auto_strategy_config)
        self.positions_binance = {}
        self.sync_positions_with_binance()
        self.refused_trades_cycle = []  # PATCH: accumule les refus "Achat REFUSÉ"

    def _preserve_and_update_dashboard(self, new_fields):
        if os.path.exists(self.data_file):
            existing_data = safe_load_shared_data(self.data_file)
        else:
            existing_data = {}
        preserved_fields = [
            "trade_history",
            "closed_positions",
            "equity_history",
            "news_data",
            "sentiment",
            "active_pauses",
            "pending_sales",
            "positions_binance",
            "market_data",
        ]
        for field in preserved_fields:
            if field in existing_data and field not in new_fields:
                new_fields[field] = existing_data[field]
        self.safe_update_shared_data(new_fields, self.data_file)

    def calculate_position_size(self, equity, confidence, volatility=0.02):
        """Calcul intelligent de la taille de position"""
        try:
            if confidence < self.min_confidence:
                print(f"[RISK] Confiance insuffisante: {confidence:.2f}")
                return 0

            base_size = float(equity) * self.position_limits["max_per_trade"]
            vol_adj = max(0.3, 1 - (volatility * 2))
            final_size = min(
                base_size * vol_adj,
                float(equity) * self.position_limits["max_per_trade"],
            )

            print(f"[RISK] Taille position: {final_size:.2f} USDC")
            return final_size

        except Exception as e:
            print(f"[RISK] Erreur calcul position: {str(e)}")
            return 0

    def check_exposure_limit(self, current_positions, new_position_size):
        """Vérifie les limites d'exposition"""
        try:
            total_exposure = sum(
                float(pos.get("size", 0)) for pos in current_positions.values()
            )
            new_total = total_exposure + float(new_position_size)
            is_valid = new_total <= self.position_limits["max_total_exposure"]

            print(f"[RISK] Exposition totale: {new_total:.2f USDC}")
            return is_valid

        except Exception as e:
            print(f"[RISK] Erreur vérification exposition: {e}")
            return False

    async def detect_and_buy_news_impulsif(
        self, news_item, min_sentiment=0.7, confirmation_threshold=0.5
    ):
        """
        Détecte une news forte sur n'importe quelle crypto USDC, calcule tous les indicateurs du bot,
        confirme l'achat, l'exécute, notifie Telegram et met à jour le dashboard.
        Mais NE SPAM PAS Telegram sur chaque refus.
        """
        symbol_list = news_item.get("symbols", [])
        sentiment = float(news_item.get("sentiment", 0))
        title = news_item.get("title", "News sans titre")
        reason = []
        bilan = ""
        # 1. Ne traiter que les news fortes
        if sentiment < min_sentiment or not symbol_list:
            return  # Rien à faire

        for symbol in symbol_list:
            pair = f"{symbol}/USDC"
            pair_key = pair.replace("/", "").upper()
            # 2. Vérifie si la paire existe sur Binance (spot USDC)
            try:
                ticker = self.binance_client.get_symbol_ticker(
                    symbol=pair.replace("/", "")
                )
                price_binance = float(ticker.get("price", 0))
                if price_binance == 0:
                    reason.append("Paire indisponible sur Binance USDC")
                    bilan = f"Achat REFUSÉ sur {pair}: " + ", ".join(reason)
                    # PATCH: Ajoute à la liste de refus, NE PAS envoyer Telegram
                    self.refused_trades_cycle.append(bilan)
                    self._preserve_and_update_dashboard({"news_bilan": [bilan]})
                    continue
            except Exception as e:
                reason.append(f"Erreur accès Binance: {e}")
                bilan = f"Achat REFUSÉ sur {pair}: " + ", ".join(reason)
                self.refused_trades_cycle.append(bilan)
                self._preserve_and_update_dashboard({"news_bilan": [bilan]})
                continue

            # 3. Récupère dynamiquement les données OHLCV (1h) pour la paire
            try:
                klines = self.binance_client.get_klines(
                    symbol=pair.replace("/", ""), interval="1h", limit=50
                )
                if not klines or len(klines) == 0:
                    reason.append("Pas de données OHLCV")
                    bilan = f"Achat REFUSÉ sur {pair}: " + ", ".join(reason)
                    self.refused_trades_cycle.append(bilan)
                    self._preserve_and_update_dashboard({"news_bilan": [bilan]})
                    continue
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
                        "qav",
                        "trades",
                        "tbbav",
                        "tbqav",
                        "ignore",
                    ],
                )
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df[["open", "high", "low", "close", "volume"]] = df[
                    ["open", "high", "low", "close", "volume"]
                ].astype(float)
            except Exception as e:
                reason.append(f"Erreur OHLCV: {e}")
                bilan = f"Achat REFUSÉ sur {pair}: " + ", ".join(reason)
                self.refused_trades_cycle.append(bilan)
                self._preserve_and_update_dashboard({"news_bilan": [bilan]})
                continue

            # 4. Calcule tous les indicateurs du bot (technique, IA, momentum, orderflow, sentiment)
            try:
                # Technique
                indics = self.add_indicators(df)
                tech_score = float(indics.get("technical_score", 0.5))
                # Momentum
                momentum_score = (
                    float(indics.get("momentum", 0.5)) if "momentum" in indics else 0.5
                )
                # Orderflow
                of = self.analyze_order_flow(df)
                orderflow_score = float(of.get("imbalance", 0.5))
                # IA : si dispo, calcule sur les features du df
                ai_score = 0.5
                if hasattr(self, "dl_model") and self.dl_model is not None:
                    features = await self._prepare_features_for_ai(pair_key)
                    try:
                        if features:
                            ai_score = float(self.dl_model.predict(features))
                    except Exception:
                        ai_score = 0.5
                # Sentiment (celui de la news)
                sentiment_score = sentiment

                # Confirmation : tous les scores doivent dépasser le seuil
                confirm = all(
                    [
                        tech_score > confirmation_threshold,
                        ai_score > confirmation_threshold,
                        momentum_score > confirmation_threshold,
                        orderflow_score > confirmation_threshold,
                        sentiment_score > min_sentiment,
                    ]
                )
                reason.append(f"Sentiment news fort ({sentiment_score:.2f})")
                reason.append(f"Technique: {tech_score:.2f}")
                reason.append(f"Momentum: {momentum_score:.2f}")
                reason.append(f"Orderflow: {orderflow_score:.2f}")
                reason.append(f"IA: {ai_score:.2f}")
            except Exception as e:
                reason.append(f"Erreur indicateurs: {e}")
                confirm = False

            # 5. Achat et log
            if confirm:
                amount = 15  # montant fixe ou à calculer selon ta logique
                try:
                    result = await self.execute_trade(
                        pair, "BUY", amount, price_binance
                    )
                    bilan = f"Achat EFFECTUÉ sur {pair}: " + ", ".join(reason)
                    await self.telegram.send_message(bilan)
                except Exception as e:
                    bilan = (
                        f"Achat REFUSÉ sur {pair}: Erreur exécution trade: {e}, "
                        + ", ".join(reason)
                    )
                    self.refused_trades_cycle.append(bilan)
            else:
                bilan = f"Achat REFUSÉ sur {pair}: " + ", ".join(reason)
                self.refused_trades_cycle.append(bilan)

            # 6. Dashboard uniquement (PATCH CORRECTIF)
            self._preserve_and_update_dashboard({"news_bilan": [bilan]})

    def auto_update_pairs_from_binance(self):
        """
        Met à jour self.pairs_valid en ajoutant automatiquement les paires USDC
        disponibles sur Binance (où les achats sont possibles).
        À appeler à l'initialisation du bot ou quand tu veux rafraîchir la liste.
        """
        available_pairs = []
        assets = [
            "BTC",
            "ETH",
            "LTC",
            "XRP",
            "DOGE",
            "BNB",
            "ADA",
            "SOL",
            "TRX",
            "SUI",
            "LINK",
            # Ajoute ici d'autres assets à tester si besoin
        ]
        for asset in assets:
            symbol = f"{asset}/USDC"
            try:
                ticker = self.binance_client.get_symbol_ticker(
                    symbol=symbol.replace("/", "")
                )
                if ticker and float(ticker.get("price", 0)) > 0:
                    available_pairs.append(symbol)
            except Exception:
                continue
        self.pairs_valid = available_pairs
        print(f"[PAIR AUTO-UPDATE] Paires valides actualisées : {self.pairs_valid}")

    def detect_pump_candidates(self, min_pct=0.05, min_volume_ratio=2, tf="1h"):
        """
        Détecte les cryptos en pump (hausse brutale prix et volume) sur TOUTES les paires USDC spot Binance,
        même hors self.pairs_valid.
        Confirmation possible via indicateurs avant achat.
        """
        candidates = []
        # --- PATCH: Récupère TOUTES les paires USDC existantes sur Binance ---
        try:
            all_pairs = []
            exchange_info = self.binance_client.get_exchange_info()
            for symbol_info in exchange_info["symbols"]:
                if (
                    symbol_info["quoteAsset"] == "USDC"
                    and symbol_info["status"] == "TRADING"
                    and symbol_info["isSpotTradingAllowed"]
                ):
                    base = symbol_info["baseAsset"]
                    pair = f"{base}/USDC"
                    all_pairs.append(pair)
        except Exception as e:
            print(f"[PUMP] Erreur récupération exchange info: {e}")
            all_pairs = self.pairs_valid

        for pair in all_pairs:
            pair_key = pair.replace("/", "").upper()
            # PATCH: Ajout sécurité si market_data ne contient pas cette paire
            if pair_key in self.market_data and tf in self.market_data[pair_key]:
                data = self.market_data[pair_key][tf]
                closes = data.get("close", [])
                volumes = data.get("volume", [])
                if len(closes) > 25 and len(volumes) > 25:
                    price_pct = (closes[-1] - closes[-10]) / closes[-10]
                    avg_vol = np.mean(volumes[-24:-1])
                    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
                    # Confirmation par indicateurs avant ajout
                    confirm = False
                    reason = []
                    # PATCH: Confirmation par indicateurs
                    tech_score = (
                        data.get("signals", {}).get("technical", {}).get("score", 0.5)
                    )
                    ai_score = data.get("ai_prediction", 0.5)
                    if price_pct > min_pct and vol_ratio > min_volume_ratio:
                        if tech_score > 0.5 and ai_score > 0.5:
                            confirm = True
                            reason.append(f"Technique OK ({tech_score:.2f})")
                            reason.append(f"IA OK ({ai_score:.2f})")
                        else:
                            if tech_score <= 0.5:
                                reason.append(f"Technique faible ({tech_score:.2f})")
                            if ai_score <= 0.5:
                                reason.append(f"IA faible ({ai_score:.2f})")
                        reason.insert(
                            0,
                            f"Pump détecté: price_pct={price_pct:.2f}, vol_ratio={vol_ratio:.2f}",
                        )
                        candidate = {
                            "pair": pair,
                            "price_pct": price_pct,
                            "vol_ratio": vol_ratio,
                            "reason": ", ".join(reason),
                            "confirm": confirm,
                        }
                        candidates.append(candidate)
                        # PATCH: Dashboard/Telegram bilan
                        self.safe_update_shared_data(
                            {"pump_bilan": candidate}, self.data_file
                        )
                        if hasattr(self, "telegram"):
                            import asyncio

                            asyncio.create_task(
                                self.telegram.send_message(
                                    f"{'Achat EFFECTUÉ' if confirm else 'Achat REFUSÉ'} sur {pair}: {candidate['reason']}"
                                )
                            )
        return candidates

    def detect_breakout_candidates(self, tf="1h"):
        """
        Détection des cassures de résistance sur TOUTES les paires USDC spot Binance,
        même hors self.pairs_valid.
        Confirmation possible via indicateurs avant achat.
        """
        candidates = []
        # --- PATCH: Récupère TOUTES les paires USDC spot Binance ---
        try:
            all_pairs = []
            exchange_info = self.binance_client.get_exchange_info()
            for symbol_info in exchange_info["symbols"]:
                if (
                    symbol_info["quoteAsset"] == "USDC"
                    and symbol_info["status"] == "TRADING"
                    and symbol_info["isSpotTradingAllowed"]
                ):
                    base = symbol_info["baseAsset"]
                    pair = f"{base}/USDC"
                    all_pairs.append(pair)
        except Exception as e:
            print(f"[BREAKOUT] Erreur récupération exchange info: {e}")
            all_pairs = self.pairs_valid

        for pair in all_pairs:
            pair_key = pair.replace("/", "").upper()
            if pair_key in self.market_data and tf in self.market_data[pair_key]:
                data = self.market_data[pair_key][tf]
                highs = data.get("high", [])
                closes = data.get("close", [])
                if len(highs) > 21 and len(closes) > 1:
                    donchian_high = max(highs[-21:-1])
                    confirm = False
                    reason = []
                    tech_score = (
                        data.get("signals", {}).get("technical", {}).get("score", 0.5)
                    )
                    ai_score = data.get("ai_prediction", 0.5)
                    if closes[-1] > donchian_high:
                        if tech_score > 0.5 and ai_score > 0.5:
                            confirm = True
                            reason.append(f"Technique OK ({tech_score:.2f})")
                            reason.append(f"IA OK ({ai_score:.2f})")
                        else:
                            if tech_score <= 0.5:
                                reason.append(f"Technique faible ({tech_score:.2f})")
                            if ai_score <= 0.5:
                                reason.append(f"IA faible ({ai_score:.2f})")
                        reason.insert(
                            0,
                            f"Breakout détecté: close={closes[-1]:.2f} > {donchian_high:.2f}",
                        )
                        candidate = {
                            "pair": pair,
                            "breakout_level": donchian_high,
                            "close": closes[-1],
                            "reason": ", ".join(reason),
                            "confirm": confirm,
                        }
                        candidates.append(candidate)
                        self.safe_update_shared_data(
                            {"breakout_bilan": candidate}, self.data_file
                        )
                        if hasattr(self, "telegram"):
                            import asyncio

                            asyncio.create_task(
                                self.telegram.send_message(
                                    f"{'Achat EFFECTUÉ' if confirm else 'Achat REFUSÉ'} sur {pair}: {candidate['reason']}"
                                )
                            )
        return candidates

    def detect_news_candidates(self, news_list, min_sentiment=0.7):
        candidates = []
        bilan_list = []  # Pour historiser les bilans dans le dashboard
        for news in news_list:
            sentiment = safe_float(news.get("sentiment", 0))
            if sentiment > min_sentiment:
                for symbol in news.get("symbols", []):
                    pair = f"{symbol}/USDC"
                    confirm = False
                    reason = []
                    try:
                        # Vérifie que la paire existe sur Binance
                        ticker = self.binance_client.get_symbol_ticker(
                            symbol=pair.replace("/", "")
                        )
                        if ticker and float(ticker.get("price", 0)) > 0:
                            # Confirmation par autres indicateurs
                            pair_key = pair.replace("/", "").upper()
                            market_data = self.market_data.get(pair_key, {}).get(
                                "1h", {}
                            )
                            tech_score = (
                                market_data.get("signals", {})
                                .get("technical", {})
                                .get("score", 0.5)
                            )
                            ai_score = market_data.get("ai_prediction", 0.5)
                            # Ajoute la logique de confirmation
                            if tech_score > 0.5 and ai_score > 0.5:
                                confirm = True
                                reason.append(f"Technique OK ({tech_score:.2f})")
                                reason.append(f"IA OK ({ai_score:.2f})")
                            else:
                                if tech_score <= 0.5:
                                    reason.append(
                                        f"Technique faible ({tech_score:.2f})"
                                    )
                                if ai_score <= 0.5:
                                    reason.append(f"IA faible ({ai_score:.2f})")
                            reason.insert(0, f"Sentiment news fort ({sentiment:.2f})")
                        else:
                            reason.append("Paire non disponible sur Binance en USDC")
                    except Exception as e:
                        reason.append(f"Erreur accès Binance: {e}")
                    # Bilan et action
                    if confirm:
                        candidates.append(
                            {
                                "pair": pair,
                                "sentiment": sentiment,
                                "title": news.get("title", ""),
                                "reason": ", ".join(reason),
                            }
                        )
                        bilan = f"Achat EFFECTUÉ sur {pair}: " + ", ".join(reason)
                    else:
                        bilan = f"Achat REFUSÉ sur {pair}: " + ", ".join(reason)
                    bilan_list.append(bilan)
                    # Telegram et dashboard
                    self.safe_update_shared_data(
                        {"news_bilan": bilan_list}, self.data_file
                    )
                    asyncio.create_task(self.telegram.send_message(bilan))
        return candidates

    async def detect_arbitrage_candidates(self, min_diff_pct=0.5):
        """
        Arbitrage rapide entre Binance et BingX/OKX/Kucoin sur TOUTES les paires USDC spot Binance.
        Confirmation via indicateurs avant achat.
        """
        candidates = []
        if not self.is_live_trading or not self.binance_client:
            return []
        # --- PATCH: Récupère TOUTES les paires USDC spot Binance ---
        try:
            all_pairs = []
            exchange_info = self.binance_client.get_exchange_info()
            for symbol_info in exchange_info["symbols"]:
                if (
                    symbol_info["quoteAsset"] == "USDC"
                    and symbol_info["status"] == "TRADING"
                    and symbol_info["isSpotTradingAllowed"]
                ):
                    base = symbol_info["baseAsset"]
                    pair = f"{base}/USDC"
                    all_pairs.append(pair)
        except Exception as e:
            print(f"[ARBITRAGE] Erreur récupération exchange info: {e}")
            all_pairs = self.pairs_valid

        for pair in all_pairs:
            symbol_binance = pair.replace("/", "")
            try:
                binance_ticker = self.binance_client.get_ticker(symbol=symbol_binance)
                binance_price = float(binance_ticker.get("lastPrice"))
                # Ex : BingX
                symbol_bingx = symbol_binance.replace("USDC", "USDT") + ":USDT"
                bingx_ticker = await self.bingx_client.fetch_ticker(symbol_bingx)
                bingx_price = float(bingx_ticker.get("last"))
                diff_pct = (bingx_price - binance_price) / binance_price * 100
                confirm = False
                reason = []
                # PATCH: Confirmation via indicateurs
                pair_key = pair.replace("/", "").upper()
                tech_score = (
                    self.market_data.get(pair_key, {})
                    .get("signals", {})
                    .get("technical", {})
                    .get("score", 0.5)
                )
                ai_score = self.market_data.get(pair_key, {}).get("ai_prediction", 0.5)
                if abs(diff_pct) > min_diff_pct:
                    if tech_score > 0.5 and ai_score > 0.5:
                        confirm = True
                        reason.append(f"Technique OK ({tech_score:.2f})")
                        reason.append(f"IA OK ({ai_score:.2f})")
                    else:
                        if tech_score <= 0.5:
                            reason.append(f"Technique faible ({tech_score:.2f})")
                        if ai_score <= 0.5:
                            reason.append(f"IA faible ({ai_score:.2f})")
                    reason.insert(0, f"Arbitrage détecté: diff_pct={diff_pct:.2f}")
                    candidate = {
                        "pair": pair,
                        "binance_price": binance_price,
                        "bingx_price": bingx_price,
                        "diff_pct": diff_pct,
                        "reason": ", ".join(reason),
                        "confirm": confirm,
                    }
                    candidates.append(candidate)
                    self.safe_update_shared_data(
                        {"arbitrage_bilan": candidate}, self.data_file
                    )
                    if hasattr(self, "telegram"):
                        import asyncio

                        asyncio.create_task(
                            self.telegram.send_message(
                                f"{'Achat EFFECTUÉ' if confirm else 'Achat REFUSÉ'} sur {pair}: {candidate['reason']}"
                            )
                        )
            except Exception:
                continue
        return candidates

    async def send_refused_trades_summary(self):
        """Envoie un résumé unique des refus de trades impulsifs/news/pump/breakout"""
        if not self.refused_trades_cycle:
            return
        count = len(self.refused_trades_cycle)
        msg = (
            f"⚪️ <b>{count} trades impulsifs/news/pump/breakout refusés ce cycle</b>\n\n"
            + "\n".join([f"• {bilan}" for bilan in self.refused_trades_cycle[:10]])
        )
        await self.telegram.send_message(msg)
        self.refused_trades_cycle.clear()

    async def plan_auto_sell(
        self,
        symbol,
        entry_price,
        amount,
        tp_pct=0.03,
        sl_pct=0.03,
        max_cycles=2,
        reason="",
    ):
        """
        Planifie une vente automatique pour une position ouverte via signal pump/breakout/news/arbitrage.
        Enregistre la raison de l'achat pour le dashboard.
        """
        auto_sell_list = []
        try:
            shared_data = safe_load_shared_data(self.data_file)
            auto_sell_list = shared_data.get("auto_sell_positions", [])
        except Exception:
            auto_sell_list = []

        auto_sell_list.append(
            {
                "symbol": symbol,
                "entry_price": entry_price,
                "amount": amount,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "cycle_open": self.current_cycle,
                "max_cycles": max_cycles,
                "reason": reason,  # <--- AJOUT
            }
        )
        self.safe_update_shared_data(
            {"auto_sell_positions": auto_sell_list}, self.data_file
        )

    def validate_tp_levels(self, levels):
        """Valide et convertit les niveaux TP depuis la config"""
        if not levels:
            return [(0.03, 0.3), (0.07, 0.3)]  # Valeurs par défaut

        validated = []
        for level in levels:
            try:
                if isinstance(level, str):
                    # Gère les formats "0.03:0.3" ou "0.03,0.3"
                    parts = level.replace(":", ",").split(",")
                    pct = float(parts[0].strip())
                    frac = float(parts[1].strip()) if len(parts) > 1 else 0.3
                else:
                    pct = float(level[0])
                    frac = float(level[1]) if len(level) > 1 else 0.3
                validated.append((pct, frac))
            except Exception as e:
                print(f"⚠️ Niveau TP ignoré: {level} - {str(e)}")
        return validated or [(0.03, 0.3), (0.07, 0.3)]

    def safe_float_conversion(self, value, default=0.0):
        """Conversion robuste vers float"""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

    def fetch_trades_fifo(self, binance_client, symbol):
        """
        Récupère la liste des achats (buys) et ventes (sells) spot pour la paire donnée (ex: "BTCUSDC"),
        formatée pour le calcul FIFO.
        Retourne:
            buys: [{"qty":..., "price":..., "time":..., "id":...}, ...]
            sells: [{"qty":..., "price":..., "time":..., "id":...}, ...]
        """
        buys, sells = [], []
        try:
            # Appel correct de l'API Binance : paramètre passé en mot-clé !
            trades = binance_client.get_my_trades(symbol=symbol)
            for trade in trades:
                qty = float(trade["qty"])
                price = float(trade["price"])
                time = int(trade["time"])
                trade_id = trade.get("id", trade.get("orderId", None))
                trade_dict = {
                    "qty": qty,
                    "price": price,
                    "time": time,
                    "id": trade_id,
                }
                if trade["isBuyer"]:
                    buys.append(trade_dict)
                else:
                    sells.append(trade_dict)
            # Tri chronologique (optionnel mais conseillé)
            buys = sorted(buys, key=lambda x: x["time"])
            sells = sorted(sells, key=lambda x: x["time"])
            return buys, sells
        except Exception as e:
            print(f"[DEBUG FIFO] Erreur fetch_trades_fifo pour {symbol}: {e}")
            return [], []

    def get_last_fifo_pnl(self, symbol):
        """
        Récupère la plus-value FIFO de la dernière vente spot pour une paire donnée.
        """
        try:
            buys, sells = self.fetch_trades_fifo(
                self.binance_client, symbol.replace("/", "")
            )
            fifo_results = self.fifo_pnl(buys, sells)
            last_result = fifo_results[-1] if fifo_results else None
            return (
                last_result["pnl_pct"]
                if last_result and last_result["pnl_pct"] is not None
                else None
            )
        except Exception:
            return None

    def fetch_trades_fifo(self, binance_client, symbol):
        """
        Récupère la liste des achats (buys) et ventes (sells) spot pour la paire donnée (ex: "BTCUSDC"),
        formatée pour le calcul FIFO.
        Retourne:
            buys: [{"qty":..., "price":..., "time":..., "id":...}, ...]
            sells: [{"qty":..., "price":..., "time":..., "id":...}, ...]
        """
        buys, sells = [], []
        try:
            trades = binance_client.get_my_trades(symbol=symbol)
            for trade in trades:
                qty = safe_float(trade.get("qty", 0))
                price = safe_float(trade.get("price", 0))
                time_val = trade.get("time", 0)
                try:
                    time_val = int(time_val)
                except Exception:
                    time_val = 0
                trade_id = trade.get("id", trade.get("orderId", None))
                trade_dict = {
                    "qty": qty,
                    "price": price,
                    "time": time_val,
                    "id": trade_id,
                }
                if trade.get("isBuyer", False):
                    buys.append(trade_dict)
                else:
                    sells.append(trade_dict)
            buys = sorted(buys, key=lambda x: x["time"])
            sells = sorted(sells, key=lambda x: x["time"])
            return buys, sells
        except Exception as e:
            print(f"[DEBUG FIFO] Erreur fetch_trades_fifo pour {symbol}: {e}")
            return [], []

    def fifo_pnl(self, buys, sells):
        """
        Calcule la plus-value de chaque vente (FIFO).
        Associe chaque vente aux achats les plus anciens restants.
        Retourne une liste de dicts: vente, prix achat, prix vente, PnL.
        """
        buy_queue = [dict(buy) for buy in buys]  # Copie des dicts
        results = []
        for sell in sells:
            qty_to_sell = safe_float(sell.get("qty"), 0)
            total_cost = 0.0
            qty_used = 0.0
            buy_used = []
            while qty_to_sell > 0 and buy_queue:
                buy = buy_queue[0]
                available_qty = safe_float(buy.get("qty"), 0)
                take_qty = min(qty_to_sell, available_qty)
                total_cost += safe_float(take_qty, 0) * safe_float(buy.get("price"), 0)
                qty_used += safe_float(take_qty, 0)
                buy_used.append((take_qty, safe_float(buy.get("price"), 0)))
                qty_to_sell -= take_qty
                buy["qty"] = safe_float(buy.get("qty"), 0) - take_qty
                if buy["qty"] <= 0.0000001:
                    buy_queue.pop(0)
            if qty_used > 0:
                entry_price = safe_float(total_cost, 0) / safe_float(qty_used, 1)
                pnl_usd = (safe_float(sell.get("price"), 0) - entry_price) * qty_used
                pnl_pct = (
                    (
                        (safe_float(sell.get("price"), 0) - entry_price)
                        / entry_price
                        * 100
                    )
                    if entry_price
                    else 0
                )
            else:
                entry_price = None
                pnl_usd = None
                pnl_pct = None
            results.append(
                {
                    "sell_qty": qty_used,
                    "sell_price": safe_float(sell.get("price"), 0),
                    "entry_price": safe_float(entry_price),
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                    "buy_details": buy_used,
                    "sell_time": sell.get("time", 0),
                    "sell_id": sell.get("id"),
                }
            )
        return results

    def calc_sizing(confidence, tech, ai, sentiment, win_rate=0.55, profit_factor=1.7):
        # Sizing base selon confiance
        if confidence > 0.8:
            base = 0.09
        elif confidence > 0.6:
            base = 0.06
        elif confidence > 0.4:
            base = 0.04
        else:
            base = 0.02
        # Ajustements
        if tech > 0.7:
            base *= 1.2
        if ai > 0.7:
            base *= 1.1
        if abs(sentiment) > 0.7:
            base *= 0.8
        # Kelly Criterion
        kelly = kelly_criterion(win_rate, profit_factor)
        if kelly > 0:
            base *= 1 + min(kelly * 0.5, 0.5)
        return f"{min(base * 100, 12):.1f}%"

    def calculate_pair_correlation(self, pair1, pair2, window=20, tf="1h"):
        """
        Calcule la corrélation entre deux paires sur une fenêtre donnée.
        Utilise les données du timeframe spécifié (par défaut 1h).
        """
        try:
            # Obtenir les données des deux paires
            pair1_key = pair1.replace("/", "").upper()
            pair2_key = pair2.replace("/", "").upper()

            # Vérifier si les données existent
            if (
                pair1_key not in self.market_data
                or pair2_key not in self.market_data
                or tf not in self.market_data[pair1_key]
                or tf not in self.market_data[pair2_key]
            ):
                return 0.0  # Pas de corrélation par défaut si données manquantes

            # Récupérer les prix de clôture
            closes1 = self.market_data[pair1_key][tf].get("close", [])[-window:]
            closes2 = self.market_data[pair2_key][tf].get("close", [])[-window:]

            # Vérifier la longueur des données
            if len(closes1) < window or len(closes2) < window:
                return 0.0

            # Calculer les rendements
            returns1 = np.diff(np.log(closes1))
            returns2 = np.diff(np.log(closes2))

            # Calculer la corrélation
            correlation = np.corrcoef(returns1, returns2)[0, 1]

            # Gérer les valeurs NaN/inf
            if np.isnan(correlation) or np.isinf(correlation):
                return 0.0

            return float(correlation)

        except Exception as e:
            self.logger.error(f"Erreur calcul corrélation {pair1}-{pair2}: {e}")
            return 0.0  # Valeur par défaut en cas d'erreur

    def get_market_correlation(self, symbol):
        """Calcule la corrélation moyenne avec le marché global (BTC)"""
        try:
            if "BTC" in symbol.upper():
                return 1.0

            # Utilise BTC comme référence marché
            correlation = self.calculate_pair_correlation(symbol, "BTC/USDC")
            return abs(correlation)  # On prend la valeur absolue pour le sizing

        except Exception as e:
            self.logger.error(f"Erreur corrélation marché pour {symbol}: {e}")
            return 0.5  # Valeur moyenne par défaut

    def get_liquidity_score(self, symbol):
        """Calcule un score de liquidité entre 0 et 1"""
        try:
            # Récupère le volume moyen sur 24h
            symbol_key = symbol.replace("/", "").upper()
            if symbol_key in self.market_data and "1h" in self.market_data[symbol_key]:
                volumes = self.market_data[symbol_key]["1h"].get("volume", [])[-24:]
                if volumes:
                    avg_volume = np.mean(volumes)
                    # Normalisation entre 0 et 1 avec seuil min/max
                    return min(1.0, max(0.1, avg_volume / 1000000))
            return 0.5  # Score moyen par défaut

        except Exception as e:
            self.logger.error(f"Erreur calcul liquidité pour {symbol}: {e}")
            return 0.5

    def get_average_spread(self, symbol):
        """Calcule le spread moyen"""
        try:
            bid, ask = self.get_ws_orderbook(symbol.replace("/", ""))
            if bid and ask:
                return (ask - bid) / ((ask + bid) / 2)
            return 0.001  # Spread par défaut de 0.1%

        except Exception as e:
            self.logger.error(f"Erreur calcul spread pour {symbol}: {e}")
            return 0.001

    def get_key_levels(self, symbol):
        """Retourne les niveaux clés (support/résistance)"""
        try:
            symbol_key = symbol.replace("/", "").upper()
            if symbol_key in self.market_data and "1h" in self.market_data[symbol_key]:
                df = pd.DataFrame(self.market_data[symbol_key]["1h"])
                return self._identify_key_levels(df)
            return []

        except Exception as e:
            self.logger.error(f"Erreur niveaux clés pour {symbol}: {e}")
            return []

    def is_near_key_level(self, symbol, price=None, threshold=0.01):
        """Vérifie si le prix est proche d'un niveau clé"""
        try:
            key_levels = self.get_key_levels(symbol)
            if not price:
                symbol_key = symbol.replace("/", "").upper()
                if (
                    symbol_key in self.market_data
                    and "1h" in self.market_data[symbol_key]
                ):
                    closes = self.market_data[symbol_key]["1h"].get("close", [])
                    if closes:
                        price = closes[-1]

            if price and key_levels:
                for level in key_levels:
                    level_price = level["price"]
                    if abs(price - level_price) / price < threshold:
                        return True
            return False

        except Exception as e:
            self.logger.error(f"Erreur vérification niveaux clés pour {symbol}: {e}")
            return False

    def debug_signals_state(self, pair_key, tf):
        """Vérifie et affiche l'état détaillé des signaux"""
        try:
            if pair_key not in self.market_data:
                print(f"❌ Pas de market_data pour {pair_key}")
                return False

            data = self.market_data[pair_key].get(tf, {})
            if not data:
                print(f"❌ Pas de données {tf} pour {pair_key}")
                return False

            # Vérification OHLCV
            has_ohlcv = all(
                k in data for k in ["open", "high", "low", "close", "volume"]
            )

            # Création de la structure signals si manquante
            if "signals" not in data:
                data["signals"] = {
                    "technical": {"score": 0, "details": {}, "factors": 0},
                    "momentum": {"score": 0, "details": {}, "factors": 0},
                    "orderflow": {
                        "score": 0,
                        "details": {},
                        "factors": 0,
                        "liquidity": 0,
                        "market_pressure": 0,
                    },
                    "ai": self.market_data[pair_key].get("ai_prediction", 0),
                    "sentiment": self.market_data[pair_key].get("sentiment", 0.5),
                }

            # Vérification des indicateurs techniques
            if has_ohlcv:
                df = self.ws_collector.get_dataframe(pair_key, tf)
                if df is not None and not df.empty:
                    indicators = self.add_indicators(df)
                    if indicators:
                        data["signals"]["technical"].update(
                            {
                                "score": float(indicators.get("technical_score", 0)),
                                "details": indicators,
                                "factors": len(indicators),
                            }
                        )

            # Vérification technique
            has_technical = (
                "signals" in data
                and "technical" in data["signals"]
                and data["signals"]["technical"].get("score") is not None
            )

            # Vérification IA
            has_ai = (
                "signals" in data
                and "ai" in data["signals"]
                and data["signals"]["ai"] is not None
            )

            # Vérification sentiment
            has_sentiment = (
                "signals" in data
                and "sentiment" in data["signals"]
                and data["signals"]["sentiment"] is not None
            )

            print(f"\n📊 État des signaux {pair_key}-{tf}:")
            print(
                f"  OHLCV: {'✅' if has_ohlcv else '❌'} ({len(data.get('close', [])) if has_ohlcv else 0} points)"
            )
            print(f"  Technical: {'✅' if has_technical else '❌'}")
            print(f"  AI: {'✅' if has_ai else '❌'}")
            print(f"  Sentiment: {'✅' if has_sentiment else '❌'}")

            if not (has_technical and has_ai and has_sentiment):
                print("\n🔍 Diagnostic approfondi:")
                if not has_technical:
                    print("  - Signaux techniques manquants")
                if not has_ai:
                    print("  - Prédictions IA non calculées")
                if not has_sentiment:
                    print("  - Analyse sentiment non effectuée")

            return has_ohlcv

        except Exception as e:
            print(f"❌ Erreur debug {pair_key}-{tf}: {e}")
            return False

    def verify_signals_completeness(self):
        """Vérifie l'état complet des signaux pour toutes les paires"""
        print("\n=== VÉRIFICATION COMPLÈTE DES SIGNAUX ===")

        all_ok = True

        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            print(f"\n🔍 Vérification {pair_key}:")

            # Initialisation du market_data si nécessaire
            if pair_key not in self.market_data:
                self.market_data[pair_key] = {}

            # Initialisation du sentiment avec valeur par défaut si manquant
            if "sentiment" not in self.market_data[pair_key]:
                self.market_data[pair_key]["sentiment"] = 0.0
                self.market_data[pair_key]["sentiment_timestamp"] = time.time()
                print(f"  ℹ️ Sentiment initialisé à 0.0 (neutre)")
            else:
                print(f"  ✅ Sentiment: {self.market_data[pair_key]['sentiment']:.4f}")

            # Vérification IA
            has_ai = "ai_prediction" in self.market_data[pair_key]
            if has_ai:
                print(
                    f"  ✅ Prédiction IA: {self.market_data[pair_key]['ai_prediction']}"
                )
            else:
                print("  ❌ Prédiction IA manquante")
                all_ok = False

            # On considère maintenant qu'un sentiment à 0.0 est valide (neutre)
            has_sentiment = "sentiment" in self.market_data[pair_key]

            if not (has_ai and has_sentiment):
                print(f"⚠️ Signaux incomplets pour {pair_key}")
                all_ok = False
            else:
                print(f"✅ Tous les signaux OK pour {pair_key}")

        return all_ok

    async def analyze_signals(self, symbol, ohlcv_df, indicators, tf="1h"):
        """
        Analyse technique complète avec indicateurs avancés.
        Retourne une décision de trading avec scores et signaux détaillés.
        """
        try:  # Premier try
            print(f"\n=== ANALYSE SIGNAUX {symbol}-{tf} ===")

            # Initialisation des scores au début
            total_score = 0
            tech_score = 0
            tech_factors = 0
            momentum_score = 0
            momentum_factors = 0
            flow_score = 0
            flow_factors = 0

            # Fonction de validation des données
            def is_valid(val):
                """Vérifie si une valeur est utilisable pour les calculs"""
                if val is None:
                    return False
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    return not (val.isnull().all() or val.isin([np.inf, -np.inf]).any())
                return not (isinstance(val, float) and (np.isnan(val) or np.isinf(val)))

            # Validation des données OHLCV
            required_columns = ["open", "high", "low", "close", "volume", "timestamp"]
            if not all(col in ohlcv_df.columns for col in required_columns):
                print(f"❌ {symbol}: Données OHLCV incomplètes")
                return {
                    "action": "neutral",
                    "confidence": 0,
                    "signals": {
                        "technical": {"score": 0, "details": {}, "factors": 0},
                        "momentum": {"score": 0, "details": {}, "factors": 0},
                        "orderflow": {
                            "score": 0,
                            "details": {},
                            "factors": 0,
                            "liquidity": 0,
                            "market_pressure": 0,
                        },
                    },
                }

            # Vérification du nombre minimum de points de données
            if len(ohlcv_df) < 20:
                print(f"❌ {symbol}: Données insuffisantes ({len(ohlcv_df)} lignes)")
                return {
                    "action": "neutral",
                    "confidence": 0,
                    "signals": {
                        "technical": {"score": 0, "details": {}, "factors": 0},
                        "momentum": {"score": 0, "details": {}, "factors": 0},
                        "orderflow": {
                            "score": 0,
                            "details": {},
                            "factors": 0,
                            "liquidity": 0,
                            "market_pressure": 0,
                        },
                    },
                }

            # Initialisation des indicateurs avancés
            try:
                advanced_indicators = AdvancedIndicators()
            except Exception as e:
                print(f"❌ Erreur initialisation indicateurs: {e}")
                return {
                    "action": "neutral",
                    "confidence": 0,
                    "signals": {
                        "technical": {"score": 0, "details": {}, "factors": 0},
                        "momentum": {"score": 0, "details": {}, "factors": 0},
                        "orderflow": {
                            "score": 0,
                            "details": {},
                            "factors": 0,
                            "liquidity": 0,
                            "market_pressure": 0,
                        },
                    },
                }

            # === CALCUL DES INDICATEURS ===
            print(f"\n📊 Calcul indicateurs {symbol}-{tf}")

            # 1. Indicateurs de tendance
            trend_indicators = advanced_indicators.indicators["trend"]
            supertrend = trend_indicators["supertrend"](ohlcv_df)
            vwma = trend_indicators["vwma"](ohlcv_df)
            kama = trend_indicators["kama"](ohlcv_df)
            psar = trend_indicators["psar"](ohlcv_df)
            trix = trend_indicators["trix"](ohlcv_df)

            # 2. Indicateurs de momentum
            momentum_indicators = advanced_indicators.indicators["momentum"]
            ao = momentum_indicators["ao"](ohlcv_df)
            williams_r = momentum_indicators["williams_r"](ohlcv_df)
            cci = momentum_indicators["cci"](ohlcv_df)

            # 3. Indicateurs orderflow
            orderflow_indicators = advanced_indicators.indicators["orderflow"]
            delta_vol = orderflow_indicators["delta_volume"](ohlcv_df)
            imbalance = orderflow_indicators["imbalance"](ohlcv_df)
            smi = orderflow_indicators["smart_money_index"](ohlcv_df)
            liq_wave = orderflow_indicators["liquidity_wave"](ohlcv_df)
            bid_ask = orderflow_indicators["bid_ask_ratio"](ohlcv_df)

            # === CALCUL DES SCORES ===
            print(f"\n💯 Calcul scores {symbol}-{tf}")
            # === SCORES TECHNIQUES ===

            tech_score = 0
            tech_factors = 0
            tech_details = {}

            # Score Supertrend
            if supertrend and is_valid(supertrend.get("value", pd.Series()).iloc[-1]):
                tech_factors += 1
                st_direction = safe_float(supertrend["direction"].iloc[-1])
                st_strength = safe_float(supertrend["strength"].iloc[-1])
                tech_score += st_strength if st_direction > 0 else -st_strength
                tech_details["supertrend"] = {
                    "direction": st_direction,
                    "strength": st_strength,
                }
                print(
                    f"[TECH] Supertrend: direction={st_direction} strength={st_strength:.3f}"
                )

            if is_valid(vwma.iloc[-1]):
                tech_factors += 1
                vwma_val = safe_float(vwma.iloc[-1])
                close_val = safe_float(ohlcv_df["close"].iloc[-1])
                vwma_diff = (close_val - vwma_val) / vwma_val if vwma_val else 0
                vwma_score = np.clip(vwma_diff * 3, -1, 1)
                tech_score += vwma_score
                tech_details["vwma"] = vwma_score
                print(f"[TECH] VWMA score: {vwma_score:.3f}")

            if is_valid(kama.iloc[-1]):
                tech_factors += 1
                kama_val = safe_float(kama.iloc[-1])
                close_val = safe_float(ohlcv_df["close"].iloc[-1])
                kama_diff = (close_val - kama_val) / kama_val if kama_val else 0
                kama_score = np.clip(kama_diff * 3, -1, 1)
                tech_score += kama_score
                tech_details["kama"] = kama_score
                print(f"[TECH] KAMA score: {kama_score:.3f}")

            if psar and is_valid(psar.get("value", pd.Series()).iloc[-1]):
                tech_factors += 1
                psar_trend = safe_float(psar["trend"].iloc[-1])
                psar_strength = safe_float(psar["strength"].iloc[-1])
                tech_score += psar_strength if psar_trend > 0 else -psar_strength
                tech_details["psar"] = {"trend": psar_trend, "strength": psar_strength}
                print(f"[TECH] PSAR: trend={psar_trend} strength={psar_strength:.3f}")

            if is_valid(trix.iloc[-1]):
                tech_factors += 1
                trix_val = safe_float(trix.iloc[-1])
                trix_score = np.clip(trix_val * 0.2, -1, 1)
                tech_score += trix_score
                tech_details["trix"] = trix_score
                print(f"[TECH] TRIX score: {trix_score:.3f}")

            # MOMENTUM
            momentum_details = {}
            if is_valid(ao.iloc[-1]):
                momentum_factors += 1
                ao_value = safe_float(ao.iloc[-1])
                ao_score = np.sign(ao_value) * min(abs(ao_value * 0.1), 1)
                momentum_score += ao_score
                momentum_details["ao"] = ao_score
                print(f"[MOMENTUM] AO score: {ao_score:.3f}")

            if is_valid(williams_r.iloc[-1]):
                momentum_factors += 1
                wr_value = safe_float(williams_r.iloc[-1])
                williams_score = 1 if wr_value < -80 else -1 if wr_value > -20 else 0
                momentum_score += williams_score
                momentum_details["williams_r"] = williams_score
                print(f"[MOMENTUM] Williams %R score: {williams_score:.3f}")

            if is_valid(cci.iloc[-1]):
                momentum_factors += 1
                cci_val = safe_float(cci.iloc[-1])
                cci_score = np.clip(cci_val / 100, -1, 1)
                momentum_score += cci_score
                momentum_details["cci"] = cci_score
                print(f"[MOMENTUM] CCI score: {cci_score:.3f}")

            # ORDERFLOW
            flow_details = {}
            if is_valid(delta_vol.iloc[-1]):
                flow_factors += 1
                delta_val = safe_float(delta_vol.iloc[-1])
                delta_mean = safe_float(delta_vol.abs().mean())
                delta_score = np.clip(
                    delta_val / delta_mean if delta_mean else 0, -1, 1
                )
                flow_score += delta_score
                flow_details["delta_volume"] = delta_score
                print(f"[FLOW] Delta Volume score: {delta_score:.3f}")

            if is_valid(imbalance.iloc[-1]):
                flow_factors += 1
                imb_val = safe_float(imbalance.iloc[-1])
                imb_mean = safe_float(imbalance.abs().mean())
                imb_score = np.clip(imb_val / imb_mean if imb_mean else 0, -1, 1)
                flow_score += imb_score
                flow_details["imbalance"] = imb_score
                print(f"[FLOW] Imbalance score: {imb_score:.3f}")

            if is_valid(smi.iloc[-1]):
                flow_factors += 1
                smi_val = safe_float(smi.iloc[-1])
                smi_mean = safe_float(smi.abs().mean())
                smi_score = np.clip(smi_val / smi_mean if smi_mean else 0, -1, 1)
                flow_score += smi_score
                flow_details["smi"] = smi_score
                print(f"[FLOW] SMI score: {smi_score:.3f}")

            # Normalisation
            if tech_factors > 0:
                tech_score /= tech_factors
            if momentum_factors > 0:
                momentum_score /= momentum_factors
            if flow_factors > 0:
                flow_score /= flow_factors

            liquidity_score = 0
            if is_valid(liq_wave.iloc[-1]):
                liq_val = safe_float(liq_wave.iloc[-1])
                liq_mean = safe_float(liq_wave.abs().mean())
                liquidity_score = -np.clip(liq_val / liq_mean if liq_mean else 0, -1, 1)
                print(f"[FLOW] Liquidity score: {liquidity_score:.3f}")

            market_pressure = 0
            if is_valid(bid_ask):
                bid_ask_val = safe_float(bid_ask)
                market_pressure = (bid_ask_val - 0.5) * 2
                print(f"[FLOW] Market pressure: {market_pressure:.3f}")

            signals = {
                "technical": {
                    "score": tech_score,
                    "details": tech_details,
                    "factors": tech_factors,
                },
                "momentum": {
                    "score": momentum_score,
                    "details": momentum_details,
                    "factors": momentum_factors,
                },
                "orderflow": {
                    "score": flow_score,
                    "details": flow_details,
                    "factors": flow_factors,
                    "liquidity": liquidity_score,
                    "market_pressure": market_pressure,
                },
            }
            weights = {"technical": 0.4, "momentum": 0.3, "orderflow": 0.3}
            total_score = (
                safe_float(signals["technical"]["score"]) * weights["technical"]
                + safe_float(signals["momentum"]["score"]) * weights["momentum"]
                + safe_float(signals["orderflow"]["score"]) * weights["orderflow"]
            )
            total_score = np.clip(total_score, -1, 1)

            # Poids adaptatifs
            if abs(liquidity_score) > 0.7:
                weights["orderflow"] *= 1.3
                weights["technical"] *= 0.7
            if abs(market_pressure) > 0.7:
                weights["momentum"] *= 1.2
                weights["technical"] *= 0.8

            volatility_adv = self.calculate_volatility_advanced(ohlcv_df)
            if volatility_adv > 0.05:
                weights["orderflow"] *= 1.2
                weights["technical"] *= 0.8

            exposure_mult = self.optimize_portfolio_exposure()
            total_score *= safe_float(exposure_mult, 1.0)
            divergence_score = self.check_volume_divergence(ohlcv_df)
            if abs(divergence_score) > 0.5:
                total_score *= 1.2 if divergence_score > 0 else 0.8

            volatility_factor = 1.0 + (
                abs(liquidity_score) * 0.5 if is_valid(liquidity_score) else 0
            )
            buy_threshold = 0.2 * volatility_factor
            sell_threshold = -0.2 * volatility_factor
            if market_pressure > 0:
                buy_threshold *= 1 - market_pressure * 0.2
                sell_threshold *= 1 + market_pressure * 0.2
            else:
                buy_threshold *= 1 + abs(market_pressure) * 0.2
                sell_threshold *= 1 - abs(market_pressure) * 0.2

            decision = {
                "action": "neutral",
                "confidence": abs(total_score),
                "signals": signals,
                "metrics": {
                    "volatility_factor": volatility_factor,
                    "market_pressure": market_pressure,
                    "liquidity_score": liquidity_score,
                    "thresholds": {"buy": buy_threshold, "sell": sell_threshold},
                    "weights": weights,
                },
                "timestamp": "2025-08-03 03:37:58",
                "symbol": symbol,
                "timeframe": tf,
            }

            if total_score > buy_threshold:
                decision["action"] = "buy"
                log_reason = "Signal d'achat"
            elif total_score < sell_threshold:
                decision["action"] = "sell"
                log_reason = "Signal de vente"
            else:
                log_reason = "Signal neutre"

            log_msg = (
                f"[ANALYZE] {symbol} {tf}\n"
                f"Technical Score: {tech_score:.3f} ({tech_factors} facteurs)\n"
                f"Momentum Score: {momentum_score:.3f} ({momentum_factors} facteurs)\n"
                f"Orderflow Score: {flow_score:.3f} ({flow_factors} facteurs)\n"
                f"Total Score: {total_score:.3f}\n"
                f"Seuils - Achat: {buy_threshold:.3f}, Vente: {sell_threshold:.3f}\n"
                f"Action: {decision['action'].upper()} ({decision['confidence']:.3f})\n"
                f"Raison: {log_reason}"
            )
            print(f"[DEBUG] {log_msg}")

            log_dashboard(
                f"[TRADE-DECISION] {symbol} | Action: {decision['action'].upper()} | "
                f"Confiance: {decision['confidence']:.2f} | Score: {total_score:.4f} | "
                f"Tech: {tech_score:.2f} | "
                f"AI: {self.market_data.get(symbol, {}).get('ai_prediction', 0):.2f}"
            )

            if hasattr(self, "save_analysis_metrics"):
                self.save_analysis_metrics(
                    {
                        "symbol": symbol,
                        "timeframe": tf,
                        "timestamp": "",
                        "user": "Patmoorea",
                        "decision": decision,
                        "raw_signals": {
                            "supertrend": tech_details.get("supertrend"),
                            "vwma": tech_details.get("vwma"),
                            "kama": tech_details.get("kama"),
                            "psar": tech_details.get("psar"),
                            "trix": tech_details.get("trix"),
                            "ao": momentum_details.get("ao"),
                            "williams_r": momentum_details.get("williams_r"),
                            "cci": momentum_details.get("cci"),
                            "delta_volume": flow_details.get("delta_volume"),
                            "imbalance": flow_details.get("imbalance"),
                            "liquidity": liquidity_score,
                            "bid_ask": market_pressure,
                        },
                    }
                )

            self.market_data[symbol][tf]["signals"] = {
                "technical": signals["technical"],
                "momentum": signals["momentum"],
                "orderflow": signals["orderflow"],
                "ai": self.market_data.get(symbol, {}).get("ai_prediction", 0),
                "sentiment": self.market_data.get(symbol, {}).get("sentiment", 0),
            }

            return decision

        except Exception as e:
            error_msg = f"Erreur dans analyze_signals: {str(e)}"
            self.logger.error(error_msg)
            return {"action": "neutral", "confidence": 0, "signals": {}}

    def calculate_atr(self, df, period=14):
        """Calcul de l'Average True Range (ATR) pour stop-loss dynamique."""
        try:
            high = np.array(df["high"])
            low = np.array(df["low"])
            close = np.array(df["close"])

            if len(high) < 2 or len(low) < 2 or len(close) < 2:
                return 0.01

            tr = np.maximum(
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            )

            atr = pd.Series(tr).rolling(window=period).mean()
            return float(atr.iloc[-1]) if len(atr) > 0 else 0.01

        except Exception as e:
            self.logger.error(f"Erreur calcul ATR: {e}")
            return 0.01

    def add_indicators(self, df):
        """
        Calcule tous les indicateurs nécessaires pour les stratégies du dossier 'strategies'.
        Retourne un dictionnaire {nom_indicateur: dernière_valeur non-NaN ou None}
        (Version enrichie avec indicateurs avancés)
        Corrige définitivement le warning VWAP/VWMA not datetime ordered de pandas-ta !
        """
        import pandas as pd
        import numpy as np

        try:
            # --- Conversion stricte et tri ---
            # Si df est une liste, transforme-le en DataFrame
            if isinstance(df, list):
                if len(df) == 0:
                    self.logger.error("add_indicators: Liste reçue vide")
                    return None
                if isinstance(df[0], dict):
                    df = pd.DataFrame(df)
                elif isinstance(df[0], (list, tuple)):
                    columns = ["timestamp", "open", "high", "low", "close", "volume"]
                    df = pd.DataFrame(df, columns=columns)
                else:
                    self.logger.error(
                        "add_indicators: Format de liste non pris en charge"
                    )
                    return None

            if not isinstance(df, pd.DataFrame):
                self.logger.error("add_indicators: df n'est pas un DataFrame")
                return None

            # --- Vérification et correction colonne timestamp ---
            if "timestamp" not in df.columns:
                self.logger.error("add_indicators: colonne 'timestamp' manquante")
                return None

            # --- Conversion stricte timestamp ---
            try:
                # Si timestamp n'est pas datetime, convertis-le
                if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
                    # Si c'est en ms, convertis-le
                    # Heuristique: timestamp > 1e12 => probablement en ms
                    if df["timestamp"].max() > 1e12:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    else:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception as e:
                self.logger.error(f"add_indicators: Erreur conversion timestamp: {e}")
                return None

            # --- Tri strict ---
            df = df.drop_duplicates(subset="timestamp", keep="last")
            df = df.sort_values("timestamp")
            df = df.reset_index(drop=True)

            required_cols = {"open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                self.logger.error(
                    f"add_indicators: Colonnes manquantes: {required_cols - set(df.columns)} | Colonnes actuelles: {df.columns.tolist()}"
                )
                return None

            MIN_LEN = 30
            if df is None or len(df) < MIN_LEN:
                self.logger.warning(
                    f"DataFrame vide ou insuffisant ({0 if df is None else len(df)}) lignes"
                )
                return None

            if df.empty:
                self.logger.warning(
                    "DataFrame vide, impossible de calculer les indicateurs"
                )
                print("[DEBUG add_indicators] DataFrame vide après tri/formatage")
                return None

            try:
                df_ta = df.copy()

                # Tri STRICT + SET INDEX avant CHAQUE calcul d'indicateur avancé (VWMA, VWAP, OBV, etc.)
                def strict_sort_and_index(df):
                    if "timestamp" in df.columns:
                        df = df.drop_duplicates(subset="timestamp", keep="last")
                        df = df.sort_values("timestamp")
                        df = df.set_index("timestamp")
                    return df

                # Calcul des indicateurs classiques (index classique)
                sma_20 = df_ta.ta.sma(length=20, append=False)
                if sma_20 is not None and not sma_20.empty:
                    if isinstance(sma_20, pd.Series):
                        df_ta["sma_20"] = sma_20
                    elif "SMA_20" in sma_20:
                        df_ta["sma_20"] = sma_20["SMA_20"]

                sma_50 = df_ta.ta.sma(length=50, append=False)
                if sma_50 is not None and not sma_50.empty:
                    if isinstance(sma_50, pd.Series):
                        df_ta["sma_50"] = sma_50
                    elif "SMA_50" in sma_50:
                        df_ta["sma_50"] = sma_50["SMA_50"]

                ema_20 = df_ta.ta.ema(length=20, append=False)
                if ema_20 is not None and not ema_20.empty:
                    if isinstance(ema_20, pd.Series):
                        df_ta["ema_20"] = ema_20
                    elif "EMA_20" in ema_20:
                        df_ta["ema_20"] = ema_20["EMA_20"]

                rsi_14 = df_ta.ta.rsi(length=14, append=False)
                if rsi_14 is not None and not rsi_14.empty:
                    if isinstance(rsi_14, pd.Series):
                        df_ta["rsi_14"] = rsi_14
                    elif "RSI_14" in rsi_14:
                        df_ta["rsi_14"] = rsi_14["RSI_14"]

                macd = df_ta.ta.macd()
                if macd is not None and not macd.empty:
                    if "MACD_12_26_9" in macd:
                        df_ta["macd"] = macd["MACD_12_26_9"]
                    if "MACDs_12_26_9" in macd:
                        df_ta["macd_signal"] = macd["MACDs_12_26_9"]
                    if "MACDh_12_26_9" in macd:
                        df_ta["macd_hist"] = macd["MACDh_12_26_9"]

                bb = df_ta.ta.bbands(length=20, std=2.0)
                if bb is not None and not bb.empty:
                    if "BBL_20_2.0" in bb:
                        df_ta["bb_lower"] = bb["BBL_20_2.0"]
                    if "BBU_20_2.0" in bb:
                        df_ta["bb_upper"] = bb["BBU_20_2.0"]

                df_ta["donchian_high"] = df_ta["high"].rolling(window=20).max()
                df_ta["donchian_low"] = df_ta["low"].rolling(window=20).min()

                psar = df_ta.ta.psar()
                if psar is not None and not psar.empty:
                    key = [col for col in psar.columns if col.startswith("PSAR")][0]
                    df_ta["psar"] = psar[key]

                mom_10 = df_ta.ta.mom(length=10, append=False)
                if mom_10 is not None and not mom_10.empty:
                    if isinstance(mom_10, pd.Series):
                        df_ta["momentum_10"] = mom_10
                    elif "MOM_10" in mom_10:
                        df_ta["momentum_10"] = mom_10["MOM_10"]

                df_ta["zscore_20"] = (
                    df_ta["close"] - df_ta["close"].rolling(20).mean()
                ) / df_ta["close"].rolling(20).std()

                # Indicateurs avancés supplémentaires : TRI + SET INDEX obligatoire pour pandas-ta VWAP/VWMA/OBV
                try:
                    df_ta_idx = strict_sort_and_index(df_ta)
                    vwma = df_ta_idx.ta.vwma(length=20)
                    # On remet l'index timestamp dans la colonne pour rester compatible
                    df_ta["vwma_20"] = vwma.values
                except Exception:
                    df_ta["vwma_20"] = np.nan
                try:
                    df_ta_idx = strict_sort_and_index(df_ta)
                    obv = df_ta_idx.ta.obv()
                    df_ta["obv"] = obv.values
                except Exception:
                    df_ta["obv"] = np.nan
                try:
                    df_ta_idx = strict_sort_and_index(df_ta)
                    vwap = df_ta_idx.ta.vwap()
                    df_ta["vwap"] = vwap.values
                except Exception:
                    df_ta["vwap"] = np.nan
                try:
                    stochrsi = df_ta.ta.stochrsi()
                    if stochrsi is not None and not stochrsi.empty:
                        df_ta["stochrsi"] = stochrsi.iloc[:, 0]
                except Exception:
                    df_ta["stochrsi"] = np.nan
                try:
                    kc = df_ta.ta.kc()
                    if kc is not None and not kc.empty:
                        df_ta["kc_upper"] = kc["KCUpper_20_2_10"]
                        df_ta["kc_lower"] = kc["KCLower_20_2_10"]
                except Exception:
                    df_ta["kc_upper"] = df_ta["kc_lower"] = np.nan
                try:
                    supertrend = df_ta.ta.supertrend(length=7, multiplier=3.0)
                    if supertrend is not None and not supertrend.empty:
                        df_ta["supertrend"] = supertrend.iloc[:, 0]
                except Exception:
                    pass
                try:
                    ichimoku = df_ta.ta.ichimoku()
                    if ichimoku is not None and not ichimoku.empty:
                        df_ta["ichimoku_a"] = ichimoku["ISA_9"]
                        df_ta["ichimoku_b"] = ichimoku["ISB_26"]
                except Exception:
                    pass
                try:
                    keltner = df_ta.ta.kc()
                    if keltner is not None and not keltner.empty:
                        df_ta["keltner_upper"] = keltner["KCUpper_20_2_10"]
                        df_ta["keltner_lower"] = keltner["KCLower_20_2_10"]
                except Exception:
                    pass
                try:
                    accdist = df_ta.ta.accdist()
                    df_ta["accumulation"] = accdist
                except Exception:
                    pass

                all_indics = [
                    "sma_20",
                    "sma_50",
                    "ema_20",
                    "rsi_14",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "bb_lower",
                    "bb_upper",
                    "donchian_high",
                    "donchian_low",
                    "psar",
                    "momentum_10",
                    "zscore_20",
                    "vwma_20",
                    "obv",
                    "vwap",
                    "stochrsi",
                    "kc_upper",
                    "kc_lower",
                ]

                indicators = {}
                for col in all_indics:
                    if col in df_ta.columns:
                        last_valid = df_ta[col].dropna()
                        indicators[col] = (
                            last_valid.iloc[-1] if not last_valid.empty else None
                        )
                    else:
                        indicators[col] = None

                # === AJOUT IMPORTANT : Calcul du score technique ===
                rsi_value = indicators.get("rsi_14")
                if rsi_value is not None:
                    tech_score = (
                        safe_float(rsi_value) / 100.0
                    )  # Normalisation entre 0 et 1
                else:
                    tech_score = 0.5  # Valeur par défaut
                indicators["technical_score"] = tech_score

            except Exception as e:
                self.logger.warning(f"Erreur pandas-ta indicateurs principaux : {e}")
                indicators = {"technical_score": 0.5}  # Au minimum le score technique

            n_valid = len([v for v in indicators.values() if v is not None])
            self.logger.info(
                f"✅ {n_valid} indicateurs extraits automatiquement sur {df.shape[0]} lignes"
            )
            print(
                f"[DEBUG add_indicators] {n_valid} indicateurs extraits: {list(indicators.keys())[:5]}"
            )
            return indicators

        except Exception as e:
            self.logger.error(f"❌ Erreur calcul indicateurs: {e}")
            return {
                "technical_score": 0.5
            }  # Toujours retourner au moins le score technique

    def analyze_order_flow(self, df):
        """
        Analyse avancée du flux d'ordres pour détecter la pression acheteur/vendeur.
        """
        try:
            # Calcul de l'absorption
            buying_pressure = (
                (df["close"] - df["low"]) / (df["high"] - df["low"])
            ) * df["volume"]
            selling_pressure = (
                (df["high"] - df["close"]) / (df["high"] - df["low"])
            ) * df["volume"]

            # CVD (Cumulative Volume Delta)
            cvd = (buying_pressure - selling_pressure).cumsum()

            # Imbalance Detection
            imbalance = (
                abs(buying_pressure.mean() - selling_pressure.mean())
                / selling_pressure.mean()
            )

            return {
                "buying_pressure": buying_pressure.iloc[-1],
                "selling_pressure": selling_pressure.iloc[-1],
                "cvd": cvd.iloc[-1],
                "imbalance": imbalance,
                "pressure_ratio": (
                    buying_pressure.iloc[-1] / selling_pressure.iloc[-1]
                    if selling_pressure.iloc[-1] != 0
                    else 1
                ),
            }
        except Exception as e:
            self.logger.error(f"Erreur analyse order flow: {e}")
            return {
                "buying_pressure": 0,
                "selling_pressure": 0,
                "cvd": 0,
                "imbalance": 0,
                "pressure_ratio": 1,
            }

    def analyze_volume_profile(self, symbol, timeframe="1h"):
        """Analyse avancée du volume profile"""
        df = self.get_timeframe_data(symbol, timeframe)
        if df is None:
            return None

        # Calcul des points d'accumulation/distribution
        volume_nodes = self.calculate_volume_nodes(df)

        # Détection des zones de haute liquidité
        liquidity_zones = self.identify_liquidity_zones(df)

        return {
            "volume_nodes": volume_nodes,
            "liquidity_zones": liquidity_zones,
            "poc_price": self.calculate_poc_price(df),
        }

    def check_volume_divergence(self, df):
        """Détecte les divergences prix/volume"""
        try:
            import pandas as pd

            if not isinstance(df, pd.DataFrame) or len(df) < 20:
                return 0

            # Calcul des tendances
            price_trend = pd.Series(df["close"]).pct_change(20)
            volume_trend = pd.Series(df["volume"]).pct_change(20)

            # Corrélation prix/volume
            correlation = price_trend.corr(volume_trend)

            # Divergence score
            if pd.isna(correlation):
                return 0

            # Normalisation entre -1 et 1
            divergence_score = -float(correlation)  # Inverse de la corrélation

            print(f"[DIVERGENCE] Score: {divergence_score:.2f}")

            return divergence_score

        except Exception as e:
            print(f"Erreur calcul divergence: {e}")
            return 0

    async def _prepare_features_for_ai(self, symbol):
        import numpy as np

        try:
            N_STEPS = self.N_STEPS

            ohlcv = self.market_data.get(symbol, {}).get("1h", {})
            if not ohlcv or not isinstance(ohlcv, dict) or "close" not in ohlcv:
                return None

            closes = np.array(ohlcv.get("close", []), dtype=float)
            highs = np.array(ohlcv.get("high", []), dtype=float)
            lows = np.array(ohlcv.get("low", []), dtype=float)
            volumes = np.array(ohlcv.get("volume", []), dtype=float)

            if (
                len(closes) < N_STEPS
                or len(highs) < N_STEPS
                or len(lows) < N_STEPS
                or len(volumes) < N_STEPS
            ):
                return None

            closes = closes[-N_STEPS:]
            highs = highs[-N_STEPS:]
            lows = lows[-N_STEPS:]
            volumes = volumes[-N_STEPS:]

            delta = np.diff(closes)
            gain = (delta > 0) * delta
            loss = (delta < 0) * -delta
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0.001
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))

            ema12 = np.mean(closes[-12:]) if len(closes) >= 12 else closes[-1]
            ema26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
            macd = ema12 - ema26

            if len(closes) >= N_STEPS:
                returns = np.diff(np.log(closes))
                volatility = float(np.std(returns[-14:])) if len(returns) >= 14 else 0
            else:
                volatility = 0

            avg_volume = np.mean(volumes) if np.mean(volumes) > 0 else 1
            vol_ratio = float(volumes[-1]) / avg_volume if avg_volume > 0 else 1
            vol_ratio = min(1, vol_ratio / 3)

            features = {
                "close": closes / closes[0] if closes[0] != 0 else closes,
                "high": highs / highs[0] if highs[0] != 0 else highs,
                "low": lows / lows[0] if lows[0] != 0 else lows,
                "volume": volumes / volumes[0] if volumes[0] != 0 else volumes,
                "rsi": float(rsi) / 100 if rsi else 0,
                "macd": float(macd) / 100 if macd else 0,
                "volatility": float(volatility),
                "vol_ratio": float(vol_ratio),
            }

            # Correction NaN/inf
            for k in features:
                arr = features[k]
                if isinstance(arr, np.ndarray):
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        print(f"[WARN] NaN/inf détecté dans {k}, correction appliquée")
                        features[k] = np.nan_to_num(arr)
                else:
                    if np.isnan(features[k]) or np.isinf(features[k]):
                        print(f"[WARN] NaN/inf détecté dans {k}, correction appliquée")
                        features[k] = float(np.nan_to_num(features[k]))

            required_keys = [
                "close",
                "high",
                "low",
                "volume",
                "rsi",
                "macd",
                "volatility",
            ]
            for k in required_keys:
                if k not in features:
                    self.logger.error(
                        f"[AI FEATURES] Clé manquante dans features : {k}"
                    )
                    return None
            for k in ["close", "high", "low", "volume"]:
                if not (
                    isinstance(features[k], np.ndarray)
                    and features[k].shape == (N_STEPS,)
                ):
                    self.logger.error(
                        f"[AI FEATURES] Mauvais shape pour {k}: {type(features[k])}, shape={getattr(features[k], 'shape', None)}"
                    )
                    return None
            for k in ["rsi", "macd", "volatility", "vol_ratio"]:
                if not isinstance(features[k], (int, float, np.floating, np.integer)):
                    self.logger.error(
                        f"[AI FEATURES] Mauvais type pour {k}: {type(features[k])}"
                    )
                    return None

            return features

        except Exception as e:
            self.logger.error(f"Error preparing AI features: {e}")
            return None

    async def _merge_signals(self, symbol, dl_prediction, ppo_action):
        try:
            ai_weight = float(getattr(self, "ai_weight", 0.4))
            tech_weight = 1.0 - ai_weight

            if symbol not in self.market_data:
                self.market_data[symbol] = {}

            defaults = {"trend": 0.0, "momentum": 0.0, "volatility": 0.0}
            signals = self.market_data[symbol].get("signals", defaults.copy())

            dl_val = (
                float(dl_prediction) if isinstance(dl_prediction, (int, float)) else 0
            )
            ppo_val = float(ppo_action) if isinstance(ppo_action, (int, float)) else 0

            ai_signal = dl_val * 0.7 + ppo_val * 0.3

            merged = {
                k: (float(v) * tech_weight + ai_signal * ai_weight)
                for k, v in signals.items()
            }

            self.market_data[symbol]["signals"] = merged
            self.market_data[symbol]["ai_prediction"] = ai_signal

            return merged

        except Exception as e:
            self.logger.error(f"Erreur fusion signaux: {e}")
            return defaults.copy()

    def calculate_correlation_matrix(self):
        correlations = {}
        for pair1 in self.pairs_valid:
            for pair2 in self.pairs_valid:
                correlation = self.calculate_pair_correlation(pair1, pair2)
                correlations[f"{pair1}-{pair2}"] = correlation
        return correlations

    def optimize_portfolio_exposure(self):
        try:
            if not self.positions:
                return 1.0

            correlations = self.calculate_correlation_matrix()
            high_corr_pairs = 0
            total_pairs = 0

            for pair1 in self.positions:
                for pair2 in self.positions:
                    if pair1 != pair2:
                        corr_key = f"{pair1}-{pair2}"
                        corr = correlations.get(corr_key, 0)
                        if abs(float(corr)) > 0.7:
                            high_corr_pairs += 1
                        total_pairs += 1

            if total_pairs == 0:
                return 1.0

            corr_ratio = high_corr_pairs / total_pairs
            exposure_multiplier = 1.0 - (corr_ratio * 0.3)

            print(
                f"[EXPOSURE] Multiplicateur: {exposure_multiplier:.2f} (corrélations élevées: {high_corr_pairs})"
            )
            return max(0.5, exposure_multiplier)

        except Exception as e:
            print(f"Erreur optimisation exposition: {e}")
            return 1.0

    def adjust_position_sizing(self, base_size, correlation_factor):
        return float(base_size) * (1 - float(correlation_factor))

    def weighted_signal_fusion(self, signals):
        weights = {"technical": 0.4, "ai": 0.3, "sentiment": 0.2, "orderflow": 0.1}
        total_score = 0
        for signal_type, value in signals.items():
            if signal_type in weights:
                total_score += float(value) * weights[signal_type]
        return total_score

    def track_advanced_metrics(self):
        metrics = {
            "sharpe_ratio": self.calculate_sharpe(),
            "sortino_ratio": self.calculate_sortino(),
            "calmar_ratio": self.calculate_calmar(),
            "win_rate": self.get_win_rate(),
            "avg_profit": self.get_avg_profit(),
            "max_drawdown": self.get_max_drawdown(),
        }
        return metrics

    def safe_trade_execution(self, order):
        try:
            self.check_margin_requirements()
            self.verify_risk_limits()
            self.check_market_conditions()
            for attempt in range(3):
                try:
                    result = self.execute_order(order)
                    return result
                except ConnectionError:
                    continue
        except Exception as e:
            self.logger.error(f"Erreur exécution: {e}")
            return None

    def backup_critical_data(self):
        try:
            backup_data = {
                "timestamp": get_current_time_tahiti(),
                "positions": self.positions_binance,
                "market_data": self.market_data,
                "indicators": self.indicators,
                "system_metrics": self.system_metrics,
                "performance": self.get_performance_metrics(),
            }
            import lz4.frame
            import json

            backup_path = (
                f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.lz4"
            )
            with lz4.frame.open(backup_path, "wb") as f:
                f.write(json.dumps(backup_data).encode())
            self.cleanup_old_backups(days=7)
            return True
        except Exception as e:
            self.logger.error(f"Erreur backup: {e}")
            return False

    def monitor_system_health(self):
        try:
            import psutil

            metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "api_latency": self.measure_api_latency(),
                "ws_status": self.check_ws_status(),
            }
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.system_metrics[key].append(
                        {"timestamp": get_current_time_tahiti(), "value": value}
                    )
            return metrics
        except Exception as e:
            self.logger.error(f"Erreur monitoring système: {e}")
            return {}

    def analyze_order_pressure(self, symbol):
        orderbook = self.get_ws_orderbook(symbol)
        if not orderbook:
            return None
        bid_pressure = self.calculate_bid_pressure(orderbook["bids"])
        ask_pressure = self.calculate_ask_pressure(orderbook["asks"])
        return {
            "bid_pressure": float(bid_pressure),
            "ask_pressure": float(ask_pressure),
            "imbalance": float(bid_pressure) - float(ask_pressure),
        }

    def calculate_dynamic_stoploss(self, symbol, timeframe="1h"):
        try:
            df = self.get_timeframe_data(symbol, timeframe)
            if df is None:
                return self.stop_loss_pct
            atr = self.calculate_atr(df, period=14)
            price = float(df["close"][-1])
            atr_pct = float(atr) / price if price != 0 else 0.01
            dynamic_sl = min(max(atr_pct * 2, 0.01), 0.03)
            return dynamic_sl
        except Exception as e:
            self.logger.error(f"Erreur calcul stop-loss dynamique: {e}")
            return self.stop_loss_pct

    def analyze_correlations(self):
        correlations = {}
        for pair1 in self.pairs_valid:
            correlations[pair1] = {}
            for pair2 in self.pairs_valid:
                if pair1 != pair2:
                    corr = self.calculate_pair_correlation(pair1, pair2)
                    correlations[pair1][pair2] = float(corr)
        return correlations

    def calculate_trade_quality_score(self, trade_data):
        score = 0
        if self.is_near_key_level(trade_data["symbol"], trade_data["price"]):
            score += 2
        if float(trade_data.get("volume_ratio", 1)) > 1.5:
            score += 1
        if self.check_momentum_alignment(trade_data):
            score += 1
        if self.check_timeframe_alignment(trade_data):
            score += 2
        return score

    def calculate_squeeze_momentum(self, df):
        import pandas as pd

        try:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            mean_tp = typical_price.rolling(window=20).mean()
            atr = (
                pd.DataFrame(
                    {
                        "h-l": df["high"] - df["low"],
                        "h-pc": abs(df["high"] - df["close"].shift(1)),
                        "l-pc": abs(df["low"] - df["close"].shift(1)),
                    }
                )
                .max(axis=1)
                .rolling(window=20)
                .mean()
            )
            keltner_up = mean_tp + (atr * 1.5)
            keltner_down = mean_tp - (atr * 1.5)
            std = df["close"].rolling(window=20).std()
            bb_up = mean_tp + (std * 2)
            bb_down = mean_tp - (std * 2)
            squeeze_on = (bb_up <= keltner_up) & (bb_down >= keltner_down)
            highest = df["high"].rolling(window=20).max()
            lowest = df["low"].rolling(window=20).min()
            momentum = df["close"] - ((highest + lowest) / 2)
            return {
                "squeeze_on": bool(squeeze_on.iloc[-1]),
                "momentum": float(momentum.iloc[-1]),
                "momentum_change": float(momentum.diff().iloc[-1]),
            }
        except Exception as e:
            self.logger.error(f"Erreur calcul squeeze momentum: {e}")
            return {"squeeze_on": False, "momentum": 0, "momentum_change": 0}

    def analyze_volume_distribution(self, df):
        import pandas as pd

        try:
            price_bins = pd.qcut(df["close"], q=10, duplicates="drop")
            vol_profile = df.groupby(price_bins)["volume"].sum()
            poc_price = float(vol_profile.idxmax().left)
            buy_volume = df["volume"][df["close"] > df["open"]].sum()
            sell_volume = df["volume"][df["close"] < df["open"]].sum()
            delta = float(buy_volume) - float(sell_volume)
            is_accumulation = delta > 0 and df["close"].iloc[-1] > df["close"].mean()
            is_distribution = delta < 0 and df["close"].iloc[-1] < df["close"].mean()
            buy_volume_ratio = (
                float(buy_volume) / (float(buy_volume) + float(sell_volume))
                if (float(buy_volume) + float(sell_volume)) > 0
                else 0.5
            )
            return {
                "poc_price": poc_price,
                "volume_delta": delta,
                "is_accumulation": bool(is_accumulation),
                "is_distribution": bool(is_distribution),
                "buy_volume_ratio": buy_volume_ratio,
            }
        except Exception as e:
            self.logger.error(f"Erreur analyse volume: {e}")
            return {
                "poc_price": 0,
                "volume_delta": 0,
                "is_accumulation": False,
                "is_distribution": False,
                "buy_volume_ratio": 0.5,
            }

    def identify_market_structure(self, df):
        import pandas as pd

        try:
            n = 5
            highs = df["high"].rolling(window=2 * n + 1, center=True).max()
            lows = df["low"].rolling(window=2 * n + 1, center=True).min()
            higher_highs = bool((highs.diff() > 0).iloc[-1])
            lower_lows = bool((lows.diff() < 0).iloc[-1])
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()
            trend_strength = (
                (float(ema20.iloc[-1]) - float(ema50.iloc[-1])) / float(ema50.iloc[-1])
                if ema50.iloc[-1] != 0
                else 0
            )
            atr = (
                pd.DataFrame(
                    {
                        "h-l": df["high"] - df["low"],
                        "h-pc": abs(df["high"] - df["close"].shift(1)),
                        "l-pc": abs(df["low"] - df["close"].shift(1)),
                    }
                )
                .max(axis=1)
                .rolling(window=14)
                .mean()
            )
            is_ranging = bool(atr.iloc[-1] < atr.mean())
            return {
                "trend_strength": trend_strength,
                "is_ranging": is_ranging,
                "higher_highs": higher_highs,
                "lower_lows": lower_lows,
                "structure": "range" if is_ranging else "trend",
            }
        except Exception as e:
            self.logger.error(f"Erreur identification structure: {e}")
            return {
                "trend_strength": 0,
                "is_ranging": False,
                "higher_highs": False,
                "lower_lows": False,
                "structure": "unknown",
            }

    def multi_timeframe_analysis(self, symbol, timeframes):
        """
        Analyse multi-timeframes pour confirmation.
        """
        try:
            analysis = {}
            for tf in timeframes:
                df = self.get_timeframe_data(symbol, tf)
                if df is None or len(df) < 50:
                    continue

                # Analyses par timeframe
                structure = self.identify_market_structure(df)
                volume = self.analyze_volume_distribution(df)
                momentum = self.calculate_squeeze_momentum(df)

                alignment = self.check_alignment(df)
                analysis[tf] = {
                    "structure": structure,
                    "volume": volume,
                    "momentum": momentum,
                    "alignment": alignment,
                }

            return analysis
        except Exception as e:
            self.logger.error(f"Erreur analyse multi-timeframes: {e}")
            return {}

    def calculate_confirmation_score(self, indicators, mtp_analysis):
        """
        Calcule un score de confirmation basé sur tous les indicateurs.
        """
        try:
            score = 0
            weight_sum = 0

            # 1. Structure de marché (40%)
            ms = indicators.get("market_structure")
            if ms and isinstance(ms, dict):
                if ms.get("structure") == "trend":
                    score += 0.4 * abs(float(ms.get("trend_strength", 0)))
                weight_sum += 0.4

            # 2. Volume Analysis (30%)
            vp = indicators.get("volume_profile")
            if vp and isinstance(vp, dict):
                if vp.get("is_accumulation"):
                    score += 0.3
                elif vp.get("is_distribution"):
                    score -= 0.3
                weight_sum += 0.3

            # 3. Momentum & Squeeze (20%)
            sq = indicators.get("squeeze_momentum")
            if sq and isinstance(sq, dict):
                if sq.get("squeeze_on") and float(sq.get("momentum", 0)) > 0:
                    score += 0.2
                elif sq.get("squeeze_on") and float(sq.get("momentum", 0)) < 0:
                    score -= 0.2
                weight_sum += 0.2

            # 4. Order Flow (10%)
            of = indicators.get("order_flow")
            if of and isinstance(of, dict):
                pressure_ratio = float(of.get("pressure_ratio", 1))
                if pressure_ratio > 1:
                    score += 0.1 * min(pressure_ratio - 1, 1)
                elif pressure_ratio < 1:
                    score -= 0.1 * min(1 - pressure_ratio, 1)
                weight_sum += 0.1

            # Normalisation
            if weight_sum > 0:
                score = score / weight_sum

            return score
        except Exception as e:
            self.logger.error(f"Erreur calcul score confirmation: {e}")
            return 0

    def get_volatility_multiplier(self, symbol):
        """
        Retourne un multiplicateur basé sur la volatilité.
        Réduit le sizing quand la volatilité est élevée.
        """
        try:
            import pandas as pd

            df = self.get_recent_data(symbol)
            if df is None or len(df) < 20:
                return 1.0

            atr = (
                pd.DataFrame(
                    {
                        "h-l": df["high"] - df["low"],
                        "h-pc": abs(df["high"] - df["close"].shift(1)),
                        "l-pc": abs(df["low"] - df["close"].shift(1)),
                    }
                )
                .max(axis=1)
                .rolling(window=14)
                .mean()
                .iloc[-1]
            )

            price = float(df["close"].iloc[-1])
            atr_pct = float(atr) / price if price != 0 else 0

            if atr_pct < 0.01:
                return 1.2
            elif atr_pct < 0.02:
                return 1.0
            elif atr_pct < 0.03:
                return 0.8
            elif atr_pct < 0.04:
                return 0.6
            else:
                return 0.4

        except Exception as e:
            self.logger.error(f"Erreur calcul multiplicateur volatilité: {e}")
            return 1.0

    def get_risk_multiplier(self, symbol):
        """
        Calcule un multiplicateur de risque basé sur plusieurs facteurs.
        """
        try:
            multiplier = 1.0

            # 1. Corrélation avec le marché
            correlation = float(self.get_market_correlation(symbol))
            if correlation > 0.8:
                multiplier *= 0.8

            # 2. Liquidité
            liquidity_score = float(self.get_liquidity_score(symbol))
            if liquidity_score < 0.5:
                multiplier *= 0.7

            # 3. Spread moyen
            spread = float(self.get_average_spread(symbol))
            if spread > 0.001:
                multiplier *= 0.9

            # 4. Distance aux supports/résistances
            key_levels = self.get_key_levels(symbol)
            if self.is_near_key_level(symbol, key_levels):
                multiplier *= 1.1

            return max(0.3, min(multiplier, 1.2))

        except Exception as e:
            self.logger.error(f"Erreur calcul multiplicateur risque: {e}")
            return 1.0

    # Méthodes utilitaires nécessaires
    def get_timeframe_data(self, symbol, timeframe):
        """Helper pour récupérer les données d'un timeframe."""
        try:
            return self.market_data.get(symbol, {}).get(timeframe, None)
        except Exception:
            return None

    def check_alignment(self, df):
        """Vérifie l'alignement des indicateurs."""
        try:
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()
            rsi = self.calculate_rsi(df["close"])

            price_trend = ema20.iloc[-1] > ema50.iloc[-1]
            momentum_aligned = rsi.iloc[-1] > 50 if price_trend else rsi.iloc[-1] < 50

            return price_trend and momentum_aligned
        except Exception:
            return False

    def safe_update_shared_data(
        self, new_fields: dict, data_file="src/shared_data.json"
    ):
        """Votre version condensée et optimisée"""
        backup_file = data_file + ".bak"
        temp_file = data_file + ".tmp"

        try:
            # 1. Backup atomique
            if os.path.exists(data_file):
                shutil.copy2(data_file, backup_file)

            # 2. Lecture des données existantes
            shared_data = {}
            if os.path.exists(data_file):
                try:
                    with open(data_file, "r") as f:
                        shared_data = json.load(f)
                        if not isinstance(shared_data, dict):
                            shared_data = {}
                except Exception as e:
                    print(f"[ERROR] Lecture JSON: {e}")
                    shared_data = {}

            # 3. Préparation des données
            current_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            # 4. Vos fonctions internes condensées
            def validate_trade_decision(data):
                return {
                    "action": data.get("action", "neutral"),
                    "confidence": max(
                        0.0, min(1.0, float(data.get("confidence", 0.5)))
                    ),
                    "tech": max(0.0, min(1.0, float(data.get("tech", 0.5)))),
                    "ai": max(0.0, min(1.0, float(data.get("ai", 0.5)))),
                    "sentiment": max(-1.0, min(1.0, float(data.get("sentiment", 0.0)))),
                    "timestamp": current_time_str,
                }

            def deep_cast(data):
                if isinstance(data, dict):
                    return {k: deep_cast(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [deep_cast(i) for i in data]
                elif isinstance(data, str):
                    try:
                        return (
                            float(data) if data.replace(".", "", 1).isdigit() else data
                        )
                    except ValueError:
                        return data
                return data

            new_fields = deep_cast(new_fields)

            # 5. Application de vos règles métier
            if "trade_decisions" in new_fields:
                new_fields["trade_decisions"] = {
                    k: validate_trade_decision(v)
                    for k, v in new_fields["trade_decisions"].items()
                }

            # 6. Fusion profonde optimisée
            def deep_update(target, source):
                for key, value in source.items():
                    if (
                        key in target
                        and isinstance(target[key], dict)
                        and isinstance(value, dict)
                    ):
                        deep_update(target[key], value)
                    elif (
                        key in target
                        and isinstance(target[key], list)
                        and isinstance(value, list)
                    ):
                        # CORRECTION : pending_sales doit être écrasé à chaque cycle
                        if key == "pending_sales":
                            target[key] = value  # <--- PATCH ici : on écrase toujours
                        else:
                            target[key] = value
                    else:
                        target[key] = value
                return target

            # 7. Préservation des champs (votre logique originale)
            preserved_fields = {
                f: shared_data.get(f)
                for f in [
                    "trade_history",
                    "closed_positions",
                    "equity_history",
                    "news_data",
                    "sentiment",
                    "active_pauses",
                    # "pending_sales",
                ]
                if f in shared_data
            }

            # 8. Fusion finale
            shared_data = deep_update(shared_data, new_fields)
            shared_data.update(preserved_fields)

            # 9. Sauvegarde atomique
            with open(temp_file, "w") as f:
                json.dump(shared_data, f, indent=4)
            os.replace(temp_file, data_file)

            return True

        except Exception as e:
            print(f"❌ Erreur: {e}")
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, data_file)
            return False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def check_tp_partial(
        self, entry_price, current_price, filled_tp_targets=None, tp_levels=None
    ):
        """Version sécurisée avec gestion de type stricte"""
        # Conversion et validation
        try:
            entry = float(entry_price)
            current = float(current_price)
            tp_levels = tp_levels or [(0.03, 0.3), (0.07, 0.3)]

            to_exit = 0.0
            filled = (
                filled_tp_targets.copy()
                if filled_tp_targets
                else [False] * len(tp_levels)
            )

            for i, (tp_pct, frac) in enumerate(tp_levels):
                if not filled[i] and entry > 0:
                    if (current - entry) / entry >= float(tp_pct):
                        to_exit += float(frac)
                        filled[i] = True

            return to_exit, filled
        except Exception as e:
            print(f"❌ Erreur calcul TP: {str(e)}")
            return 0.0, filled_tp_targets or [False] * len(tp_levels or [])

    def check_trailing(self, entry_price, price_history, max_price, trailing_pct=0.03):
        """
        Trailing stop universel : sort si le prix retombe de X% par rapport au max atteint.
        """
        if not price_history or len(price_history) < 3:
            return False, max_price
        current_price = price_history[-1]
        if current_price > max_price:
            max_price = current_price
        if current_price < max_price * (1 - trailing_pct):
            return True, max_price
        return False, max_price

    def get_last_fifo_pnl(self, symbol):
        """
        Récupère la plus-value FIFO (en %) et en USD de la dernière vente spot pour une paire donnée.
        Retourne un tuple : (pnl_pct, pnl_usd)
        """
        if not hasattr(self, "binance_client"):
            print("[ERROR] Binance client non initialisé")
            return None, None

        try:
            symbol_key = symbol.replace("/", "").upper()
            buys, sells = self.fetch_trades_fifo(self.binance_client, symbol_key)

            if not buys or not sells:
                return None, None

            fifo_results = self.fifo_pnl(buys, sells)
            if not fifo_results:
                return None, None

            last_result = fifo_results[-1]
            return (
                (
                    float(last_result.get("pnl_pct", 0))
                    if last_result.get("pnl_pct") is not None
                    else None
                ),
                (
                    float(last_result.get("pnl_usd", 0))
                    if last_result.get("pnl_usd") is not None
                    else None
                ),
            )
        except Exception as e:
            print(f"[ERROR] get_last_fifo_pnl pour {symbol}: {str(e)}")
            return None, None

    def log_closed_position(self, symbol, pos, exit_price, reason):
        fifo_pnl_pct, fifo_pnl_usd = self.get_last_fifo_pnl(symbol)
        entry_price = safe_float(pos.get("entry_price"), 0)
        amount = safe_float(pos.get("amount"), 0)
        exit_price = safe_float(exit_price, 0)

        closed_position = {
            "symbol": symbol,
            "side": pos.get("side", ""),
            "amount": amount,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": (
                fifo_pnl_pct
                if fifo_pnl_pct is not None
                else (
                    (exit_price - entry_price) / entry_price * 100 if entry_price else 0
                )
            ),
            "pnl_usd": (
                fifo_pnl_usd
                if fifo_pnl_usd is not None
                else ((exit_price - entry_price) * amount if entry_price else 0)
            ),
            "date": datetime.utcnow().isoformat(),
            "reason": reason,
        }
        # Ajoute à closed_positions dans shared_data.json
        closed = []
        try:
            shared_data = safe_load_shared_data(self.data_file)
            deep_cast_floats(shared_data)
            closed = shared_data.get("closed_positions", [])
        except Exception:
            closed = []
        closed.append(closed_position)
        self.safe_update_shared_data({"closed_positions": closed}, self.data_file)

    def get_pending_sales(self):
        """
        Affiche TOUTES les positions spot Binance avec leur état actuel.
        Seule la position concernée par une pause (asset ou globale) sera marquée pause_blocage = Oui et note = Trading suspendu.
        Les autres champs sont calculés intelligemment.
        PATCH: Corrige l'affichage : pause_blocage = Oui uniquement si pauses != [] et is_paused(symbol) == True
        """
        try:
            pending = []
            now = datetime.utcnow()

            # Récupération des pauses actives RAM
            pauses = []
            if hasattr(self, "news_pause_manager") and self.news_pause_manager:
                pauses = self.news_pause_manager.get_active_pauses()
            print("[DEBUG PATCH] pauses:", pauses)

            def is_paused(symbol):
                # Retourne True uniquement si une pause existe pour la paire ou globale
                if not pauses:
                    return False, ""
                for p in pauses:
                    asset = p.get("asset", "GLOBAL")
                    if asset == "GLOBAL" or asset == symbol:
                        return True, p.get("reason", "Indéterminée")
                return False, ""

            # Pour chaque position SPOT Binance
            if hasattr(self, "positions_binance") and self.positions_binance:
                for symbol, pos in self.positions_binance.items():
                    entry_price = safe_float(pos.get("entry_price"), 0)
                    # PATCH: récupère le prix live Binance à chaque tick/cycle
                    try:
                        ticker = self.binance_client.get_symbol_ticker(
                            symbol=symbol.replace("/", "")
                        )
                        current_price = float(ticker.get("price", 0))
                        pos["current_price"] = current_price
                    except Exception:
                        current_price = safe_float(pos.get("current_price"), 0)
                    amount = safe_float(pos.get("amount"), 0)

                    # Calcul PnL (FIFO ou classique)
                    fifo_pnl_pct, _ = self.get_last_fifo_pnl(symbol)
                    pnl_pct = (
                        fifo_pnl_pct
                        if fifo_pnl_pct is not None
                        else (
                            (current_price - entry_price) / entry_price * 100
                            if entry_price and current_price
                            else 0
                        )
                    )

                    td = self.trade_decisions.get(symbol.replace("/", "").upper(), {})
                    action = td.get("action", "neutral")
                    confidence = td.get("confidence", 0.5)

                    pause_for_pos, pause_reason = is_paused(symbol)
                    # PATCH: pause_blocage = "Oui" UNIQUEMENT si une pause est active
                    if pauses and pause_for_pos:
                        pause_blocage = "Oui"
                        note = f"Trading suspendu ({pause_reason})"
                        reason = "Pause active"
                        decision = "Vente bloquée"
                    elif action == "SELL":
                        pause_blocage = "Non"
                        note = ""
                        reason = "Signal SELL détecté"
                        decision = "Vente prévue au prochain cycle"
                    elif pnl_pct < -5:
                        pause_blocage = "Non"
                        note = "Risque de stop-loss"
                        reason = f"Perte latente {pnl_pct:.1f}%"
                        decision = "Surveillance stop-loss"
                    elif pnl_pct > 7:
                        pause_blocage = "Non"
                        note = "En zone de profit, TP possible"
                        reason = f"Gain latent {pnl_pct:.1f}%"
                        decision = "Surveillance TP"
                    else:
                        pause_blocage = "Non"
                        note = ""
                        reason = f"Position normale ({action})"
                        decision = "Position maintenue"

                    pending.append(
                        {
                            "symbol": symbol,
                            "reason": reason,
                            "decision": decision,
                            "entry_price": safe_float(entry_price),
                            "current_price": safe_float(current_price),
                            "amount": amount,
                            "pnl_pct": pnl_pct,
                            "% Gain/Perte": f"{pnl_pct:.2f}%",
                            "pause_blocage": pause_blocage,
                            "note": note,
                            "confidence": confidence,
                        }
                    )

            # Ajoute ici les autres positions (futures/simu) si besoin, selon ta logique

            # PATCH: DEBUG pour vérifier le résultat
            print("[DEBUG] get_pending_sales generated:", pending)
            # Sauvegarde dans shared_data.json
            self.safe_update_shared_data({"pending_sales": pending}, self.data_file)
            return pending

        except Exception as e:
            print(f"[ERROR] Erreur dans get_pending_sales: {e}")
            return []

    def get_active_pauses(self):
        """
        Retourne la liste des pauses actives : [{"asset": ..., "action": ..., "cycles_left": ..., "type": ...}, ...]
        """
        pauses = []
        # Recupère les pauses du NewsPauseManager
        if hasattr(self, "news_pause_manager") and self.news_pause_manager:
            for item in self.news_pause_manager.get_active_pauses():
                pauses.append(item)
        return pauses

    def enrich_news_symbols(self, news_list):
        """
        Ajoute automatiquement le champ 'symbols' à chaque news si manquant,
        en utilisant le mapping présent dans la config.
        """
        symbol_mapping = self.config["news"]["symbol_mapping"]
        for news in news_list:
            if "symbols" not in news or not news["symbols"]:
                found_symbols = []
                title = news.get("title", "").lower()
                for k, v in symbol_mapping.items():
                    if k in title:
                        found_symbols.append(v)
                if found_symbols:
                    news["symbols"] = found_symbols
                else:
                    news["symbols"] = []
        return news_list

    def load_signal_fusion_params(self):
        path = "config/best_signal_params.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                params = json.load(f)
            print(f"[OPTIM] Pondérations optimisées chargées : {params}")
            return params
        # Valeurs par défaut
        return {
            "tech_weight": 0.6,
            "ia_weight": 0.3,
            "sentiment_weight": 0.1,
            "buy_threshold": 0.2,
            "sell_threshold": -0.2,
            "mm_risk": 0.05,
        }

    def aggregate_timeframe_signals(self, pair, signals_per_tf):
        """
        Fusionne les signaux multi-timeframes pour une paire donnée.
        signals_per_tf : dict { "1m": {"action":..., "confidence":...}, ... }
        Retourne : action globale ('buy', 'sell', 'neutral') et confiance moyenne pondérée.
        """
        # Pondération : TF + importance (ex : plus fort sur 1h, 4h)
        tf_weights = {"1m": 1, "5m": 2, "15m": 3, "1h": 5, "4h": 4, "1d": 2}
        total_weight = 0
        score = 0
        for tf, d in signals_per_tf.items():
            w = tf_weights.get(tf, 1)
            total_weight += w
            if d["action"] == "buy":
                score += w * d.get("confidence", 0.5)
            elif d["action"] == "sell":
                score -= w * d.get("confidence", 0.5)
            # neutral = 0
        if total_weight == 0:
            return "neutral", 0
        avg_score = score / total_weight
        # Seuils ajustables
        if avg_score > 0.2:
            return "buy", avg_score
        elif avg_score < -0.2:
            return "sell", abs(avg_score)
        else:
            return "neutral", abs(avg_score)

    def sync_positions_with_binance(self):
        if self.is_live_trading and self.binance_client:
            account = self.binance_client.get_account()
            positions = {}
            for bal in account["balances"]:
                asset = bal["asset"]
                free = safe_float(bal.get("free", 0))
                if free > 0 and asset not in ("USDC", "USDT"):
                    symbol = f"{asset}/USDC"
                    try:
                        ticker = self.binance_client.get_symbol_ticker(
                            symbol=symbol.replace("/", "")
                        )
                        current_price = safe_float(ticker.get("price"))
                    except Exception:
                        current_price = None

                    # --- PATCH FIFO entry price ---
                    buys, sells = self.fetch_trades_fifo(
                        self.binance_client, symbol.replace("/", "")
                    )
                    last_sell_time = sells[-1]["time"] if sells else None

                    if last_sell_time is not None:
                        buys_since_last_sell = [
                            b for b in buys if b["time"] > last_sell_time
                        ]
                    else:
                        buys_since_last_sell = buys

                    if buys_since_last_sell:
                        total_qty = sum(b["qty"] for b in buys_since_last_sell)
                        entry_price = (
                            sum(b["qty"] * b["price"] for b in buys_since_last_sell)
                            / total_qty
                            if total_qty > 0
                            else None
                        )
                    else:
                        entry_price = None

                    fifo_pnl_pct = self.get_last_fifo_pnl(symbol)
                    prev_pos = self.positions_binance.get(symbol, {})
                    positions[symbol] = {
                        "side": self.positions.get(symbol, {}).get("side", "long"),
                        "amount": free,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "pnl_pct": fifo_pnl_pct,
                        "pnl_usd": (
                            (current_price - entry_price) * free
                            if entry_price and current_price
                            else 0.0
                        ),
                        "value_usd": (
                            free * current_price if free and current_price else 0.0
                        ),
                        "filled_tp_targets": prev_pos.get(
                            "filled_tp_targets", [False, False]
                        ),
                        "price_history": prev_pos.get("price_history", [entry_price]),
                        "max_price": prev_pos.get("max_price", entry_price),
                    }
            self.positions_binance = positions

    def is_short(self, symbol):
        return self.positions.get(symbol, {}).get("side") == "short"

    # Ajoute cette méthode pour savoir si on est long
    def is_long(self, symbol):
        # En mode live, ne regarder QUE la position réelle Binance
        if getattr(self, "is_live_trading", False):
            pos_spot = self.positions_binance.get(symbol)
            return pos_spot and float(pos_spot.get("amount", 0)) > 0
        # En simulation, garder la logique actuelle
        return self.positions.get(symbol, {}).get("side") == "long"

    def get_entry_price(self, symbol):
        return self.positions.get(symbol, {}).get("entry_price")

    def update_pairs(self, new_pairs):
        """
        Met à jour dynamiquement la liste des paires et réinitialise PPO avec le bon input_dim.
        """
        self.pairs_valid = new_pairs
        self._initialize_ai()  # Recrée PPO et l'input_dim pour les nouvelles paires

    def check_short_stop(self, symbol, price: float = None, trailing_pct: float = 0.03):
        """
        Déclenche le stop-loss court et/ou le trailing stop sur une position short BingX.
        - trailing_pct : trailing stop en % (ex: 0.03 = 3%)
        """
        pos = self.positions.get(symbol)
        if not pos or pos.get("side") != "short":
            return False
        entry = pos.get("entry_price")
        if entry is None:
            return False

        # Récupère le prix courant si non fourni
        if price is None:
            try:
                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                ticker = self.bingx_client.fetch_ticker_sync(symbol_bingx)
                price = float(ticker["last"])
            except Exception:
                return False

        # Initialisation du plus bas atteint depuis l'ouverture
        if "min_price" not in pos or pos["min_price"] is None:
            pos["min_price"] = price

        # Màj du plus bas (pour trailing stop)
        if price < pos["min_price"]:
            pos["min_price"] = price

        # Stop-loss court (si perte trop forte = prix monte trop)
        if price > entry * (1 + self.stop_loss_pct):
            self.logger.warning(
                f"[SHORT STOPLOSS] Déclenché sur {symbol}: prix={price} > {entry} + {self.stop_loss_pct*100}%"
            )
            return True

        # Trailing stop (si le prix remonte de X% par rapport au plus bas atteint)
        if price > pos["min_price"] * (1 + trailing_pct):
            self.logger.warning(
                f"[SHORT TRAILING STOP] Déclenché sur {symbol}: prix={price} > min={pos['min_price']} + {trailing_pct*100}%"
            )
            return True

        return False

    def update_pairs_from_config(self):
        self.pairs_valid = self.config["TRADING"]["pairs"]

        self.ws_collector = BufferedWSCollector(
            symbols=[s.replace("/", "").upper() for s in self.pairs_valid],
            timeframes=self.config["TRADING"]["timeframes"],
            maxlen=2000,
        )
        self.env = TradingEnv(
            trading_pairs=self.pairs_valid,
            timeframes=self.config["TRADING"]["timeframes"],
        )
        self._initialize_ai()

    def get_ws_orderbook(self, symbol):
        """
        Récupère le carnet d'ordres (bid/ask) depuis le ws_collector (WebSocket) ou via Binance API REST en fallback.
        - symbol : exemple 'BTCUSDC'
        Retourne : tuple (bid, ask) ou (None, None) si non dispo.
        """
        try:
            # Essai WebSocket
            if hasattr(self, "ws_collector") and self.ws_collector is not None:
                bid, ask = self.ws_collector.get_orderbook(symbol)
                # Si les valeurs existent et sont numériques, retourne-les
                if bid is not None and ask is not None:
                    return float(bid), float(ask)
            # Fallback sur Binance API REST
            if (
                getattr(self, "is_live_trading", False)
                and hasattr(self, "binance_client")
                and self.binance_client is not None
            ):
                try:
                    ob = self.binance_client.get_order_book(symbol=symbol, limit=5)
                    best_bid = float(ob["bids"][0][0]) if ob["bids"] else None
                    best_ask = float(ob["asks"][0][0]) if ob["asks"] else None
                    print("[FALLBACK REST] Carnet d'ordres récupéré via REST Binance.")
                    return best_bid, best_ask
                except Exception as e:
                    self.logger.warning(
                        f"[WS] Erreur récupération orderbook Binance API pour {symbol}: {e}"
                    )
        except Exception as e:
            self.logger.warning(
                f"[WS] Erreur récupération orderbook WS pour {symbol}: {e}"
            )
        return None, None

    async def execute_arbitrage_cross_exchange(self, opportunity, amount):
        """
        Exécute un arbitrage spot cross-exchange réel avec gestion des erreurs, logs et notifications Telegram.
        Args:
            opportunity (dict): dict contenant buy_exchange, sell_exchange, symbol, buy_price, sell_price, etc.
            amount (float): montant à investir (en devise quote, ex USDC)
        """
        try:
            buy_exchange = self.brokers[opportunity["buy_exchange"]]
            sell_exchange = self.brokers[opportunity["sell_exchange"]]
            symbol = opportunity["symbol"]
            base_currency = symbol.split("/")[0]
            quote_currency = symbol.split("/")[1]

            # 1. Vérification du solde disponible
            balance = await buy_exchange.fetch_balance()
            available = balance[quote_currency]["free"]
            if available < amount:
                msg = f"❌ Solde insuffisant sur {opportunity['buy_exchange']} ({available} {quote_currency} < {amount})"
                log_dashboard(msg)
                await self.telegram.send_message(msg)
                return {"status": "error", "step": "balance", "message": msg}

            # 2. Achat sur buy_exchange
            buy_qty = round(amount / opportunity["buy_price"], 6)
            log_dashboard(
                f"🔄 Achat {buy_qty} {base_currency} sur {opportunity['buy_exchange']} @ {opportunity['buy_price']}"
            )
            await self.telegram.send_message(
                f"🔄 Achat {buy_qty} {base_currency} sur {opportunity['buy_exchange']} @ {opportunity['buy_price']}"
            )
            buy_order = await buy_exchange.create_order(
                symbol=symbol, type="market", side="buy", amount=buy_qty
            )
            log_dashboard(f"✅ Ordre d'achat passé: {buy_order}")
            await self.telegram.send_message(
                f"✅ Ordre d'achat passé sur {opportunity['buy_exchange']}: {buy_order.get('id','?')}"
            )

            # 3. Retrait vers sell_exchange
            deposit_address = await sell_exchange.fetch_deposit_address(base_currency)
            withdrawal_fee = self.config["withdrawal_fees"][
                opportunity["buy_exchange"]
            ][base_currency]
            transfer_amount = buy_qty - withdrawal_fee
            if transfer_amount <= 0:
                msg = f"❌ Montant à transférer insuffisant (après frais: {transfer_amount} {base_currency})"
                log_dashboard(msg)
                await self.telegram.send_message(msg)
                return {"status": "error", "step": "withdraw", "message": msg}

            log_dashboard(
                f"🔄 Retrait {transfer_amount} {base_currency} vers {deposit_address['address']} ({opportunity['sell_exchange']})"
            )
            await self.telegram.send_message(
                f"🔄 Retrait {transfer_amount} {base_currency} vers {deposit_address['address']} ({opportunity['sell_exchange']})"
            )
            withdraw_result = await buy_exchange.withdraw(
                code=base_currency,
                amount=transfer_amount,
                address=deposit_address["address"],
            )
            log_dashboard(f"✅ Retrait initié: {withdraw_result}")
            await self.telegram.send_message(
                f"✅ Retrait initié: {withdraw_result.get('id','?')}"
            )

            # 4. Attente confirmation dépôt sur sell_exchange
            poll_interval = 30
            max_wait = 1800
            waited = 0
            while waited < max_wait:
                deposits = await sell_exchange.fetch_deposits(code=base_currency)
                if any(
                    d.get("amount", 0) >= transfer_amount and d.get("status") == "ok"
                    for d in deposits
                ):
                    log_dashboard(
                        f"✅ Dépôt confirmé sur {opportunity['sell_exchange']}"
                    )
                    await self.telegram.send_message(
                        f"✅ Dépôt confirmé sur {opportunity['sell_exchange']}"
                    )
                    break
                await asyncio.sleep(poll_interval)
                waited += poll_interval
            else:
                msg = (
                    f"❌ Timeout confirmation dépôt sur {opportunity['sell_exchange']}"
                )
                log_dashboard(msg)
                await self.telegram.send_message(msg)
                return {"status": "error", "step": "deposit", "message": msg}

            # 5. Vente sur sell_exchange
            log_dashboard(
                f"🔄 Vente {transfer_amount} {base_currency} sur {opportunity['sell_exchange']} @ {opportunity['sell_price']}"
            )
            await self.telegram.send_message(
                f"🔄 Vente {transfer_amount} {base_currency} sur {opportunity['sell_exchange']} @ {opportunity['sell_price']}"
            )
            sell_order = await sell_exchange.create_order(
                symbol=symbol, type="market", side="sell", amount=transfer_amount
            )
            log_dashboard(f"✅ Ordre de vente passé: {sell_order}")
            await self.telegram.send_message(
                f"✅ Ordre de vente passé sur {opportunity['sell_exchange']}: {sell_order.get('id','?')}"
            )

            # 6. Calcul du profit réel
            initial_value = amount
            final_value = sell_order.get(
                "cost", transfer_amount * opportunity["sell_price"]
            )
            profit = final_value - initial_value
            msg = f"💰 Arbitrage terminé sur {symbol}: Profit net {profit:.2f} {quote_currency}"
            log_dashboard(msg)
            await self.telegram.send_message(msg)

            return {
                "status": "success",
                "profit": profit,
                "buy_order": buy_order,
                "sell_order": sell_order,
                "transfer_amount": transfer_amount,
            }

        except Exception as e:
            msg = f"❌ Erreur arbitrage cross-exchange: {str(e)}"
            log_dashboard(msg)
            await self.telegram.send_message(msg)
            return {"status": "error", "step": "exception", "message": str(e)}

    async def test_news_sentiment(self):
        """
        Test manuel du batch d'analyse de sentiment des news.
        Exécute l'analyse Bert/FinBERT sur toutes les news du buffer et affiche le résumé global.
        """
        news = await self.news_analyzer.fetch_all_news()
        results = self.news_analyzer.analyze_sentiment_batch(news)
        summary = self.news_analyzer.get_sentiment_summary()
        print("Sentiment summary:", summary)

    def check_reload_dl_model(self):
        path = "src/models/cnn_lstm_model.pth"
        if os.path.exists(path):
            if self.dl_model is None:
                print(
                    "[ERROR] Modèle IA non initialisé, impossible de charger les poids."
                )
                return
            mtime = os.path.getmtime(path)
            if self.dl_model_last_mtime is None or mtime > self.dl_model_last_mtime:
                self.dl_model.load_weights(path)
                self.ai_enabled = self.dl_model is not None
                self.dl_model_last_mtime = mtime
                print(f"♻️ Nouveau modèle IA chargé automatiquement ({path})")

    async def _news_analysis_loop(self):
        """
        Boucle d’analyse des news avec pause automatique intelligente.
        Déclenche la pause selon sentiment, impact, classification, multi-source, volatilité, etc.
        PATCH: Protection anti-index out of range sur news_data et toutes les listes utilisées.
        Synchronisation forcée des pauses et du tableau pending_sales à chaque boucle.
        """
        log_dashboard("[NEWS] Lancement boucle d'analyse des news…")
        while True:
            try:
                if not self.news_enabled or not self.news_analyzer:
                    await asyncio.sleep(self.news_update_interval)
                    continue

                self.logger.info("Fetching latest news for sentiment analysis")
                news_data = await self.news_analyzer.fetch_all_news()
                news_data = self.enrich_news_symbols(news_data)

                # PATCH: vérifie la liste avant tout accès indexé ou boucle
                if not news_data or not isinstance(news_data, list):
                    log_dashboard(
                        "[NEWS] Aucun élément à analyser dans news_data, skip boucle"
                    )
                    await asyncio.sleep(self.news_update_interval)
                    continue

                sentiment_analysis = {}
                try:
                    sentiment_analysis = await self.news_analyzer.update_analysis()
                except Exception:
                    self.logger.error("Erreur update_analysis", exc_info=True)
                    sentiment_analysis = {}

                sentiment_scores = (
                    sentiment_analysis.get("items", [])
                    if isinstance(sentiment_analysis, dict)
                    else []
                )

                try:
                    await self._update_sentiment_data(sentiment_scores)
                except Exception:
                    pass

                try:
                    await self._save_sentiment_data(sentiment_scores, news_data)
                except Exception as e:
                    self.logger.error(f"Erreur lors de la sauvegarde du sentiment: {e}")

                try:
                    # PATCH: slicing [:5] est safe, mais vérification quand même
                    safe_news = (
                        news_data[:5]
                        if news_data
                        and isinstance(news_data, list)
                        and len(news_data) > 0
                        else []
                    )
                    await self.telegram.send_news_summary(safe_news)
                except Exception:
                    pass

                # PATCH: boucle protégée
                if news_data and isinstance(news_data, list):
                    for news in news_data:
                        pause_decision = self.news_pause_manager.should_pause(
                            news, self.market_data
                        )
                        if pause_decision:
                            self.news_pause_manager.activate_pause(pause_decision)
                            log_dashboard(
                                f"🚨 Pause déclenchée automatique: {pause_decision}"
                            )
                            try:
                                await self.telegram.send_message(
                                    f"🚨 Pause automatique déclenchée\n"
                                    f"Type: {pause_decision.get('type')}\n"
                                    f"Paire: {pause_decision.get('pair', 'Toutes')}\n"
                                    f"Raison: {pause_decision.get('reason')}\n"
                                    f"Durée: {pause_decision.get('duration', 'N/A')} cycles"
                                )
                            except Exception:
                                pass

                try:
                    shared_data = safe_load_shared_data(self.data_file)
                    deep_cast_floats(shared_data)
                    sentiment_data = shared_data.get("sentiment", {})
                    avg_sentiment = float(
                        sentiment_data.get("overall_sentiment", 0) or 0
                    )
                    impact_score = float(sentiment_data.get("impact_score", 0) or 0)
                    major_events = sentiment_data.get("major_events", "")

                    log_dashboard(
                        f"[NEWS] Score sentiment global: {avg_sentiment:.2f} | Impact: {impact_score:.2f} | Événements: {major_events}"
                    )
                except Exception as e:
                    print(
                        f"[NEWS] Impossible d'afficher le score sentiment global: {e}"
                    )

                # PATCH NOUVEAU : Synchronisation des pauses et pending_sales à chaque boucle
                active_pauses = self.news_pause_manager.get_active_pauses()
                self.safe_update_shared_data(
                    {"active_pauses": active_pauses}, self.data_file
                )
                self.get_pending_sales()  # met à jour le tableau dans le JSON partagé

            except Exception as e:
                self.logger.error(f"News analysis error: {e}")

            await asyncio.sleep(self.news_update_interval)

    def get_binance_real_balance(self, asset="USDC"):
        if self.is_live_trading and self.binance_client:
            try:
                balance_info = self.binance_client.get_asset_balance(asset=asset)
                if balance_info:
                    return float(balance_info["free"])
            except Exception as e:
                self.logger.error(f"Erreur récupération balance Binance: {e}")
        return None

    async def detect_arbitrage_opportunities(self, pair=None):
        """
        Détecte les opportunités d'arbitrage cross-quote USDC/USDT.
        Compare par exemple BTC/USDC sur Binance à BTC/USDT sur les autres brokers,
        avec adaptation du format des symboles selon chaque broker.
        """

        def get_broker_symbol(coin, quote, broker):
            if broker == "binance":
                return f"{coin}{quote}"
            elif broker in ["okx", "bingx"]:
                return f"{coin}-{quote}"
            elif broker == "gateio":
                return f"{coin}_{quote}"
            elif broker == "blofin":
                return f"{coin}{quote}"
            else:
                return f"{coin}/{quote}"

        if not self.is_live_trading:
            log_dashboard("[ARBITRAGE] Pas en mode live trading, détection annulée.")
            return []
        log_dashboard("[ARBITRAGE] Démarrage détection arbitrage USDC/USDT…")
        opportunities = []
        pairs_to_check = [pair] if pair else self.pairs_valid
        MIN_PROFIT_THRESHOLD = 0.15
        MIN_VOLUME_USD = 0
        MAX_SPREAD = 0.5

        try:
            for current_pair in pairs_to_check:
                try:
                    coin = current_pair.split("/")[0]
                    binance_symbol = get_broker_symbol(coin, "USDC", "binance")
                    binance_ticker = self.binance_client.get_ticker(
                        symbol=binance_symbol
                    )
                    binance_price = float(binance_ticker.get("lastPrice") or 0)
                    binance_volume = float(binance_ticker.get("volume", 0))

                    exchanges_to_check = [
                        {"name": "okx", "client": self.brokers.get("okx")},
                        {"name": "gateio", "client": self.brokers.get("gateio")},
                        {"name": "blofin", "client": self.brokers.get("blofin")},
                        {"name": "bingx", "client": self.brokers.get("bingx")},
                    ]

                    for exchange in exchanges_to_check:
                        broker = exchange["name"]
                        binance_symbol = get_broker_symbol(coin, "USDC", "binance")
                        other_symbol = get_broker_symbol(coin, "USDT", broker)
                        if not exchange["client"]:
                            continue

                        try:
                            other_symbol = get_broker_symbol(coin, "USDT", broker)
                            ticker = await exchange["client"].fetch_ticker(other_symbol)
                            exchange_price = float(ticker["last"])
                            if not exchange_price or not binance_price:
                                continue

                            price_diff = exchange_price - binance_price
                            profit_pct = (
                                (price_diff / binance_price) * 100
                                if binance_price > 0
                                else 0
                            )

                            if profit_pct > MIN_PROFIT_THRESHOLD:
                                opportunity = {
                                    "pair": coin,
                                    "exchange1": "Binance (USDC)",
                                    "exchange2": f"{broker} (USDT)",
                                    "price1": binance_price,
                                    "price2": exchange_price,
                                    "diff_percent": profit_pct,
                                    "volume_24h": binance_volume * binance_price,
                                    "estimated_profit": profit_pct - 0.2,
                                    "route": f"Buy {coin}/USDC (Binance) -> Transfer {coin} -> Sell {coin}/USDT ({broker})",
                                }
                                log_dashboard(
                                    f"[ARBITRAGE] OPPORTUNITE: {coin}: {binance_price} (Binance USDC) <> {exchange_price} ({broker} USDT) | Diff: {profit_pct:.2f}%"
                                )
                                opportunities.append(opportunity)
                                self.logger.info(
                                    f"Opportunité d'arbitrage cross-quote détectée pour {coin}: {opportunity}"
                                )

                        except Exception as e:
                            print(f"[ARBITRAGE] Erreur sur {broker}: {e}")
                            self.logger.error(f"Erreur sur {broker}: {e}")
                            continue

                except Exception as e:
                    print(
                        f"[ARBITRAGE] Erreur lors du traitement de {current_pair}: {e}"
                    )
                    self.logger.error(
                        f"Erreur lors du traitement de {current_pair}: {e}"
                    )
                    continue

            if opportunities:
                print(
                    f"[ARBITRAGE] {len(opportunities)} opportunités détectées ce cycle."
                )
            else:
                print("[ARBITRAGE] Aucune opportunité détectée ce cycle.")

            return opportunities

        except Exception as e:
            print(f"[ARBITRAGE] Erreur globale détection arbitrage: {e}")
            self.logger.error(f"Erreur globale détection arbitrage: {e}")
            return []

    async def execute_arbitrage(self, opportunity):
        """Exécute une opportunité d'arbitrage"""
        try:
            result = await self.arbitrage_executor.execute(
                opportunity=opportunity,
                max_slippage=0.1,
                timeout=5,
            )

            if result.get("success"):
                profit = float(result.get("realized_profit", 0))
                message = (
                    f"✅ Arbitrage réussi!\n"
                    f"💰 Profit: {profit:.2f} USDT\n"
                    f"📊 Paire: {opportunity['pair']}\n"
                    f"🔄 Route: {opportunity.get('route', '')}"
                )
                await self.telegram.send_message(message)
                self._update_performance_metrics(
                    {"type": "arbitrage", "profit": profit, "pair": opportunity["pair"]}
                )
            else:
                self.logger.warning(f"Échec arbitrage: {result.get('error')}")

        except Exception as e:
            self.logger.error(f"Erreur exécution arbitrage: {e}")

    def secure_withdraw(self, address, amount, asset):
        message = f"{address}|{amount}|{asset}|{get_current_time()}"
        signature = self.key_manager.sign_message(message)
        print(
            f"Retrait sécurisé demandé : {amount} {asset} vers {address}, signature: {signature}"
        )
        return signature

    def _initialize_ai(self):
        """Initialise les composants d'IA et du trading live Binance"""
        try:
            log_dashboard("Initialisation des modèles d'IA...")
            if not self.env:
                raise ValueError("L'environnement de trading n'est pas initialisé")

            # 1. Constantes IA
            self.N_FEATURES = 8
            self.N_STEPS = 63

            # 2. Hyperparams AutoML si dispo
            hp_path = "config/best_hyperparams.json"
            if os.path.exists(hp_path):
                with open(hp_path, "r") as f:
                    best_hp = json.load(f)
                self.config["AI"].update(best_hp)
                print(f"[AI] Hyperparams optimisés chargés depuis {hp_path}: {best_hp}")
            else:
                print(
                    "[AI] Pas d'hyperparams optimisés trouvés, utilisation des valeurs par défaut."
                )

            # 3. Deep Learning Model
            self.dl_model = DeepLearningModel()
            self.dl_model.initialize()
            weights_path = "src/models/cnn_lstm_model.pth"
            if os.path.exists(weights_path):
                self.dl_model.load_weights(weights_path)
                print(f"[DL] Modèle chargé depuis {weights_path}")
            else:
                print(
                    f"[DL WARNING] Aucun modèle entraîné trouvé à {weights_path} ! Prédictions IA non fiables."
                )
            if os.path.exists(weights_path):
                self.dl_model_last_mtime = os.path.getmtime(weights_path)
            else:
                self.dl_model_last_mtime = None
            print(
                f"[DEBUG] paires_valid utilisées IA: {self.pairs_valid} (count={len(self.pairs_valid)})"
            )

            # 4. PPO
            input_dim = self.get_input_dim()
            num_pairs = len(self.pairs_valid)
            env_config = {
                "env": self.env,
                "input_dim": input_dim,
                "learning_rate": self.config["AI"]["learning_rate"],
                "batch_size": self.config["AI"]["batch_size"],
                "n_epochs": self.config["AI"]["n_epochs"],
                "verbose": 1,
            }
            self.ppo_strategy = PPOStrategy(env_config)
            if self.ppo_strategy.model is None:
                raise ValueError("Échec de l'initialisation du modèle PPO")
            self.ai_enabled = True
            log_dashboard("✅ Modèles d'IA initialisés avec succès")
        except Exception as e:
            print(f"❌ Erreur initialisation IA: {str(e)}")
            self.ai_enabled = False
            self.dl_model = None
            self.ppo_strategy = None

        # 5. Telegram & Logger
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.last_telegram_update = datetime.utcnow()
        self.logger = logger

        # 6. Initialisation de l'API Binance (live/simu)
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        if self.api_key and self.api_secret:
            self.binance_client = Client(self.api_key, self.api_secret)
            self.binance_connector = BinanceConnector()
            self.executor = SmartOrderExecutor()
            self.is_live_trading = True
            self.logger.info("Binance API initialized for live trading")
        else:
            self.is_live_trading = False
            self.binance_client = None
            self.binance_connector = None
            self.executor = None
            self.logger.warning(
                "Binance API credentials not found, running in simulation mode"
            )
        print("Vérification des clés API:")
        print(f"API Key présente: {'Oui' if self.api_key else 'Non'}")
        print(f"API Secret présente: {'Oui' if self.api_secret else 'Non'}")
        print(f"[DEBUG] is_live_trading après init: {self.is_live_trading}")

        # 7. PPO (recheck, for redundancy)
        try:
            print("Configuration de la stratégie PPO...")
            N_FEATURES = self.N_FEATURES
            N_STEPS = self.N_STEPS
            num_pairs = len(self.pairs_valid)
            env_config = {
                "env": self.env,
                "input_dim": N_FEATURES * N_STEPS * num_pairs,
                "learning_rate": 3e-4,
                "batch_size": 64,
                "n_epochs": 10,
                "verbose": 1,
            }
            if not hasattr(self.env, "reset") or not hasattr(self.env, "step"):
                raise ValueError("Trading environment missing required methods")
            self.ppo_strategy = PPOStrategy(env_config)
            if self.ppo_strategy.model is None:
                raise ValueError("PPO model failed to initialize")
            log_dashboard("✅ PPO Strategy initialized successfully")
        except Exception as e:
            print(f"❌ Erreur initialisation PPO: {str(e)}")
            self.ppo_strategy = None

        # 8. Sentiment Analyzer
        try:
            self.news_analyzer = NewsSentimentAnalyzer(self.config)
            self.news_enabled = True
            self.dl_model_last_mtime = None
            self.news_weight = 0.2
            self.news_update_interval = 300
            self.logger.info("News sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize news analyzer: {e}")
            self.news_enabled = False
            self.news_analyzer = None

        log_dashboard(f"✅ Bot initialisé avec Telegram: {bool(TELEGRAM_BOT_TOKEN)}")
        log_dashboard(f"✅ Trading en direct: {self.is_live_trading}")
        log_dashboard(f"✅ IA activée: {self.ai_enabled}")
        log_dashboard(f"✅ Analyse de news activée: {self.news_enabled}")

    def calculate_sharpe(self, returns=None):
        """Calcule le ratio de Sharpe"""
        try:
            if returns is None:
                # Récupère l'historique des rendements depuis equity_history
                shared_data = safe_load_shared_data(self.data_file)
                deep_cast_floats(data)
                equity_history = data.get("equity_history", [])
                if not equity_history or len(equity_history) < 2:
                    return 0.0

                balances = [pt["balance"] for pt in equity_history if "balance" in pt]
                if len(balances) < 2:
                    return 0.0

                returns = np.diff(balances) / balances[:-1]

            if len(returns) == 0:
                return 0.0

            # Annualisation (252 jours de trading)
            avg_return = np.mean(returns) * 252
            std_return = np.std(returns) * np.sqrt(252)

            # Risk-free rate (utilise 2% comme taux sans risque)
            risk_free = 0.02

            if std_return == 0:
                return 0.0

            sharpe = (avg_return - risk_free) / std_return
            return float(sharpe)

        except Exception as e:
            print(f"[ERROR] Erreur calcul Sharpe: {e}")
            return 0.0

    def calculate_sortino(self, returns=None):
        """Calcule le ratio de Sortino"""
        try:
            if returns is None:
                # Récupère l'historique des rendements
                shared_data = safe_load_shared_data(self.data_file)
                deep_cast_floats(data)
                equity_history = data.get("equity_history", [])
                if not equity_history or len(equity_history) < 2:
                    return 0.0

                balances = [pt["balance"] for pt in equity_history if "balance" in pt]
                if len(balances) < 2:
                    return 0.0

                returns = np.diff(balances) / balances[:-1]

            if len(returns) == 0:
                return 0.0

            # Annualisation
            avg_return = np.mean(returns) * 252

            # Calcul de la volatilité négative uniquement
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return float("inf") if avg_return > 0 else 0.0

            downside_std = np.std(negative_returns) * np.sqrt(252)

            # Risk-free rate
            risk_free = 0.02

            if downside_std == 0:
                return 0.0

            sortino = (avg_return - risk_free) / downside_std
            return float(sortino)

        except Exception as e:
            print(f"[ERROR] Erreur calcul Sortino: {e}")
            return 0.0

    def calculate_calmar(self):
        """Calcule le ratio de Calmar"""
        try:
            shared_data = safe_load_shared_data(self.data_file)
            deep_cast_floats(data)
            equity_history = data.get("equity_history", [])

            if not equity_history or len(equity_history) < 2:
                return 0.0

            balances = np.array(
                [pt["balance"] for pt in equity_history if "balance" in pt]
            )
            if len(balances) < 2:
                return 0.0

            # Calcul du rendement annualisé
            total_return = (balances[-1] / balances[0]) - 1
            n_days = len(balances)
            annual_return = (1 + total_return) ** (252 / n_days) - 1

            # Calcul du Max Drawdown
            max_dd = self.get_max_drawdown()
            if max_dd == 0:
                return 0.0

            calmar = annual_return / abs(max_dd)
            return float(calmar)

        except Exception as e:
            print(f"[ERROR] Erreur calcul Calmar: {e}")
            return 0.0

    def get_win_rate(self):
        """Calcule le win rate sur l'historique des trades"""
        try:
            shared_data = safe_load_shared_data(self.data_file)
            deep_cast_floats(data)
            trade_history = data.get("trade_history", [])

            if not trade_history:
                return 0.0

            wins = sum(1 for trade in trade_history if trade.get("pnl_usd", 0) > 0)
            return float(wins / len(trade_history)) if trade_history else 0.0

        except Exception as e:
            print(f"[ERROR] Erreur calcul win rate: {e}")
            return 0.0

    def get_avg_profit(self):
        """Calcule le profit moyen par trade"""
        try:
            shared_data = safe_load_shared_data(self.data_file)
            deep_cast_floats(data)
            trade_history = data.get("trade_history", [])

            if not trade_history:
                return 0.0

            profits = [trade.get("pnl_usd", 0) for trade in trade_history]
            return float(np.mean(profits)) if profits else 0.0

        except Exception as e:
            print(f"[ERROR] Erreur calcul profit moyen: {e}")
            return 0.0

    def get_max_drawdown(self):
        """Calcule le maximum drawdown"""
        try:
            shared_data = safe_load_shared_data(self.data_file)
            deep_cast_floats(data)
            equity_history = data.get("equity_history", [])

            if not equity_history or len(equity_history) < 2:
                return 0.0

            balances = np.array(
                [pt["balance"] for pt in equity_history if "balance" in pt]
            )
            if len(balances) < 2:
                return 0.0

            peaks = np.maximum.accumulate(balances)
            drawdowns = (balances - peaks) / peaks
            max_dd = float(np.min(drawdowns))
            return max_dd

        except Exception as e:
            print(f"[ERROR] Erreur calcul max drawdown: {e}")
            return 0.0

    def track_advanced_metrics(self):
        """Suivi des métriques avancées"""
        try:
            metrics = {
                "sharpe_ratio": self.calculate_sharpe(),
                "sortino_ratio": self.calculate_sortino(),
                "calmar_ratio": self.calculate_calmar(),
                "win_rate": self.get_win_rate(),
                "avg_profit": self.get_avg_profit(),
                "max_drawdown": self.get_max_drawdown(),
            }
            return metrics

        except Exception as e:
            print(f"[ERROR] Erreur calcul métriques avancées: {e}")
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "max_drawdown": 0.0,
            }

    def get_input_dim(self):
        return self.N_FEATURES * self.N_STEPS * len(self.pairs_valid)

    async def test_news_sentiment(self):
        """
        Test manuel du batch d'analyse de sentiment des news.
        Exécute l'analyse Bert/FinBERT sur toutes les news du buffer et affiche le résumé global.
        """
        news = await self.news_analyzer.fetch_all_news()
        results = self.news_analyzer.analyze_sentiment_batch(news)
        summary = self.news_analyzer.get_sentiment_summary()
        print("Sentiment summary:", summary)

    def check_stop_loss(self, symbol, price: float = None):
        """
        Stop-loss dynamique basé sur la volatilité (ATR).
        """
        try:
            pos = self.positions.get(symbol)
            if not pos or pos.get("side") != "long":
                return False
            entry = safe_float(pos.get("entry_price"), 0)
            if entry == 0:
                return False

            symbol_ws = symbol.replace("/", "").upper()
            if price is None:
                price = self.ws_collector.get_last_price(symbol_ws)
                if (
                    price is None
                    and symbol_ws in self.market_data
                    and "1h" in self.market_data[symbol_ws]
                ):
                    closes = self.market_data[symbol_ws]["1h"].get("close", [])
                    if closes:
                        price = closes[-1]
            price = safe_float(price, 0)
            if price == 0:
                return False

            # Utilisation de self.calculate_atr au lieu de calculate_atr
            df_ohlcv = pd.DataFrame(self.market_data[symbol_ws]["1h"])
            atr = safe_float(self.calculate_atr(df_ohlcv, period=14), 0)
            dynamic_stop_pct = max(
                0.01, min(safe_float(atr, 0) / entry, 0.10)
            )  # Entre 1% et 10% max

            loss = (price - entry) / entry
            if loss < -dynamic_stop_pct:
                print(
                    f"[STOPLOSS] Déclenché sur {symbol}: perte = {loss:.2%} (ATR dynamique={dynamic_stop_pct:.2%})"
                )
                return True
            return False
        except Exception as e:
            print(f"[STOPLOSS] Erreur vérification stop-loss: {e}")
            return False

    async def execute_trade(
        self, symbol, side, amount, price=None, iceberg=False, iceberg_visible_size=0.1
    ):
        amount = safe_float(amount, 0)
        price = safe_float(price, 0) if price is not None else None

        if not self.is_live_trading:
            log_dashboard(
                f"[ORDER] SIMULATION: {side} {amount} {symbol} @ {price} (iceberg={iceberg})"
            )
            self.logger.info(
                f"SIMULATION: {side} {amount} {symbol} @ {price} (iceberg={iceberg})"
            )
            # Gestion état simulée
            if side.upper() == "BUY":
                if self.is_long(symbol):
                    log_dashboard(
                        f"[ORDER] Déjà long sur {symbol}, achat ignoré (simu)"
                    )
                    return {"status": "skipped", "reason": "already long"}
                self.positions[symbol] = {
                    "side": "long",
                    "entry_price": safe_float(price or 0),
                    "amount": safe_float(amount),
                }
            elif side.upper() == "SELL":
                if not self.is_long(symbol):
                    log_dashboard(
                        f"[ORDER] Pas en position long sur {symbol}, vente ignorée (simu)"
                    )
                    return {"status": "skipped", "reason": "not in position"}
                self.positions.pop(symbol, None)
            elif side.upper() == "SHORT":
                if self.is_short(symbol):
                    log_dashboard(
                        f"[ORDER] Déjà short sur {symbol}, short ignoré (simu)"
                    )
                    return {"status": "skipped", "reason": "already short"}
                self.positions[symbol] = {
                    "side": "short",
                    "entry_price": safe_float(price or 0),
                    "amount": safe_float(amount),
                    "min_price": safe_float(price or 0),
                }
            elif side.upper() == "BUY" and self.is_short(symbol):
                if not self.is_short(symbol):
                    log_dashboard(
                        f"[ORDER] Pas en position short sur {symbol}, rachat ignoré (simu)"
                    )
                    return {"status": "skipped", "reason": "not in short"}
                self.positions.pop(symbol, None)
            return {
                "status": "simulated",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "iceberg": iceberg,
            }

        try:
            log_dashboard(
                f"[ORDER] Tentative d'exécution: {side} {amount} {symbol} (iceberg: {iceberg})"
            )

            # ----- ACHAT SPOT -----
            if side.upper() == "BUY" and symbol.endswith("USDC"):
                if self.is_long(symbol):
                    log_dashboard(f"[ORDER] Déjà long sur {symbol}, achat ignoré.")
                    return {"status": "skipped", "reason": "already long"}
                bid, ask = self.get_ws_orderbook(symbol)
                if bid is None or ask is None:
                    log_dashboard(
                        f"[ORDER] Orderbook WS non dispo pour {symbol}, annulation de l'ordre."
                    )
                    return {"status": "error", "reason": "Orderbook WS not available"}
                orderbook = {"bids": [[bid, 1.0]], "asks": [[ask, 1.0]]}
                recent_trades = []
                market_data = {
                    "recent_trades": recent_trades,
                    "volatility": self.calculate_volatility(
                        self.market_data.get(symbol, {}).get("1h", {})
                    ),
                    "regime": self.regime,
                    "binance_client": self.binance_client,
                }
                result = await self.executor.execute_order(
                    symbol=symbol,
                    side=side,
                    quoteOrderQty=amount,
                    orderbook=orderbook,
                    market_data=market_data,
                    iceberg=iceberg,
                    iceberg_visible_size=iceberg_visible_size,
                )
                if result.get("status") == "completed":
                    self.positions[symbol] = {
                        "side": "long",
                        "entry_price": safe_float(result.get("avg_price", price)),
                        "amount": safe_float(result.get("filled_amount", amount)),
                    }

            # ----- VENTE SPOT -----
            elif side.upper() == "SELL" and symbol.endswith("USDC"):
                allow_sell = False
                use_amount = None
                if self.is_long(symbol):
                    allow_sell = True
                    use_amount = safe_float(self.positions[symbol]["amount"])
                else:
                    asset = symbol.replace("USDC", "")
                    balance = None
                    try:
                        balance = self.binance_client.get_asset_balance(asset=asset)
                    except Exception as e:
                        log_dashboard(
                            f"[ORDER] Erreur récupération balance {asset}: {e}"
                        )
                    if balance and safe_float(balance.get("free", 0)) >= amount:
                        allow_sell = True
                        use_amount = safe_float(amount)
                        log_dashboard(
                            f"[ORDER] Vente autorisée sur solde réel {asset}: {balance['free']}"
                        )
                    else:
                        log_dashboard(
                            f"[ORDER] Pas en position long ni de solde suffisant sur {symbol}, vente ignorée."
                        )
                        return {
                            "status": "skipped",
                            "reason": "not in position or insufficient balance",
                        }

                bid, ask = self.get_ws_orderbook(symbol)
                if bid is None or ask is None:
                    log_dashboard(
                        f"[ORDER] Orderbook WS non dispo pour {symbol}, annulation de l'ordre."
                    )
                    return {"status": "error", "reason": "Orderbook WS not available"}
                orderbook = {"bids": [[bid, 1.0]], "asks": [[ask, 1.0]]}
                market_data = {
                    "recent_trades": [],
                    "volatility": self.calculate_volatility(
                        self.market_data.get(symbol, {}).get("1h", {})
                    ),
                    "regime": self.regime,
                    "binance_client": self.binance_client,
                }
                result = await self.executor.execute_order(
                    symbol=symbol,
                    side=side,
                    quoteOrderQty=use_amount,
                    orderbook=orderbook,
                    market_data=market_data,
                    iceberg=iceberg,
                    iceberg_visible_size=iceberg_visible_size,
                )
                if result.get("status") == "completed" and self.is_long(symbol):
                    self.positions.pop(symbol, None)

            # ----- OUVERTURE SHORT BINGX -----
            elif side.upper() == "SHORT":
                if self.is_short(symbol):
                    log_dashboard(f"[ORDER] Déjà short sur {symbol}, short ignoré.")
                    return {"status": "skipped", "reason": "already short"}
                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                ticker = await self.bingx_client.fetch_ticker(symbol_bingx)
                price_bingx = safe_float(ticker["last"])
                qty = safe_float(amount) / price_bingx if price_bingx > 0 else 0
                result = await self.bingx_executor.short_order(
                    symbol_bingx, qty, leverage=3
                )
                if result.get("status") == "completed":
                    self.positions[symbol] = {
                        "side": "short",
                        "entry_price": price_bingx,
                        "amount": qty,
                        "min_price": price_bingx,
                    }

            # ----- FERMETURE SHORT BINGX -----
            elif side.upper() == "BUY" and self.is_short(symbol):
                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                pos = self.positions[symbol]
                qty = safe_float(pos["amount"])
                result = await self.bingx_executor.close_short_order(symbol_bingx, qty)
                if result.get("status") == "completed":
                    self.positions.pop(symbol, None)

            else:
                return {"status": "rejected", "reason": "unsupported side"}

            if result.get("status") == "completed":
                log_dashboard(
                    f"[ORDER] Exécuté avec succès: {side} {result.get('filled_amount', amount)} {symbol} @ {result.get('avg_price', price)}"
                )
                self.logger.info(
                    f"Order executed: {side} {result.get('filled_amount', amount)} {symbol} @ {result.get('avg_price', price)}"
                )
                self._update_performance_metrics(result)
                iceberg_info = (
                    f"\n🧊 <b>Ordre Iceberg</b> ({result.get('n_suborders', '')} sous-ordres)"
                    if result.get("iceberg")
                    else ""
                )
                await self.telegram.send_message(
                    f"💰 <b>Ordre exécuté</b>\n"
                    f"📊 {side} {result.get('filled_amount', amount)} {symbol} @ {result.get('avg_price', price)}\n"
                    f"💵 Total: ${safe_float(result.get('filled_amount', amount)) * safe_float(result.get('avg_price', price) or 0):.2f}"
                    f"{iceberg_info}"
                )
            else:
                print(f"[ORDER] Echec d'exécution: {side} {amount} {symbol}")

            return result

        except BinanceAPIException as e:
            print(f"[ORDER] Binance API error: {e}")
            self.logger.error(f"Binance API error: {e}")
            await self.telegram.send_message(f"⚠️ Erreur API Binance: {e}")
            return {"status": "error", "reason": str(e)}
        except Exception as e:
            print(f"[ORDER] Execution error: {e}")
            self.logger.error(f"Execution error: {e}")
            return {"status": "error", "reason": str(e)}

    async def plan_auto_sell(
        self,
        symbol,
        entry_price,
        amount,
        tp_pct=0.03,
        sl_pct=0.03,
        max_cycles=2,
        reason="",
    ):
        """
        Planifie une vente automatique pour une position ouverte via signal pump/breakout/news/arbitrage.
        Enregistre la raison de l'achat pour le dashboard.
        """
        auto_sell_list = []
        try:
            shared_data = safe_load_shared_data(self.data_file)
            auto_sell_list = shared_data.get("auto_sell_positions", [])
        except Exception:
            auto_sell_list = []

        auto_sell_list.append(
            {
                "symbol": symbol,
                "entry_price": entry_price,
                "amount": amount,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "cycle_open": self.current_cycle,
                "max_cycles": max_cycles,
                "reason": reason,
            }
        )
        self.safe_update_shared_data(
            {"auto_sell_positions": auto_sell_list}, self.data_file
        )

    async def handle_auto_sell(self):
        """
        Vérifie toutes les positions auto_sell et exécute la vente si TP/SL/durée atteint.
        Appel à chaque cycle.
        """
        auto_sell_list = []
        try:
            shared_data = safe_load_shared_data(self.data_file)
            auto_sell_list = shared_data.get("auto_sell_positions", [])
        except Exception:
            auto_sell_list = []

        updated_list = []
        for pos in auto_sell_list:
            symbol = pos["symbol"]

            # Récupère le prix courant live depuis Binance
            try:
                ticker = self.binance_client.get_symbol_ticker(
                    symbol=symbol.replace("/", "")
                )
                current_price = float(ticker.get("price", 0))
            except Exception:
                # Fallback: utilise le WS collector si Binance échoue
                try:
                    current_price = safe_float(
                        self.ws_collector.get_last_price(symbol), 0
                    )
                except Exception:
                    current_price = None

            entry = safe_float(pos["entry_price"], 0)
            amount = safe_float(pos["amount"], 0)
            tp_pct = safe_float(pos.get("tp_pct", 0.03))
            sl_pct = safe_float(pos.get("sl_pct", 0.03))
            cycle_open = int(pos.get("cycle_open", 0))
            max_cycles = int(pos.get("max_cycles", 2))

            # Calcul TP/SL/durée
            if current_price and entry:
                gain = (current_price - entry) / entry
                if gain >= tp_pct:
                    await self.execute_trade(symbol, "SELL", amount)
                    log_dashboard(
                        f"[AUTO-SELL] Vente auto sur {symbol} : TP/SL/durée atteint."
                    )
                    continue  # Ne conserve pas cette position dans la liste

            # PATCH: Ajoute le prix courant dans la position pour affichage dashboard
            pos["current_price"] = current_price
            updated_list.append(pos)

        self.safe_update_shared_data(
            {"auto_sell_positions": updated_list}, self.data_file
        )

    def _update_performance_metrics(self, trade_result):
        try:
            if not trade_result or "status" not in trade_result:
                return

            perf = self.get_performance_metrics()

            # PATCH GLOBAL : force le cast sur tous les champs utilisés
            for key in [
                "balance",
                "total_trades",
                "wins",
                "losses",
                "total_profit",
                "total_loss",
                "win_rate",
                "profit_factor",
            ]:
                perf[key] = safe_float(perf.get(key), 0)

            # -------- PATCH ABSOLU : protection anti-str ---------
            # On vérifie que les champs sont bien en float juste avant toute addition
            perf["total_trades"] = safe_float(perf.get("total_trades"), 0) + 1

            # Calcul P&L
            if trade_result["status"] == "completed":
                amount = safe_float(trade_result.get("filled_amount", 0))
                price = safe_float(trade_result.get("avg_price", 0))
                side = trade_result.get("side", "")

                if side.upper() == "SELL":
                    entry = safe_float(trade_result.get("entry_price", 0))
                    if entry > 0:
                        pnl = (price - entry) * amount

                        if pnl > 0:
                            perf["wins"] = safe_float(perf.get("wins"), 0) + 1
                        else:
                            perf["losses"] = safe_float(perf.get("losses"), 0) + 1

                        # PATCH: type-safe division
                        total_trades = safe_float(perf.get("total_trades"), 0)
                        wins = safe_float(perf.get("wins"), 0)
                        perf["win_rate"] = (
                            wins / total_trades if total_trades > 0 else 0
                        )

                        # PATCH : addition type-safe
                        perf["total_profit"] = safe_float(
                            perf.get("total_profit"), 0
                        ) + max(0, pnl)
                        perf["total_loss"] = safe_float(
                            perf.get("total_loss"), 0
                        ) + max(0, -pnl)

                        total_loss = safe_float(perf.get("total_loss"), 0)
                        total_profit = safe_float(perf.get("total_profit"), 0)
                        if total_loss > 0:
                            perf["profit_factor"] = total_profit / total_loss

                        perf["balance"] = safe_float(perf.get("balance"), 0) + pnl

            self.safe_update_shared_data(
                {"bot_status": {"performance": perf}}, self.data_file
            )

        except Exception as e:
            self.logger.error(f"Erreur update métriques: {e}")

    async def _update_sentiment_data(self, sentiment_scores):
        """Met à jour les données de sentiment avec validation"""
        try:
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if pair_key not in self.market_data:
                    self.market_data[pair_key] = {}

                current_sentiment = safe_float(
                    self.market_data[pair_key].get("sentiment", 0.0)
                )
                new_sentiment = 0.0
                matching_scores = [
                    safe_float(score.get("sentiment", 0.0))
                    for score in sentiment_scores
                    if isinstance(score, dict)
                    and "sentiment" in score
                    and score["sentiment"] is not None
                    and (
                        not score.get("symbols")
                        or pair_key in [s.upper() for s in score.get("symbols", [])]
                    )
                ]
                if matching_scores:
                    new_sentiment = sum(matching_scores) / len(matching_scores)
                self.market_data[pair_key]["sentiment"] = new_sentiment
                self.market_data[pair_key][
                    "sentiment_timestamp"
                ] = datetime.utcnow().timestamp()

                print(f"[SENTIMENT] {pair_key}: {new_sentiment:.4f}")

        except Exception as e:
            print(f"❌ Erreur mise à jour sentiment: {e}")

    async def _save_sentiment_data(self, sentiment_scores, news_data=None):
        """
        Enregistre les données de sentiment du marché (scores, news, global) dans le fichier partagé.
        Correction : merge les news pour préserver le champ 'processed' à chaque sauvegarde.
        PATCH: Toutes les listes indexées sont protégées.
        """
        headlines = []
        if news_data is None:
            news_data = sentiment_scores
        if isinstance(news_data, list):
            for item in news_data[:10]:
                if isinstance(item, dict) and "title" in item:
                    headlines.append(str(item["title"]))

        valid_scores = [
            safe_float(data.get("sentiment"))
            for key, data in self.market_data.items()
            if data.get("sentiment") is not None
        ]
        print(
            f"[DEBUG _save_sentiment_data] valid_scores from market_data={valid_scores}"
        )

        if not valid_scores:
            valid_scores = [
                safe_float(item.get("sentiment"))
                for item in sentiment_scores
                if isinstance(item, dict) and item.get("sentiment") is not None
            ]
            print(
                f"[DEBUG _save_sentiment_data] fallback valid_scores from sentiment_scores={valid_scores}"
            )

        summary = get_sentiment_summary_from_batch(sentiment_scores)
        sentiment_global = safe_float(summary.get("sentiment_global", 0.0), 0.0)
        impact_score = float(
            np.mean(
                [
                    abs(safe_float(item.get("sentiment", 0)))
                    for item in sentiment_scores
                    if isinstance(item, dict)
                ]
            )
            if sentiment_scores
            else 0.0
        )
        top_news = summary.get("top_news", [])
        # PATCH: slicing est safe, mais accès direct doit être protégé
        major_events = (
            "; ".join(top_news[:3]) if top_news and len(top_news) >= 1 else "Aucun"
        )

        print(
            f"[DEBUG SENTIMENT GLOBAL] sentiment_global={sentiment_global} impact={impact_score} major_events={major_events}"
        )

        sentiment_data = {
            "timestamp": datetime.now().isoformat(),
            "scores": sentiment_scores,
            "latest_news": summary.get("top_news", []),
            "overall_sentiment": sentiment_global,
            "impact_score": impact_score,
            "major_events": major_events,
            "top_symbols": summary.get("top_symbols", []),
            "n_news": summary.get("n_news", 0),
        }

        try:
            shared_data_prev = safe_load_shared_data(self.data_file)
            deep_cast_floats(shared_data_prev)
            old_scores = shared_data_prev.get("sentiment", {}).get("scores", [])
        except Exception:
            old_scores = []

        sentiment_data["scores"] = merge_news_processed(
            old_scores, sentiment_data["scores"]
        )

        try:
            self.safe_update_shared_data({"sentiment": sentiment_data}, self.data_file)
            self.logger.info(
                f"[SENTIMENT] Data written successfully to {self.data_file}"
            )
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")

    async def generate_market_analysis_report(self, cycle=None):
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        debug_market_data_structure(
            self.market_data, self.pairs_valid, ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
        report = (
            f"⏰ {current_time}\n"
            f"📊 RAPPORT DE MARCHÉ (Cycle {cycle})\n"
            f"Current User's Login: {CURRENT_USER}\n"
            "╔═════════════════════════════════════════════════╗\n"
            "║           RAPPORT D'ANALYSE DE MARCHÉ           ║\n"
            "╠═════════════════════════════════════════════════╣\n"
            f"║ Régime: {self.regime}                               ║\n"
            "╚═════════════════════════════════════════════════╝\n\n"
            "    📊 Analyse par Timeframe/Paire :\n"
        )
        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                if pair_key not in self.market_data or tf not in self.market_data.get(
                    pair_key, {}
                ):
                    print(f"ABSENT: {pair_key} {tf}")

        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        for tf in timeframes:
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                report += f"""
        🕒 {tf} | {pair} :
        ├─ 📈 Tendance: {self.get_trend_analysis(pair, tf)}
        ├─ 📊 Volatilité: {self.get_volatility_analysis(pair, tf)}
        ├─ 📉 Volume: {self.get_volume_analysis(pair, tf)}
        └─ 🎯 Signal dominant: {self.get_dominant_signal(pair, tf)}
        """

        # Ajout des informations d'IA si disponibles
        if self.ai_enabled:
            report += "\n    🧠 Analyse IA :\n"
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if (
                    pair_key in self.market_data
                    and "ai_prediction" in self.market_data[pair_key]
                ):
                    ai_score = safe_float(
                        self.market_data[pair_key]["ai_prediction"], 0
                    )
                    ai_signal = (
                        "ACHAT"
                        if ai_score > 0.6
                        else "VENTE" if ai_score < 0.4 else "NEUTRE"
                    )
                    report += f"""
        🤖 {pair} :
        └─ Prédiction: {ai_signal} ({ai_score:.2f})
        """

        # Section news/sentiment globale détaillée
        try:
            shared_data = safe_load_shared_data(self.data_file)
            deep_cast_floats(shared_data)
            news_sentiment = shared_data.get("sentiment", None)
        except Exception:
            news_sentiment = None

        if news_sentiment and isinstance(news_sentiment, dict):
            try:
                sentiment = safe_float(news_sentiment.get("overall_sentiment", 0), 0)
            except Exception:
                sentiment = 0.0
            try:
                impact = safe_float(news_sentiment.get("impact_score", 0), 0)
            except Exception:
                impact = 0.0
            major_events = news_sentiment.get("major_events", "Aucun")
            report += (
                "\n📰 Analyse des News:\n"
                f"Sentiment: {sentiment:.2%}\n"
                f"Impact estimé: {impact:.2%}\n"
                f"Événements majeurs: {major_events}\n"
            )
            major_news = news_sentiment.get("latest_news", [])
            if major_news and len(major_news) > 0:
                report += "Dernières news :\n"
                for news in major_news[:3]:
                    report += f"- {news}\n"
            else:
                report += "Aucune news disponible.\n"
        else:
            report += "\n📰 Analyse des News: Aucune donnée disponible.\n"

        if self.news_enabled:
            report += "\n    📰 Analyse de Sentiment :\n"
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if (
                    pair_key in self.market_data
                    and "sentiment" in self.market_data[pair_key]
                ):
                    sentiment_score = safe_float(
                        self.market_data[pair_key]["sentiment"], 0
                    )
                    sentiment_type = (
                        "Positif"
                        if sentiment_score > 0.2
                        else "Négatif" if sentiment_score < -0.2 else "Neutre"
                    )
                    report += f"""
        📊 {pair} :
        └─ Sentiment: {sentiment_type} ({sentiment_score:.2f})
        """

        return report

    def calculate_trend(self, data):
        try:
            closes = [
                c for c in data.get("close", []) if c is not None and not np.isnan(c)
            ]
            if len(closes) < 10:
                return 0.0
            closes = closes[-20:]
            if len(closes) < 10:
                return 0.0
            ma_fast = np.mean(closes[-10:])
            ma_slow = np.mean(closes)
            if ma_slow == 0 or np.isnan(ma_fast) or np.isnan(ma_slow):
                return 0.0
            trend = (ma_fast / ma_slow) - 1
            return float(trend)
        except Exception as e:
            print("DEBUG calculate_trend error:", e)
            return 0.0

    def calculate_volatility(self, data):
        try:
            closes = [
                c
                for c in data.get("close", [])
                if c is not None and not np.isnan(c) and c > 0
            ]
            if len(closes) < 2:
                return 0.0
            closes = closes[-20:]
            if len(closes) < 2 or any(c <= 0 for c in closes):
                return 0.0  # protège contre log(0) ou log négatif
            returns = np.diff(np.log(closes))
            if np.isnan(returns).any() or np.isinf(returns).any():
                return 0.0
            return float(np.std(returns) * np.sqrt(252))
        except Exception as e:
            print("DEBUG calculate_volatility error:", e)
            return 0.0

    def calculate_volatility_advanced(self, df, window=20):
        """Calcul de volatilité avancé avec prise en compte des gaps"""
        try:
            if not isinstance(df, pd.DataFrame):
                return 0.02

            if "close" not in df.columns:
                return 0.02

            returns = np.log(df["close"] / df["close"].shift(1))
            # Détection et gestion des gaps
            gaps = df["low"] > df["high"].shift(1)  # Gap haussier
            gaps |= df["high"] < df["low"].shift(1)  # Gap baissier

            # Ajustement volatilité selon gaps
            vol = returns.std() * np.sqrt(252)
            if gaps.any():
                vol *= 1.2  # Augmentation si présence de gaps

            return float(vol)
        except Exception as e:
            print(f"Erreur calcul volatilité avancée: {e}")
            return 0.02

    def calculate_volume_profile(self, data):
        try:
            if isinstance(data, dict) and "volume" in data:
                volumes = data["volume"][-20:]
                if not volumes or len(volumes) < 2:
                    return {"strength": 1.0}
                current_vol = volumes[-1]
                avg_vol = sum(volumes) / len(volumes)
                return {
                    "strength": float(current_vol / avg_vol) if avg_vol > 0 else 1.0
                }
            return {"strength": 1.0}
        except Exception as e:
            print("DEBUG calculate_volume_profile error:", e)
            return {"strength": 1.0}

    def get_trend_analysis(self, pair, timeframe):
        try:
            pair_key = pair.replace("/", "").upper()
            if pair_key in self.market_data and timeframe in self.market_data[pair_key]:
                trend = self.calculate_trend(self.market_data[pair_key][timeframe])
                if trend > 0.02:
                    return "Haussière"
                elif trend < -0.02:
                    return "Baissière"
                return "Neutre"
            return "N/A"
        except Exception as e:
            return "N/A"

    def get_volatility_analysis(self, pair, timeframe):
        """Analyse de volatilité détaillée"""
        try:
            pair_key = pair.replace("/", "").upper()
            if pair_key in self.market_data and timeframe in self.market_data[pair_key]:
                vol = self.calculate_volatility(self.market_data[pair_key][timeframe])
                if vol > 0.8:
                    return "Élevée"
                elif vol > 0.4:
                    return "Moyenne"
                return "Faible"
            return "N/A"
        except Exception as e:
            print(f"DEBUG get_volatility_analysis error: {e}")
            return "N/A"

    def get_volume_analysis(self, pair, timeframe):
        """Analyse du volume"""
        try:
            pair_key = pair.replace("/", "").upper()
            if pair_key in self.market_data and timeframe in self.market_data[pair_key]:
                data = self.market_data[pair_key][timeframe]
                if (
                    data
                    and "volume" in data
                    and isinstance(data["volume"], list)
                    and len(data["volume"]) >= 2
                ):
                    vol_dict = self.calculate_volume_profile(data)
                    # Sécurisation : toujours prendre la clé 'strength' si c'est un dict
                    if isinstance(vol_dict, dict):
                        vol = vol_dict.get("strength", 1.0)
                    else:
                        vol = vol_dict  # fallback : float direct si jamais
                    if vol > 1.5:
                        return "Fort"
                    elif vol > 0.7:
                        return "Moyen"
                    return "Faible"
                else:
                    return "N/A"
            return "N/A"
        except Exception as e:
            print(f"DEBUG get_volume_analysis error: {e}")
            return "N/A"

    def get_dominant_signal(self, pair, timeframe):
        """Signal dominant"""
        try:
            trend = self.get_trend_analysis(pair, timeframe)
            vol = self.get_volatility_analysis(pair, timeframe)
            volume = self.get_volume_analysis(pair, timeframe)
            if trend == "Haussière" and vol != "Élevée" and volume != "Faible":
                return "LONG"
            elif trend == "Baissière" and vol != "Élevée" and volume != "Faible":
                return "SHORT"
            elif vol == "Élevée" or volume == "Faible":
                return "ATTENTE"
            return "NEUTRE"
        except Exception as e:
            print(f"DEBUG get_dominant_signal error: {e}")
            return "N/A"

    async def study_market(self, timeframe):
        """Analyse le marché"""
        try:
            await asyncio.sleep(0.5)  # Simule le temps de calcul

            # Récupération des données de marché
            if self.is_live_trading:
                # Utilisation de l'API Binance pour les données réelles
                await self._fetch_real_market_data()
            else:
                # Utilisation de données simulées
                self.market_data = await self.get_latest_data()

            # Analyse du régime global
            volatility = self.calculate_volatility(
                self.market_data.get("BTCUSDC", {}).get("1h", {})
            )
            trend = self.calculate_trend(
                self.market_data.get("BTCUSDC", {}).get("1h", {})
            )

            if volatility > 0.8:
                self.regime = MARKET_REGIMES["VOLATILE"]
            elif trend > 0.02:
                self.regime = MARKET_REGIMES["TRENDING_UP"]
            elif trend < -0.02:
                self.regime = MARKET_REGIMES["TRENDING_DOWN"]
            else:
                self.regime = MARKET_REGIMES["RANGING"]

            # Si l'IA est activée, ajoutez les prédictions de l'IA
            if self.ai_enabled:
                await self._add_ai_predictions()

            log_dashboard(
                f"[MARKET ANALYSIS] Régime détecté: {self.regime} | Volatilité: {volatility:.4f} | Tendance: {trend:.4f}"
            )

            return self.regime, self.market_data, {}
        except Exception as e:
            self.logger.error(f"Erreur analyse marché: {e}")
            return self.regime, None, {}

    async def _fetch_real_market_data(self):
        """Récupère les données de marché réelles depuis Binance"""
        try:
            if not self.is_live_trading or not self.binance_client:
                return

            timeframes = {
                "1m": Client.KLINE_INTERVAL_1MINUTE,
                "5m": Client.KLINE_INTERVAL_5MINUTE,
                "15m": Client.KLINE_INTERVAL_15MINUTE,
                "1h": Client.KLINE_INTERVAL_1HOUR,
                "4h": Client.KLINE_INTERVAL_4HOUR,
                "1d": Client.KLINE_INTERVAL_1DAY,
            }

            market_data = {}

            for pair in self.pairs_valid:
                pair_binance = pair.replace("/", "")
                market_data[pair_binance] = {}

                for tf_name, tf_value in timeframes.items():
                    try:
                        # Récupération des données historiques
                        klines = self.binance_client.get_klines(
                            symbol=pair_binance, interval=tf_value, limit=100
                        )

                        # Conversion au format OHLCV
                        ohlcv = {
                            "open": [float(k[1]) for k in klines],
                            "high": [float(k[2]) for k in klines],
                            "low": [float(k[3]) for k in klines],
                            "close": [float(k[4]) for k in klines],
                            "volume": [float(k[5]) for k in klines],
                            "timestamp": [int(k[0]) for k in klines],
                        }

                        market_data[pair_binance][tf_name] = ohlcv

                    except BinanceAPIException as e:
                        self.logger.error(
                            f"Binance API error for {pair} {tf_name}: {e}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error fetching data for {pair} {tf_name}: {e}"
                        )

            self.market_data = market_data

        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")

    async def _add_ai_predictions(self):
        """Ajoute les prédictions IA aux données de marché avec gestion améliorée des erreurs"""
        if not self.ai_enabled or not self.dl_model:
            print("❌ IA désactivée ou modèle non initialisé")
            return

        print("\n=== AJOUT PRÉDICTIONS IA ===")

        dl_predictions = {}
        ppo_features_list = []

        # 1. Prédictions CNN-LSTM pour chaque paire
        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            print(f"\n[DEBUG] Préparation features IA pour {pair_key}")

            # Initialisation/vérification structure
            if pair_key not in self.market_data:
                self.market_data[pair_key] = {}

            # Préparation des features
            features = await self._prepare_features_for_ai(pair_key)
            if features is None:
                print(f"❌ Échec préparation features pour {pair_key}")
                continue

            print(f"✅ Features prêtes pour {pair_key}")

            try:
                # Prédiction CNN-LSTM
                dl_prediction = self.dl_model.predict(features)
                dl_predictions[pair_key] = dl_prediction
                self.market_data[pair_key]["ai_prediction"] = dl_prediction
                print(f"✅ Prédiction CNN-LSTM pour {pair_key}: {dl_prediction:.4f}")

                # Préparation des features PPO
                feature_keys = [
                    "close",
                    "high",
                    "low",
                    "volume",
                    "rsi",
                    "macd",
                    "volatility",
                    "vol_ratio",
                ]

                vec = np.concatenate(
                    [
                        (
                            features[k]
                            if isinstance(features[k], np.ndarray)
                            else np.full(self.N_STEPS, features[k])
                        )
                        for k in feature_keys
                    ]
                )

                if vec.shape != (self.N_FEATURES * self.N_STEPS,):
                    print(f"❌ Mauvais shape features PPO pour {pair_key}")
                    continue

                ppo_features_list.append(vec)

            except Exception as e:
                print(f"❌ Erreur prédiction IA pour {pair_key}: {e}")
                continue

        # 2. Prédiction PPO globale
        if ppo_features_list:
            try:
                ppo_features = np.concatenate(ppo_features_list)
                expected_shape = (
                    self.N_FEATURES * self.N_STEPS * len(self.pairs_valid),
                )

                if ppo_features.shape == expected_shape:
                    raw_action = self.ppo_strategy.get_action(ppo_features)

                    # Conversion intelligente de l'action
                    if isinstance(raw_action, dict):
                        ppo_action = float(raw_action.get("value", 0.5))
                    elif isinstance(raw_action, str):
                        # Mapping des actions textuelles
                        action_map = {"BUY": 1.0, "SELL": 0.0, "HOLD": 0.5}
                        ppo_action = action_map.get(raw_action, 0.5)
                    else:
                        ppo_action = float(raw_action)

                    print(f"✅ Prédiction PPO globale: {ppo_action:.4f}")

                    # Application des signaux fusionnés
                    for pair in self.pairs_valid:
                        pair_key = pair.replace("/", "").upper()
                        if pair_key not in self.market_data:
                            self.market_data[pair_key] = {}
                        dl_pred = dl_predictions.get(pair_key, 0.5)
                        await self._merge_signals(pair_key, dl_pred, ppo_action)
                else:
                    print(
                        f"❌ Shape PPO incorrect: {ppo_features.shape}, attendu: {expected_shape}"
                    )

            except Exception as e:
                print(f"❌ Erreur prédiction PPO: {str(e)}")
                # Utilisation des prédictions CNN-LSTM seules en cas d'erreur PPO
                for pair in self.pairs_valid:
                    pair_key = pair.replace("/", "").upper()
                    if pair_key in dl_predictions:
                        await self._merge_signals(
                            pair_key, dl_predictions[pair_key], 0.5
                        )

    async def study_market_period(self, symbol, start_time, end_time, timeframe="1h"):
        """Étudie le marché sur une période définie et établit un plan de trading"""
        try:
            # Convertir les dates en timestamps (ms)
            start_ts = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_time, "%Y-%m-%d").timestamp() * 1000)

            # Récupérer les données historiques
            tf_binance = getattr(Client, f"KLINE_INTERVAL_{timeframe.upper()}")
            klines = self.binance_client.get_historical_klines(
                symbol=symbol, interval=tf_binance, start_str=start_ts, end_str=end_ts
            )

            # Convertir en DataFrame
            ohlcv = {
                "open": [float(k[1]) for k in klines],
                "high": [float(k[2]) for k in klines],
                "low": [float(k[3]) for k in klines],
                "close": [float(k[4]) for k in klines],
                "volume": [float(k[5]) for k in klines],
                "timestamp": [int(k[0]) for k in klines],
            }

            # Analyser les données
            trend = self.calculate_trend(ohlcv)
            volatility = self.calculate_volatility(ohlcv)
            volume_profile = self.calculate_volume_profile(ohlcv)

            # Identifier les régimes de marché
            if volatility > 0.8:
                regime = MARKET_REGIMES["VOLATILE"]
                strategy = "Protection du capital - trades limités, stop-loss étroits"
            elif trend > 0.02:
                regime = MARKET_REGIMES["TRENDING_UP"]
                strategy = "Suivre la tendance - positions longues, trailing stop"
            elif trend < -0.02:
                regime = MARKET_REGIMES["TRENDING_DOWN"]
                strategy = "Ventes courtes ou attente - protection des positions"
            else:
                regime = MARKET_REGIMES["RANGING"]
                strategy = "Range trading - achats aux supports, ventes aux résistances"

            # Préparer le rapport d'analyse
            analysis_report = {
                "symbol": symbol,
                "period": f"{start_time} à {end_time}",
                "timeframe": timeframe,
                "data_points": len(klines),
                "regime": regime,
                "trend": trend,
                "volatility": volatility,
                "volume_profile": volume_profile,
                "strategy": strategy,
                "key_levels": self._identify_key_levels(ohlcv),
            }

            # Envoyer le rapport sur Telegram
            report_message = (
                f"📊 <b>Analyse de Marché: {symbol}</b>\n\n"
                f"⏱️ Période: {start_time} à {end_time}\n"
                f"📈 Régime: {regime}\n"
                f"🔍 Tendance: {trend:.2%}\n"
                f"📏 Volatilité: {volatility:.2%}\n\n"
                f"🎯 <b>Stratégie recommandée:</b>\n{strategy}\n\n"
                f"🔑 <b>Niveaux clés:</b>\n"
            )

            for level in analysis_report["key_levels"][:3]:
                report_message += f"- {level['type']}: {level['price']:.2f}\n"

            await self.telegram.send_message(report_message)

            return analysis_report

        except Exception as e:
            self.logger.error(f"Error studying market period: {e}")
            return None

    def _identify_key_levels(self, ohlcv):
        """Identifie les niveaux clés (support/résistance) dans les données"""
        levels = []

        try:
            highs = np.array(ohlcv["high"])
            lows = np.array(ohlcv["low"])
            closes = np.array(ohlcv["close"])

            # Identifier les sommets locaux (résistances potentielles)
            for i in range(2, len(highs) - 2):
                if (
                    highs[i] > highs[i - 1]
                    and highs[i] > highs[i - 2]
                    and highs[i] > highs[i + 1]
                    and highs[i] > highs[i + 2]
                ):
                    levels.append(
                        {"price": highs[i], "type": "Résistance", "strength": 1}
                    )

            # Identifier les creux locaux (supports potentiels)
            for i in range(2, len(lows) - 2):
                if (
                    lows[i] < lows[i - 1]
                    and lows[i] < lows[i - 2]
                    and lows[i] < lows[i + 1]
                    and lows[i] < lows[i + 2]
                ):
                    levels.append({"price": lows[i], "type": "Support", "strength": 1})

            # Regrouper les niveaux proches
            grouped_levels = []
            sorted_levels = sorted(levels, key=lambda x: x["price"])

            if sorted_levels:
                current_group = [sorted_levels[0]]
                current_price = sorted_levels[0]["price"]

                for level in sorted_levels[1:]:
                    # Si le niveau est proche du groupe actuel (0.5% de différence)
                    if abs(level["price"] - current_price) / current_price < 0.005:
                        current_group.append(level)
                    else:
                        # Calculer le niveau moyen du groupe
                        avg_price = sum(l["price"] for l in current_group) / len(
                            current_group
                        )
                        avg_strength = sum(l["strength"] for l in current_group)
                        type_counts = {"Support": 0, "Résistance": 0}
                        for l in current_group:
                            type_counts[l["type"]] += 1

                        # Déterminer le type dominant
                        level_type = (
                            "Support"
                            if type_counts["Support"] > type_counts["Résistance"]
                            else "Résistance"
                        )

                        grouped_levels.append(
                            {
                                "price": avg_price,
                                "type": level_type,
                                "strength": avg_strength,
                            }
                        )

                        # Commencer un nouveau groupe
                        current_group = [level]
                        current_price = level["price"]

                # Ajouter le dernier groupe
                if current_group:
                    avg_price = sum(l["price"] for l in current_group) / len(
                        current_group
                    )
                    avg_strength = sum(l["strength"] for l in current_group)
                    type_counts = {"Support": 0, "Résistance": 0}
                    for l in current_group:
                        type_counts[l["type"]] += 1

                    level_type = (
                        "Support"
                        if type_counts["Support"] > type_counts["Résistance"]
                        else "Résistance"
                    )

                    grouped_levels.append(
                        {
                            "price": avg_price,
                            "type": level_type,
                            "strength": avg_strength,
                        }
                    )

            # Trier par force décroissante
            return sorted(grouped_levels, key=lambda x: x["strength"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error identifying key levels: {e}")
            return []

    def initialize_shared_data(self):
        """
        Initialise le fichier partagé en CONSERVANT l'historique
        """
        # Charge l'existant si présent
        if os.path.exists(self.data_file):
            try:
                data = safe_load_shared_data(self.data_file)
                deep_cast_floats(data)
            except Exception as e:
                print(f"Erreur lecture shared_data: {e}")
                data = {}
        else:
            data = {}

        # Préserve les données importantes
        preserved_fields = [
            "trade_history",
            "closed_positions",
            "sentiment",
            "equity_history",
            "news_data",
            "pending_sales",
            "active_pauses",
            "positions_binance",
            "market_data",
        ]
        preserved_data = {
            field: data.get(field, {}) for field in preserved_fields if field in data
        }

        # Réinitialise uniquement les champs de statut
        data.update(
            {
                "timestamp": get_current_time_tahiti(),
                "user": CURRENT_USER,
                "bot_status": {
                    "regime": self.regime,
                    "cycle": self.current_cycle,
                    "last_update": get_current_time(),
                    "performance": self.get_performance_metrics(),
                },
            }
        )

        # Restaure les données préservées
        data.update(preserved_data)

        # PATCH: Utilise safe_update_shared_data pour ne pas écraser le fichier
        self.safe_update_shared_data(data, self.data_file)

    def save_shared_data(self):
        try:
            if os.path.exists(self.data_file):
                data = safe_load_shared_data(self.data_file)
                deep_cast_floats(data)
            else:
                data = {}

            # MAJ des sections
            data.update(
                {
                    "timestamp": get_current_time_tahiti(),
                    "user": CURRENT_USER,
                    "bot_status": {
                        "regime": self.regime,
                        "cycle": self.current_cycle,
                        "last_update": get_current_time(),
                        "performance": self.get_performance_metrics(),
                    },
                    "market_data": self.market_data,
                    "indicators": self.indicators,
                    "positions_binance": getattr(self, "positions_binance", {}),
                    "pending_sales": data.get("pending_sales", []),
                    "active_pauses": data.get("active_pauses", []),
                }
            )

            # Ajoute les métriques avancées pour dashboard
            perf = data["bot_status"]["performance"]
            equity_history = data.get("equity_history", [])
            if equity_history and len(equity_history) > 10:
                import numpy as np

                balances = [pt["balance"] for pt in equity_history if "balance" in pt]
                perf["max_drawdown"] = float(
                    np.min(
                        [0]
                        + [
                            (min(balances[i:], default=0) - b) / b
                            for i, b in enumerate(balances)
                            if b > 0
                        ]
                    )
                )
                returns = np.diff(np.array(balances)) / np.array(balances)[:-1]
                perf["sharpe_ratio"] = (
                    float(np.mean(returns) / np.std(returns))
                    if np.std(returns) > 0
                    else 0
                )
            data["bot_status"]["performance"] = perf

            # PATCH: Utilise safe_update_shared_data pour préserver tout le JSON
            self.safe_update_shared_data(data, self.data_file)
        except Exception as e:
            self.logger.error(f"Error saving shared data: {e}")

    def get_performance_metrics(self):
        try:
            # Première priorité : solde réel Binance
            if getattr(self, "is_live_trading", False) and hasattr(
                self, "binance_client"
            ):
                try:
                    # USDC est la devise principale dans ton dashboard
                    balance_info = self.binance_client.get_asset_balance(asset="USDC")
                    if balance_info:
                        real_balance = float(balance_info.get("free", 0))
                    else:
                        real_balance = 0.0
                except Exception as e:
                    print(f"[BALANCE] Erreur récupération solde Binance : {e}")
                    real_balance = 0.0
            else:
                real_balance = 0.0

            # Récupère les autres métriques (trade_history, etc.)
            try:
                data = safe_load_shared_data(self.data_file)
                saved_perf = data.get("bot_status", {}).get("performance", {})
            except Exception:
                saved_perf = {}

            # Mets à jour la balance avec la valeur réelle
            saved_perf["balance"] = real_balance

            # Cast des autres champs pour éviter les erreurs
            for k in [
                "total_trades",
                "wins",
                "losses",
                "total_profit",
                "total_loss",
                "win_rate",
                "profit_factor",
            ]:
                if k in saved_perf:
                    saved_perf[k] = safe_float(saved_perf[k], 0)

            return saved_perf

        except Exception as e:
            print(f"[BALANCE] Erreur get_performance_metrics: {e}")
            return {"balance": 0.0}

    async def _setup_components(self):
        try:
            # >>>> DEMARRAGE WS <<<<
            await self.ws_collector.start()
            # >>>> FIN AJOUT <<<<

            # Lancement du processus d'analyse des news
            if self.news_enabled and self.news_analyzer:
                asyncio.create_task(self._news_analysis_loop())
                self.logger.info("News analysis loop started")

                # Initialisation des connexions WebSocket Binance si en mode trading réel
                if self.is_live_trading:
                    # Ici vous pouvez initialiser les connexions WebSocket
                    self.logger.info("Binance WebSocket connections initialized")

                await asyncio.sleep(0.5)  # Simule le temps de configuration
                return True

        except Exception as e:
            self.logger.error(f"Error setting up components: {e}")
            return False

    def choose_strategy(self, regime, indicators):
        """Choisit la stratégie"""
        return f"{regime}"

    async def get_latest_data(self):
        """Récupère les dernières données simulées"""
        await asyncio.sleep(0.3)  # Simule le temps de récupération

        # Données simulées pour toutes les paires configurées
        data = {}
        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            data[pair_key] = {}

            # Génération de données OHLCV pour différents timeframes
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                base_price = 100 if "BTC" in pair else 1.5
                volatility = (
                    0.01 if tf in ["1m", "5m"] else 0.02 if tf == "15m" else 0.05
                )

                # Génération de données avec une petite tendance aléatoire
                n_points = 100
                trend = np.random.choice([-0.0001, 0.0001]) * np.arange(n_points)
                noise = np.random.normal(0, volatility, n_points)
                price_movement = trend + noise

                # Création des séries de prix
                closes = base_price * (1 + np.cumsum(price_movement))
                opens = closes * (1 + np.random.normal(0, 0.001, n_points))
                highs = np.maximum(opens, closes) * (
                    1 + np.abs(np.random.normal(0, 0.003, n_points))
                )
                lows = np.minimum(opens, closes) * (
                    1 - np.abs(np.random.normal(0, 0.003, n_points))
                )
                volumes = np.random.normal(1000, 200, n_points)

                data[pair_key][tf] = {
                    "open": opens.tolist(),
                    "high": highs.tolist(),
                    "low": lows.tolist(),
                    "close": closes.tolist(),
                    "volume": volumes.tolist(),
                    "timestamp": [
                        int(datetime.now().timestamp()) - i * 60
                        for i in range(n_points)
                    ],
                }

                # Ajout des signaux simulés
                if "signals" not in data[pair_key]:
                    data[pair_key]["signals"] = {
                        "trend": np.random.uniform(-0.5, 0.5),
                        "momentum": np.random.uniform(-0.5, 0.5),
                        "volatility": np.random.uniform(0, 1),
                    }

        return data

    def train_cnn_lstm_on_live(self, pair="BTCUSDT", tf="1h"):
        """
        Entraîne le modèle CNN-LSTM sur les données live de ws_collector pour la paire/timeframe donnée,
        et sauvegarde les poids dans src/models/cnn_lstm_model.pth
        (NE RESET PLUS à cause de NaN/inf)
        """
        try:
            from src.ai.train_cnn_lstm import train_with_live_data
        except ImportError:
            print("Impossible d'importer train_with_live_data")
            return
        pair_key = pair.replace("/", "").upper()
        print(f"Chargement du DataFrame live pour {pair_key} / {tf}")
        print(
            f"[DEBUG] ws_collector.get_dataframe({pair_key}, {tf}) keys: {list(self.ws_collector.data.keys()) if hasattr(self.ws_collector, 'data') else 'no data attr'}"
        )
        df_live = self.ws_collector.get_dataframe(pair_key, tf)
        if df_live is not None and not df_live.empty:
            df_live = add_dl_features(df_live)
            # Ici : plus jamais de reset si NaN/inf, on log juste le nombre de NaN restant
            for col in ["rsi", "macd", "volatility"]:
                n_nan = df_live[col].isna().sum() if col in df_live.columns else 0
                if n_nan > 0:
                    print(f"⚠️ Attention : {n_nan} NaN dans {col} même après correction")
            print(f"Entraînement du modèle IA sur {len(df_live)} lignes live…")
            train_with_live_data(df_live)
        else:
            print("Aucune donnée live disponible pour entraîner le modèle.")

    def train_cnn_lstm_on_all_live(self):
        """
        Entraîne le modèle CNN-LSTM sur toutes les paires et timeframes de la config,
        en utilisant les données live du ws_collector.
        (NE RESET PLUS à cause de NaN/inf)
        """
        try:
            from src.ai.train_cnn_lstm import train_with_live_data
        except ImportError:
            print("Impossible d'importer train_with_live_data")
            return

        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            for tf in self.config["TRADING"]["timeframes"]:
                print(f"→ Entraînement IA sur {pair_key} / {tf}")
                print(
                    f"[DEBUG] ws_collector.get_dataframe({pair_key}, {tf}) keys: {list(self.ws_collector.data.keys()) if hasattr(self.ws_collector, 'data') else 'no data attr'}"
                )
                df_live = self.ws_collector.get_dataframe(pair_key, tf)
                if df_live is not None and not df_live.empty:
                    df_live = add_dl_features(df_live)
                    for col in ["rsi", "macd", "volatility"]:
                        n_nan = (
                            df_live[col].isna().sum() if col in df_live.columns else 0
                        )
                        if n_nan > 0:
                            print(
                                f"⚠️ Attention : {n_nan} NaN dans {col} même après correction"
                            )
                    print(
                        f"  {len(df_live)} lignes live trouvées, entraînement en cours…"
                    )
                    train_with_live_data(df_live)
                else:
                    print(f"  Pas de données live pour {pair_key} / {tf}, skip.")


def filter_pairs(
    bot,
    min_volatility=0.01,
    min_signal=0.3,
    top_n=5,
    vol_anomaly_filter=True,
    vol_threshold=0.12,
    anomaly_threshold=4.0,
):
    """
    Filtre dynamiquement les paires selon :
    - Volatilité
    - Score du signal
    - Propreté du marché
    - NOUVEAU: Corrélations entre paires
    """
    from src.analysis.filters.volatility_anomaly_filter import filter_market
    from src.analysis.filters.correlation_filter import filter_uncorrelated_pairs
    import numpy as np
    import pandas as pd

    # NOUVEAU: Calcul des corrélations
    correlations = bot.calculate_correlation_matrix()

    candidates = []
    for pair in bot.pairs_valid:
        pair_key = pair.replace("/", "").upper()

        # 1. Analyse volatilité
        vol = 0
        if (
            pair_key in bot.market_data
            and "1h" in bot.market_data[pair_key]
            and "close" in bot.market_data[pair_key]["1h"]
        ):

            closes = bot.market_data[pair_key]["1h"]["close"]
            if len(closes) >= 20:
                returns = np.diff(np.log(closes[-20:]))
                vol = float(np.std(returns))

        # 2. Récupération signal
        signal = 0
        if pair_key in bot.market_data and "ai_prediction" in bot.market_data[pair_key]:
            signal = bot.market_data[pair_key]["ai_prediction"]

        # 3. Analyse anomalies
        df_ohlcv = None
        if (
            pair_key in bot.market_data
            and "1h" in bot.market_data[pair_key]
            and all(
                k in bot.market_data[pair_key]["1h"]
                for k in ["close", "high", "low", "volume"]
            )
        ):

            df_ohlcv = pd.DataFrame(
                {
                    "close": bot.market_data[pair_key]["1h"]["close"],
                    "high": bot.market_data[pair_key]["1h"]["high"],
                    "low": bot.market_data[pair_key]["1h"]["low"],
                    "volume": bot.market_data[pair_key]["1h"]["volume"],
                }
            )

        # 4. Vérification propreté marché
        is_clean = True
        if vol_anomaly_filter and df_ohlcv is not None and len(df_ohlcv) >= 50:
            is_clean = filter_market(
                df_ohlcv,
                vol_threshold=vol_threshold,
                anomaly_threshold=anomaly_threshold,
                price_col="close",
            )

        # 5. NOUVEAU: Score de corrélation
        corr_score = max([v for k, v in correlations.items() if pair in k], default=1.0)

        # 6. Score final combiné
        final_score = (vol * abs(signal)) * (1 - corr_score)

        print(f"[FILTER DEBUG] {pair_key}:")
        print(f"  - Volatilité: {vol:.4f}")
        print(f"  - Signal: {signal:.4f}")
        print(f"  - Corrélation: {corr_score:.4f}")
        print(f"  - Marché propre: {is_clean}")
        print(f"  - Score final: {final_score:.4f}")

        if is_clean and final_score > min_signal:
            candidates.append((pair, final_score))
            print(f"✅ {pair_key} ACCEPTÉ")
        else:
            print(f"❌ {pair_key} REJETÉ")

    # Tri par score final
    candidates.sort(key=lambda x: x[1], reverse=True)
    filtered_candidates = [c[0] for c in candidates]

    # Filtrage corrélation final
    filtered_uncorr = filter_uncorrelated_pairs(
        bot.market_data,
        filtered_candidates,
        timeframe="1h",
        window=50,
        corr_threshold=0.85,
        top_n=top_n,
    )

    return filtered_uncorr


async def log_external_crypto_alert(bot, symbol, news, extra=None):
    """
    Log et notifie une alerte sur une crypto absente de la liste pairs_valid.
    - symbol : code de la crypto (ex : "MATIC")
    - news   : dict de news ou signal
    - extra  : infos additionnelles (score, volume, etc.)
    """
    alert = {
        "timestamp": get_current_time_tahiti(),
        "symbol": symbol,
        "type": "external_alert",
        "title": news.get("title", ""),
        "sentiment": news.get("sentiment", None),
        "details": news,
    }
    if extra:
        alert.update(extra)
    # Ajout au dashboard
    bot.safe_update_shared_data({"external_alerts": [alert]}, bot.data_file)
    # Notification Telegram (si dispo)
    try:
        msg = (
            f"⚠️ Alerte sur {symbol} (non tradée)\n"
            f"Sentiment: {news.get('sentiment')}\n"
            f"Titre: {news.get('title')}\n"
        )
        if extra:
            msg += "\n" + "\n".join(f"{k}: {v}" for k, v in extra.items())
        await bot.telegram.send_message(msg)
    except Exception:
        pass


def load_config():
    """Charge la configuration"""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            return config.get("valid_pairs", ["BTC/USDT", "ETH/USDT"])
    except:
        return ["BTC/USDT", "ETH/USDT"]


async def run_clean_bot():
    """
    Fonction principale du bot de trading
    Gère l'initialisation, l'analyse de marché et l'exécution des stratégies
    """
    print(">>> RUN_CLEAN_BOT DEMARRE <<<")
    orderflow_indicators = AdvancedIndicators()
    logger = logging.getLogger(__name__)

    async def initialize_bot():
        """Initialisation du bot et de ses composants"""
        print(">>> INITIALIZE_BOT <<<")
        bot = None
        try:
            print("\n=== DÉMARRAGE DU BOT ===")
            print("🚀 Trading Bot Ultimate v4 - Version Ultra-Propre")

            # 1. Configuration initiale
            valid_pairs = load_config()

            # 2. Création et configuration du bot
            bot = TradingBotM4()
            bot.pairs_valid = valid_pairs
            bot.auto_update_pairs_from_binance()

            # 3. Préchargement historique (optionnel, sécurisé)
            if hasattr(bot, "ws_collector") and hasattr(bot, "binance_client"):
                for symbol in bot.pairs_valid:  # PAS bot.config["TRADING"]["pairs"]
                    symbol_binance = symbol.replace("/", "").upper()
                    for tf in bot.config["TRADING"]["timeframes"]:
                        try:
                            bot.ws_collector.preload_historical(
                                bot.binance_client, symbol_binance, tf, limit=2000
                            )
                            print(f"Préchargement {symbol_binance} {tf} OK")
                        except Exception as e:
                            print(f"Erreur préchargement {symbol_binance} {tf} : {e}")

            # === AJOUT DIAGNOSTIC DATAFRAME ===
            if hasattr(bot, "ws_collector"):
                print("\n=== DIAGNOSTIC : Contenu du ws_collector ===")
                for pair in bot.pairs_valid:
                    pair_key = pair.replace("/", "").upper()
                    for tf in bot.config["TRADING"]["timeframes"]:
                        df = bot.ws_collector.get_dataframe(pair_key, tf)
                        print(
                            f"{pair_key}-{tf}: {len(df) if df is not None else 0} lignes"
                        )
                print("=== FIN DIAGNOSTIC ===\n")
            # === FIN AJOUT ===

            # 4. Setup des composants internes (websockets, news, etc)
            ok = await bot._setup_components()
            if not ok:
                print("❌ Echec de l'initialisation des composants.")
                return None, None

            # 5. Chargement des données de marché réelles si trading live
            if getattr(bot, "is_live_trading", False):
                await bot._fetch_real_market_data()
                for sym in bot.market_data:
                    print(f"{sym}: {list(bot.market_data[sym].keys())}")

            # 6. Premier rapport d'analyse
            try:
                initial_report = await bot.generate_market_analysis_report(cycle=0)
            except Exception as e:
                initial_report = (
                    f"[ERREUR] Impossible de générer le rapport initial: {e}"
                )

            # 7. Envoi du message Telegram d'initialisation
            try:
                await bot.telegram.send_message(
                    "🚀 <b>Bot Trading démarré</b>\n"
                    "✅ Initialisation réussie\n"
                    f"📊 Paires configurées: {', '.join(valid_pairs)}\n\n"
                    f"{initial_report}"
                )
            except Exception as e:
                print(f"Erreur lors de l'envoi Telegram : {e}")

            print("✅ Bot initialized successfully")
            return bot, valid_pairs

        except Exception as e:
            logger.error(f"Erreur d'initialisation: {e}", exc_info=True)
            print(f"❌ ERREUR FATALE lors de l'initialisation: {e}")
            return None, None

    async def market_analysis_cycle(bot, pair, market_data, tf="1h"):
        """Analyse le marché pour une paire et un timeframe donnés"""
        try:
            pair_key = pair.replace("/", "").upper()
            print(f"\n[DEBUG] Analyse de {pair_key} sur {tf}")

            # 1. Vérification données marché
            if not market_data or pair_key not in market_data:
                print(f"❌ Pas de données market_data pour {pair_key}")
                return None

            # 2. Récupération OHLCV
            ohlcv_df = bot.ws_collector.get_dataframe(pair_key, tf)
            if ohlcv_df is None:
                print(f"❌ Pas de données OHLCV pour {pair_key}-{tf}")
                return None

            if len(ohlcv_df) < 20:
                print(
                    f"❌ Données insuffisantes pour {pair_key}-{tf}: {len(ohlcv_df)} points"
                )
                return None

            # 3. Calcul des indicateurs techniques
            print(f"[DEBUG] Calcul indicateurs pour {pair_key}-{tf}")
            indicators_data = bot.add_indicators(ohlcv_df)

            if indicators_data is None:
                print(f"❌ Échec calcul indicateurs pour {pair_key}-{tf}")
                return None

            print(
                f"✅ Indicateurs calculés pour {pair_key}-{tf}: {list(indicators_data.keys())}"
            )

            # 4. Analyse des signaux avec les indicateurs
            signal = await bot.analyze_signals(
                pair_key, ohlcv_df, indicators_data, tf=tf
            )

            if signal and isinstance(signal.get("signals"), dict):
                signal["pair"] = pair
                signal["tf"] = tf
                print(f"✅ Signaux analysés pour {pair_key}-{tf}")
                return signal
            else:
                print(f"❌ Analyse des signaux échouée pour {pair_key}-{tf}")
                return None

        except Exception as e:
            print(f"❌ Erreur analyse {pair}-{tf}: {str(e)}")
            return None

    async def execute_trading_cycle(bot, valid_pairs):
        """
        Cycle complet de trading avec corrections des problèmes de dashboard et signaux
        Version optimisée (I/O unique, recalcul minimal, parallélisation), sans suppression de fonctionnalités.
        """
        try:
            debug = getattr(bot, "debug", True)

            if debug:
                print("\n=== DÉBUT CYCLE TRADING ===")
            log_dashboard("Démarrage cycle trading")

            # ---------------------------------------------------------------------
            # 1) LECTURE UNIQUE DES DONNÉES PARTAGÉES (réduire I/O)
            # ---------------------------------------------------------------------
            try:
                with open(bot.data_file, "r") as f:
                    shared_data = json.load(f)
            except Exception as e:
                if debug:
                    print(f"[WARNING] Lecture data_file échouée: {e}")
                shared_data = {}

            # 1.a) Initialisation structures bot.market_data si manquantes
            if not hasattr(bot, "market_data"):
                bot.market_data = {}

            if debug:
                print("[DEBUG] Initialisation des structures de base...")
            for pair in bot.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if pair_key not in bot.market_data:
                    bot.market_data[pair_key] = {
                        "sentiment": 0.5,
                        "sentiment_timestamp": time.time(),
                        "ai_prediction": 0.5,
                    }
                    if debug:
                        print(
                            f"[DEBUG] Initialisation {pair_key} avec valeurs par défaut"
                        )

            # ---------------------------------------------------------------------
            # 2) VÉRIFICATION DES PAUSES NEWS (identique, mais I/O limitées)
            # ---------------------------------------------------------------------
            if (
                bot.news_pause_manager
                and bot.news_pause_manager.global_cycles_remaining > 0
            ):
                bot.safe_update_shared_data(
                    {
                        "active_pauses": bot.news_pause_manager.get_active_pauses(),
                        "market_data": bot.market_data,
                    },
                    bot.data_file,
                )
                log_dashboard("🚫 Cycle bloqué - Pause news active")
                return [], bot.regime

            # ---------------------------------------------------------------------
            # 3) ANALYSE DES NEWS (réutilise shared_data chargé au début)
            # ---------------------------------------------------------------------
            if debug:
                print("\n[DEBUG] Analyse des news...")
            try:
                news_sentiment = (
                    shared_data.get("sentiment", {})
                    if isinstance(shared_data, dict)
                    else {}
                )
                news_list = (
                    news_sentiment.get("scores", [])
                    if isinstance(news_sentiment, dict)
                    else []
                )
                unprocessed_news = [n for n in news_list if not n.get("processed")]

                if unprocessed_news:
                    if bot.news_pause_manager.scan_news(unprocessed_news):
                        (
                            print("🚨 Pause trading activée suite à news critique")
                            if debug
                            else None
                        )
                        for n in unprocessed_news:
                            n["processed"] = True

                        bot.safe_update_shared_data(
                            {
                                "sentiment": {**news_sentiment, "scores": news_list},
                                "active_pauses": bot.news_pause_manager.get_active_pauses(),
                            },
                            bot.data_file,
                        )

                        if bot.news_pause_manager.global_cycles_remaining > 0:
                            return [], bot.regime
            except Exception as e:
                if debug:
                    print(f"[WARNING] Erreur analyse news: {e}")

            # ---------------------------------------------------------------------
            # 4) MISE À JOUR DES DONNÉES MARCHÉ (parallélisée & recalcul minimal)
            # ---------------------------------------------------------------------
            if debug:
                print("\n[DEBUG] Mise à jour des données marché...")

            try:
                orderflow_indicators = AdvancedIndicators()
            except Exception as e:
                if debug:
                    print(f"[WARNING] Erreur init indicators: {e}")
                orderflow_indicators = None  # conservé même si pas utilisé ici

            async def process_pair_tf(pair, tf):
                """
                Traite une paire/timeframe :
                - Récupère df
                - Corrige timestamp
                - Accumule sans doublons
                - Recalcule indicateurs SEULEMENT si nouvelles bougies
                """
                pair_key = pair.replace("/", "").upper()

                # Récupération dataframe en thread pour ne pas bloquer l'event loop
                df = await asyncio.to_thread(
                    bot.ws_collector.get_dataframe, pair_key, tf
                )
                if df is None or df.empty:
                    return

                required_cols = ["open", "high", "low", "close", "volume", "timestamp"]
                if not all(col in df.columns for col in required_cols):
                    return

                # Correction timestamp si nécessaire
                if len(df["timestamp"]) != len(df["close"]):
                    if (
                        hasattr(df.index, "dtype")
                        and np.issubdtype(df.index.dtype, np.datetime64)
                        and len(df.index) == len(df["close"])
                    ):
                        df["timestamp"] = df.index
                    else:
                        df["timestamp"] = pd.date_range(
                            end=pd.Timestamp.utcnow(),
                            periods=len(df["close"]),
                            freq="T",
                        )

                # Assure structure tf
                if tf not in bot.market_data.get(pair_key, {}):
                    bot.market_data[pair_key].setdefault(
                        tf,
                        {
                            "open": [],
                            "high": [],
                            "low": [],
                            "close": [],
                            "volume": [],
                            "timestamp": [],
                        },
                    )

                # Construction OHLCV dict (listes)
                ohlcv_dict = {
                    "open": df["open"].tolist(),
                    "high": df["high"].tolist(),
                    "low": df["low"].tolist(),
                    "close": df["close"].tolist(),
                    "volume": df["volume"].tolist(),
                    "timestamp": [
                        int(pd.Timestamp(t).timestamp()) for t in df["timestamp"]
                    ],
                }

                # Accumulation SANS DOUBLONS
                last_ts = (
                    bot.market_data[pair_key][tf]["timestamp"][-1]
                    if bot.market_data[pair_key][tf]["timestamp"]
                    else None
                )
                if last_ts is not None and isinstance(last_ts, str):
                    try:
                        last_ts = int(pd.Timestamp(last_ts).timestamp())
                    except Exception:
                        last_ts = None

                new_indices = []
                for i, ts in enumerate(ohlcv_dict["timestamp"]):
                    ts_int = ts
                    if isinstance(ts, str):
                        try:
                            ts_int = int(pd.Timestamp(ts).timestamp())
                        except Exception:
                            continue
                    if last_ts is None or ts_int > last_ts:
                        new_indices.append(i)

                if not new_indices:
                    # Rien de nouveau => ne recalcule pas les indicateurs
                    if debug:
                        print(
                            f"[DEBUG] {pair_key}-{tf} pas de nouvelles bougies, skip calcul indicateurs."
                        )
                    return

                for k in ohlcv_dict:
                    md_list = bot.market_data[pair_key][tf][k]
                    md_list.extend([ohlcv_dict[k][i] for i in new_indices])

                # DEBUG tailles
                if debug:
                    print(
                        f"[DEBUG] {pair_key}-{tf} nb bougies : {len(bot.market_data[pair_key][tf]['close'])}"
                    )
                    print(
                        f"[DEBUG] {pair_key}-{tf} timestamp (type): {type(bot.market_data[pair_key][tf]['timestamp'])}"
                    )

                # Calcul des indicateurs (inchangé, mais seulement si nouvelles bougies)
                indicators_data = await asyncio.to_thread(bot.add_indicators, df)
                bot.market_data[pair_key][tf]["signals"] = {
                    "technical": {
                        "score": float(indicators_data.get("technical_score", 0.5)),
                        "details": indicators_data,
                        "factors": len(indicators_data),
                    },
                    "momentum": {"score": 0.5, "details": {}, "factors": 0},
                    "orderflow": {
                        "score": 0.5,
                        "details": {},
                        "factors": 0,
                        "liquidity": 0.5,
                        "market_pressure": 0.5,
                    },
                    "ai": float(bot.market_data[pair_key].get("ai_prediction", 0.5)),
                    "sentiment": float(bot.market_data[pair_key].get("sentiment", 0.5)),
                }

                if debug:
                    print(f"[DEBUG] {pair_key}-{tf} Indicateurs calculés:")
                    print(
                        f"- Technical score: {indicators_data.get('technical_score', 0.5)}"
                    )

            # Lancer TOUTES les MAJ OHLCV/indicateurs en parallèle
            tasks = []
            for pair in bot.pairs_valid:
                for tf in bot.config["TRADING"]["timeframes"]:
                    tasks.append(process_pair_tf(pair, tf))
            if tasks:
                await asyncio.gather(*tasks)

            # ---------------------------------------------------------------------
            # 6) INITIALISATION DES DÉCISIONS (identique, I/O évitées)
            # ---------------------------------------------------------------------
            trade_decisions = []
            decisions_for_dashboard = {}

            if debug:
                print("\n[DEBUG] Initialisation des décisions...")

            current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            for pair in bot.pairs_valid:
                try:
                    pair_key = pair.replace("/", "").upper()

                    if pair_key not in bot.market_data:
                        bot.market_data[pair_key] = {}

                    market_signals = bot.market_data[pair_key]
                    confidence = 0.5  # valeur par défaut

                    # Signaux techniques depuis 1h si dispo
                    tf_data = market_signals.get("1h", {}).get("signals", {})
                    tech_data = tf_data.get("technical", {})

                    # 1. Technical Score
                    try:
                        tech_score = safe_float(tech_data.get("score", 0.5))
                        tech_score = max(0.0, min(1.0, tech_score))
                    except (TypeError, ValueError):
                        if debug:
                            print(f"[WARNING] Invalid tech score for {pair_key}")
                        tech_score = 0.5

                    # 2. AI Score
                    try:
                        ai_score = market_signals.get("ai_prediction")
                        if ai_score is not None:
                            ai_score = safe_float(ai_score)
                            ai_score = max(0.0, min(1.0, ai_score))
                        else:
                            if debug:
                                print(f"[WARNING] No AI prediction for {pair_key}")
                            ai_score = 0.5
                    except (TypeError, ValueError):
                        if debug:
                            print(f"[WARNING] Invalid AI score for {pair_key}")
                        ai_score = 0.5

                    # 3. Sentiment Score
                    try:
                        sentiment_score = market_signals.get("sentiment")
                        if sentiment_score is not None:
                            sentiment_score = safe_float(sentiment_score)
                            sentiment_score = max(-1.0, min(1.0, sentiment_score))
                        else:
                            # Récupération depuis shared_data déjà chargé
                            news_sentiment = shared_data.get("sentiment", {})
                            sentiment_score = safe_float(
                                news_sentiment.get("overall_sentiment", 0.0)
                            )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        sentiment_score = 0.0

                    # Pondération des signaux pour la confiance
                    if all(
                        x is not None for x in [tech_score, ai_score, sentiment_score]
                    ):
                        tech_weight = 0.5
                        ai_weight = 0.3
                        sentiment_weight = 0.2
                        norm_sentiment = (sentiment_score + 1) / 2
                        confidence = (
                            tech_score * tech_weight
                            + ai_score * ai_weight
                            + norm_sentiment * sentiment_weight
                        )
                        confidence = max(0.5, min(1.0, confidence))

                    # Décision initiale
                    decision = {
                        "pair": pair,
                        "action": "neutral",
                        "confidence": round(confidence, 4),
                        "tech": round(tech_score, 4),
                        "ai": round(ai_score, 4),
                        "sentiment": round(sentiment_score, 4),
                        "timestamp": current_time,
                    }

                    weighted_signal = (
                        tech_score * 0.5
                        + ai_score * 0.3
                        + ((sentiment_score + 1) / 2) * 0.2
                    )
                    if weighted_signal > 0.7:
                        decision["action"] = "buy"
                    elif weighted_signal < 0.3:
                        decision["action"] = "sell"

                    if debug:
                        print(f"\n[DEBUG] Décision pour {pair}:")
                        print(f"- Tech score: {tech_score:.4f}")
                        print(f"- AI score: {ai_score:.4f}")
                        print(f"- Sentiment: {sentiment_score:.4f}")
                        print(f"- Confiance: {confidence:.4f}")
                        print(f"- Action: {decision['action']}")

                    decisions_for_dashboard[pair] = decision

                    # Si signaux forts → ajout aux décisions
                    if decision["action"] != "neutral" and decision["confidence"] > 0.7:
                        trade_decisions.append(decision)

                except Exception as e:
                    print(f"[ERROR] Failed to process {pair}: {str(e)}")
                    continue

            # Mise à jour market_data avec décisions
            for pair, decision in decisions_for_dashboard.items():
                pair_key = pair.replace("/", "").upper()
                if pair_key not in bot.market_data:
                    bot.market_data[pair_key] = {}
                bot.market_data[pair_key]["last_decision"] = decision
                bot.market_data[pair_key]["last_update"] = current_time

            # Debug final (limité)
            if debug:
                print("\n[DEBUG] Résumé des décisions:")
                for pair, decision in decisions_for_dashboard.items():
                    print(f"\n{pair}:")
                    try:
                        print(json.dumps(decision, indent=2))
                    except Exception:
                        print(str(decision))

            # ---------------------------------------------------------------------
            # 7) ANALYSE DES SIGNAUX (identique, mais on évite recalculs inutiles)
            # ---------------------------------------------------------------------
            signals_ok = bot.verify_signals_completeness()
            if not signals_ok:
                log_dashboard("⚠️ Signaux incomplets")

            if debug:
                print("\n[DEBUG] Vérification finale des décisions:")
                for pair, decision in decisions_for_dashboard.items():
                    print(f"\n{pair}:")
                    for key, value in decision.items():
                        if isinstance(value, (int, float)):
                            print(f"- {key}: {value:.4f}")
                        else:
                            print(f"- {key}: {value}")

            for pair, decision in decisions_for_dashboard.items():
                pair_key = pair.replace("/", "").upper()
                if pair_key not in bot.market_data:
                    bot.market_data[pair_key] = {}
                bot.market_data[pair_key].update(
                    {"last_decision": decision, "last_update": current_time}
                )

            # ---------------------------------------------------------------------
            # 8) GÉNÉRATION DES DÉCISIONS DÉTAILLÉES (inchangée dans la logique)
            # ---------------------------------------------------------------------
            if debug:
                print("\n[DEBUG] Génération des décisions...")
            for pair in bot.pairs_valid:
                pair_signals = {}
                pair_key = pair.replace("/", "").upper()
                for tf in bot.config["TRADING"]["timeframes"]:
                    # Récupération df à la volée (si besoin) ; pourrait se réutiliser
                    df = bot.ws_collector.get_dataframe(pair_key, tf)
                    if df is not None and len(df) >= 20:
                        decision = await bot.analyze_signals(
                            pair_key, df, bot.add_indicators(df), tf
                        )
                        if debug:
                            print(f"[DEBUG] {pair} - decision:", decision)
                        if decision and isinstance(decision.get("signals"), dict):
                            decision["tf"] = tf
                            pair_signals[tf] = decision

                if pair_signals:
                    dominant_tf = "1h"
                    dominant_signals = pair_signals.get(dominant_tf, {}).get(
                        "signals", {}
                    )

                    if bot.risk_manager.validate_trade(dominant_signals):
                        action, base_confidence = bot.aggregate_timeframe_signals(
                            pair, pair_signals
                        )

                        # Calcul confiance (utilise les RÉELS scores)
                        tech_score = safe_float(
                            dominant_signals.get("technical", {}).get("score", 0.5)
                        )
                        momentum_score = float(
                            dominant_signals.get("momentum", {}).get("score", 0.5)
                        )
                        orderflow_score = float(
                            dominant_signals.get("orderflow", {}).get("score", 0.5)
                        )
                        ai_score = safe_float(
                            bot.market_data[pair_key].get("ai_prediction", 0.5)
                        )
                        sentiment_score = safe_float(
                            bot.market_data[pair_key].get("sentiment", 0.5)
                        )

                        if debug:
                            print(f"\n[DEBUG] Scores pour {pair}:")
                            print(f"- Technical: {tech_score:.3f}")
                            print(f"- Momentum: {momentum_score:.3f}")
                            print(f"- Orderflow: {orderflow_score:.3f}")
                            print(f"- AI: {ai_score:.3f}")
                            print(f"- Sentiment: {sentiment_score:.3f}")

                        confidence = (
                            tech_score * 0.35
                            + momentum_score * 0.25
                            + orderflow_score * 0.2
                            + ai_score * 0.15
                            + sentiment_score * 0.05
                        )
                        confidence = max(0.5, min(confidence, 1.0))

                        if debug:
                            print(f"=> Confidence calculée: {confidence:.3f}")

                        sizing_multiplier = 1.0
                        # NOTE: logique d’origine conservée (l’elif >0.8 ne s’exécute pas si >0.7)
                        if confidence > 0.7:
                            sizing_multiplier = 1.5
                        elif confidence > 0.8:
                            sizing_multiplier = 2.0

                        final_decision = {
                            "pair": pair,
                            "action": action,
                            "confidence": float(confidence),
                            "sizing_multiplier": sizing_multiplier,
                            "signals": {
                                "technical": tech_score,
                                "momentum": momentum_score,
                                "orderflow": orderflow_score,
                                "ai": ai_score,
                                "sentiment": sentiment_score,
                            },
                        }

                        # Mise à jour dashboard avec les vraies valeurs
                        dashboard_update = {
                            "action": str(action),
                            "confidence": float(confidence),
                            "tech": float(tech_score),
                            "ai": float(ai_score),
                            "sentiment": float(sentiment_score),
                        }

                        if debug:
                            print(f"\n[DEBUG] Mise à jour dashboard pour {pair}:")
                            try:
                                print(
                                    f"Avant: {json.dumps(decisions_for_dashboard[pair], indent=2)}"
                                )
                            except Exception:
                                print(str(decisions_for_dashboard.get(pair)))

                        decisions_for_dashboard[pair].update(dashboard_update)

                        if debug:
                            try:
                                print(
                                    f"Après: {json.dumps(decisions_for_dashboard[pair], indent=2)}"
                                )
                                print(f"\n[DEBUG] Dashboard mis à jour pour {pair}:")
                                print(
                                    json.dumps(decisions_for_dashboard[pair], indent=2)
                                )
                            except Exception:
                                pass

                        signal_score = (
                            tech_score * 0.3
                            + momentum_score * 0.2
                            + orderflow_score * 0.2
                            + ai_score * 0.2
                            + sentiment_score * 0.1
                        )

                        # Filtre final (inchangé)
                        if (
                            signal_score >= 0.6
                            and confidence >= 0.7
                            and bot.calculate_volatility_advanced(df) <= 0.08
                        ):
                            trade_decisions.append(final_decision)

            # ---------------------------------------------------------------------
            # 9) VALIDATION FINALE DES DONNÉES (identique)
            # ---------------------------------------------------------------------
            if debug:
                print("\n[DEBUG] Validation finale des données...")
            for pair in decisions_for_dashboard:
                decision = decisions_for_dashboard[pair]
                if debug:
                    print(f"\n[DEBUG] Validation {pair} avant correction:")
                    try:
                        print(json.dumps(decision, indent=2))
                    except Exception:
                        print(str(decision))

                for key in ["confidence", "tech", "ai", "sentiment"]:
                    current_val = decision.get(key)
                    if current_val is None:
                        decisions_for_dashboard[pair][key] = 0.5
                        if debug:
                            print(f"[DEBUG] {pair}: {key} était None -> 0.5")
                    else:
                        try:
                            decisions_for_dashboard[pair][key] = float(current_val)
                        except Exception:
                            if debug:
                                print(f"[DEBUG] {pair}: {key} non convertible -> 0.5")
                            decisions_for_dashboard[pair][key] = 0.5

                if debug:
                    print(f"[DEBUG] {pair} après validation:")
                    try:
                        print(json.dumps(decision, indent=2))
                    except Exception:
                        print(str(decision))

            # ---------------------------------------------------------------------
            # 10) SAUVEGARDE DES DONNÉES (UNE SEULE FOIS)
            # ---------------------------------------------------------------------
            try:
                if debug:
                    print("\n[DEBUG] Sauvegarde des données...")
                current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                regime = getattr(bot, "regime", "Indéterminé")

                # On réutilise shared_data (évite re-lire disque)
                existing_data = shared_data if isinstance(shared_data, dict) else {}

                # Fusion decisions
                existing_trade_decisions = existing_data.get("trade_decisions", {})
                merged_trade_decisions = existing_trade_decisions.copy()
                for pair, decision in decisions_for_dashboard.items():
                    if pair in merged_trade_decisions:
                        merged_trade_decisions[pair].update(decision)
                    else:
                        merged_trade_decisions[pair] = decision

                data_to_save = {
                    "trade_decisions": merged_trade_decisions,
                    "market_data": {
                        **existing_data.get("market_data", {}),
                        **bot.market_data,
                    },
                    "cycle_metrics": {
                        "cycle": bot.current_cycle,
                        "regime": regime,
                        "balance": bot.get_performance_metrics().get(
                            "balance", 10000.0
                        ),
                        "timestamp": current_time,
                    },
                    "bot_status": {
                        "cycle": bot.current_cycle,
                        "regime": regime,
                        "performance": bot.get_performance_metrics(),
                        "last_update": current_time,
                    },
                }

                preserved_fields = [
                    "trade_history",
                    "closed_positions",
                    "equity_history",
                    "news_data",
                    "sentiment",
                    "active_pauses",
                    "pending_sales",
                    "positions_binance",
                    "market_data",  # ← déjà fusionné, mais à garder si besoin
                ]
                for field in preserved_fields:
                    if field in existing_data:
                        data_to_save[field] = existing_data[field]

                # Backup + sauvegarde
                if os.path.exists(bot.data_file):
                    shutil.copyfile(bot.data_file, bot.data_file + ".bak")

                bot.safe_update_shared_data(data_to_save, bot.data_file)
                if debug:
                    print("✅ Données sauvegardées avec succès")

            except Exception as e:
                print(f"❌ Erreur sauvegarde: {e}")
                # Sauvegarde minimale en cas d'erreur (identique à l’origine)
                regime = getattr(bot, "regime", "Indéterminé")
                bot.safe_update_shared_data(
                    {
                        "trade_decisions": decisions_for_dashboard,
                        "bot_status": {"cycle": bot.current_cycle, "regime": regime},
                    },
                    bot.data_file,
                )

            # ---------------------------------------------------------------------
            # 11) EXÉCUTION DES TRADES (identique, avec mêmes filtres)
            # ---------------------------------------------------------------------
            if signals_ok and trade_decisions:
                current_exposure = sum(
                    safe_float(pos.get("amount", 0))
                    * safe_float(pos.get("entry_price", 0))
                    for pos in bot.positions.values()
                ) / bot.get_performance_metrics().get("balance", 1)

                if (
                    current_exposure
                    < bot.risk_manager.position_limits["max_total_exposure"]
                ):
                    filtered_decisions = []

                    for decision in trade_decisions:
                        pair_key = decision["pair"].replace("/", "").upper()
                        volatility = bot.calculate_volatility(
                            bot.market_data.get(pair_key, {}).get("1h", {})
                        )

                        # === PAUSE INTELLIGENTE ===
                        action = decision.get("action")
                        pair = decision.get("pair")
                        trading_paused = (
                            bot.news_pause_manager.global_cycles_remaining > 0
                            if bot.news_pause_manager
                            else False
                        )
                        pair_paused = (
                            (
                                pair in bot.news_pause_manager.pair_pauses
                                and bot.news_pause_manager.pair_pauses[pair] > 0
                            )
                            if bot.news_pause_manager
                            else False
                        )
                        buy_paused = (
                            (pair in bot.news_pause_manager.buy_paused_pairs)
                            if bot.news_pause_manager
                            else False
                        )

                        if trading_paused and action == "buy":
                            if debug:
                                print(
                                    f"[SMART PAUSE] Achat {pair} bloqué par pause globale/news."
                                )
                            continue
                        if pair_paused and action == "buy":
                            if debug:
                                print(
                                    f"[SMART PAUSE] Achat {pair} bloqué par pause sur la paire."
                                )
                            continue
                        if buy_paused and action == "buy":
                            if debug:
                                print(
                                    f"[SMART PAUSE] Achat {pair} bloqué par pause BUY (régulation/news)."
                                )
                            continue

                        # Filtre volatilité
                        if volatility > 0.08:
                            if debug:
                                print(
                                    f"[FILTER] Volatilité trop élevée sur {pair}, trade ignoré."
                                )
                            continue

                        base_size = 12
                        confidence = decision.get("confidence", 0.5)
                        sizing_multiplier = decision.get("sizing_multiplier", 1.0)

                        final_size = base_size * sizing_multiplier * (confidence / 0.5)
                        final_size = max(12, min(final_size, 50))

                        decision["amount"] = final_size
                        filtered_decisions.append(decision)

                        if debug:
                            print(
                                f"[SIZING] {decision['pair']} : {final_size:.2f} USDC "
                                f"(conf={confidence:.2f}, mult={sizing_multiplier})"
                            )

                    if filtered_decisions:
                        await execute_trade_decisions(bot, filtered_decisions)
                        log_dashboard(
                            f"✅ {len(filtered_decisions)}/{len(trade_decisions)} trades exécutés"
                        )
                    else:
                        log_dashboard("ℹ️ Toutes décisions rejetées après filtrage")
                else:
                    log_dashboard(f"🚫 Exposition ({current_exposure:.1%}) > limite")

            return trade_decisions, getattr(bot, "regime", "Indéterminé")

        except Exception as e:
            logger.error(f"❌ Erreur cycle trading: {e}")
            raise

    async def main():
        try:
            # Initialisation
            bot, valid_pairs = await initialize_bot()
            if bot is None:
                print("Erreur critique à l'initialisation du bot. Arrêt.")
                return

            await bot.test_news_sentiment()

            # Analyse initiale du marché
            regime, _, _ = await bot.study_market("7d")
            log_dashboard(f"🔈 Régime de marché détecté: {regime}")

            # Boucle principale
            cycle = 0
            while True:
                await bot.handle_auto_sell()
                bot.sync_positions_with_binance()

                for symbol in list(bot.positions_binance.keys()):
                    if safe_float(bot.positions_binance[symbol].get("amount", 0)) <= 0:
                        del bot.positions_binance[symbol]
                print(f"=== NOUVEAU CYCLE {cycle} ===")
                print(f"[DEBUG CYCLE] Positions (avant TP/SL): {bot.positions}")
                print(f"[DEBUG CYCLE] bot.positions: {bot.positions}")
                print(
                    f"[DEBUG CYCLE] Positions binance: {getattr(bot, 'positions_binance', {})}"
                )
                print(f"[DEBUG CYCLE] bot.portfolio: {getattr(bot, 'portfolio', {})}")
                try:
                    cycle += 1
                    start = datetime.utcnow()

                    # Mise à jour des positions en attente
                    bot.get_pending_sales()

                    # Gestion des news et pauses
                    try:
                        with open(bot.data_file, "r") as f:
                            shared_data = json.load(f)
                        news_sentiment = (
                            shared_data.get("sentiment", {})
                            if isinstance(shared_data, dict)
                            else {}
                        )
                        news_list = (
                            news_sentiment.get("scores", [])
                            if isinstance(news_sentiment, dict)
                            else []
                        )
                    except Exception:
                        news_list = []

                    # Traitement des news non traitées
                    unprocessed_news = [n for n in news_list if not n.get("processed")]
                    if unprocessed_news and bot.news_pause_manager.scan_news(
                        unprocessed_news
                    ):
                        print("🚨 Pause trading à cause d'une news critique!")
                        for n in unprocessed_news:
                            n["processed"] = True

                        # Merge processed flags before saving
                        old_scores = shared_data.get("sentiment", {}).get("scores", [])
                        news_list = merge_news_processed(old_scores, news_list)

                        bot.safe_update_shared_data(
                            {
                                "sentiment": {
                                    **shared_data.get("sentiment", {}),
                                    "scores": news_list,
                                }
                            },
                            bot.data_file,
                        )

                    # Décompte des pauses à chaque tick
                    bot.news_pause_manager.on_cycle_end()
                    active_pauses = bot.get_active_pauses()
                    print("[DEBUG PATCH] Pauses RAM après tick:", active_pauses)

                    # Synchronisation systématique avec le fichier partagé
                    bot.safe_update_shared_data(
                        {"active_pauses": active_pauses}, bot.data_file
                    )

                    # Vérification pause globale
                    trading_paused = bot.news_pause_manager.global_cycles_remaining > 0
                    if trading_paused:
                        print(
                            "Trading en pause: calculs et signaux mis à jour, EXÉCUTION DES TRADES BLOQUÉE."
                        )

                    # Hot reload du modèle IA
                    bot.check_reload_dl_model()

                    # Gestion des stop-loss SPOT
                    for symbol, pos in list(
                        getattr(bot, "positions_binance", {}).items()
                    ):
                        if pos.get("side") == "long" and bot.check_stop_loss(symbol):
                            print(
                                f"[STOPLOSS] Déclenchement automatique du stop-loss pour {symbol}"
                            )
                            await bot.execute_trade(
                                symbol, "SELL", safe_float(pos.get("amount"), 0)
                            )
                            print(f"[STOPLOSS] Position fermée pour {symbol}")
                            getattr(bot, "positions_binance", {}).pop(symbol, None)
                            continue  # Passe à la position suivante

                    # Gestion des TP et trailing stop
                    for symbol, pos in list(
                        getattr(bot, "positions_binance", {}).items()
                    ):
                        print(f"\n[DEBUG CYCLE] Analyse {symbol} | pos={pos}")

                        # Filtre seulement les positions "long"
                        if pos.get("side") != "long":
                            print(
                                f"[DEBUG SKIP] {symbol}: side={pos.get('side')} (not long)"
                            )
                            continue

                        # Initialisation des champs si absents
                        if "filled_tp_targets" not in pos:
                            pos["filled_tp_targets"] = [False, False]
                        if "price_history" not in pos:
                            pos["price_history"] = [
                                safe_float(pos.get("entry_price"), 0)
                            ]
                        if "max_price" not in pos:
                            pos["max_price"] = safe_float(pos.get("entry_price"), 0)

                        # Récupération du dernier prix
                        last_price = (
                            bot.ws_collector.get_last_price(symbol)
                            if hasattr(bot, "ws_collector")
                            else None
                        )
                        if (
                            last_price is None
                            and symbol in bot.market_data
                            and "1h" in bot.market_data[symbol]
                        ):
                            closes = bot.market_data[symbol]["1h"].get("close", [])
                            if closes:
                                last_price = closes[-1]
                        if last_price is None:
                            continue

                        last_price = safe_float(last_price, 0)
                        pos["price_history"].append(last_price)

                        # Take Profit partiel
                        entry_price = safe_float(pos.get("entry_price"), 0)
                        amount = safe_float(pos.get("amount"), 0)
                        to_exit, new_filled = bot.exit_manager.check_tp_partial(
                            entry_price, last_price, pos["filled_tp_targets"]
                        )
                        to_exit = safe_float(to_exit, 0)
                        if to_exit > 0 and amount > 0:
                            amount_to_sell = amount * to_exit
                            await bot.execute_trade(symbol, "SELL", amount_to_sell)
                            pos["amount"] = amount - amount_to_sell
                            pos["filled_tp_targets"] = new_filled
                            print(
                                f"[TP PARTIEL] {symbol}: Vente {amount_to_sell} à TP, nouveau amount={pos['amount']} | filled_tp_targets={pos['filled_tp_targets']}"
                            )
                            if safe_float(pos.get("amount"), 0) <= 0:
                                bot.positions.pop(symbol)
                                continue

                        # Trailing stop
                        should_exit, new_max = bot.exit_manager.check_trailing(
                            entry_price,
                            pos["price_history"],
                            safe_float(pos.get("max_price", entry_price), entry_price),
                        )
                        pos["max_price"] = safe_float(new_max, 0)
                        if should_exit and safe_float(pos.get("amount"), 0) > 0:
                            await bot.execute_trade(
                                symbol, "SELL", safe_float(pos.get("amount"), 0)
                            )
                            bot.positions.pop(symbol)

                    # Gestion des shorts BingX
                    for symbol, pos in list(
                        getattr(bot, "positions_binance", {}).items()
                    ):
                        if bot.is_short(symbol):
                            try:
                                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                                ticker = await bot.bingx_client.fetch_ticker(
                                    symbol_bingx
                                )
                                price = safe_float(ticker["last"], 0)

                                if bot.check_short_stop(
                                    symbol, price=price, trailing_pct=0.03
                                ):
                                    print(
                                        f"[SHORT STOP] Fermeture short {symbol} (prix: {price})"
                                    )
                                    await bot.telegram.send_message(
                                        f"🔴 <b>STOP SHORT déclenché</b>\n"
                                        f"Pair: {symbol}\n"
                                        f"Prix actuel: {price}\n"
                                        f"Position couverte automatiquement (stop/trailing stop)"
                                    )
                                    await bot.execute_trade(
                                        symbol, "BUY", safe_float(pos.get("amount"), 0)
                                    )
                            except Exception:
                                continue

                    # Analyse de marché et génération des signaux
                    trade_decisions, regime = await execute_trading_cycle(
                        bot, valid_pairs
                    )
                    # 1. Pump
                    pump_candidates = bot.detect_pump_candidates()
                    for c in pump_candidates:
                        symbol = c["pair"].split("/")[0]
                        if c["pair"] not in bot.pairs_valid:
                            # Crypto non tradée : log & alerte spéciale
                            await log_external_crypto_alert(bot, symbol, c)
                            continue
                        if bot.is_long(c["pair"]):
                            print(
                                f"[SKIP] Achat impulsif ignoré sur {c['pair']} — déjà en portefeuille."
                            )
                            continue
                        await bot.telegram.send_message(
                            f"🚀 Pump détecté sur {c['pair']}: +{c['price_pct']*100:.1f}%, volume x{c['vol_ratio']:.1f}"
                        )
                        base_amount = 15
                        result = await bot.execute_trade(c["pair"], "BUY", base_amount)
                        bot.safe_update_shared_data(
                            {
                                "pump_opportunities": [
                                    {
                                        "timestamp": get_current_time_tahiti(),
                                        "pair": c["pair"],
                                        "type": "pump",
                                        "price_pct": c["price_pct"],
                                        "vol_ratio": c["vol_ratio"],
                                        "result": result,
                                    }
                                ]
                            },
                            bot.data_file,
                        )
                        if result and result.get("status") == "completed":
                            entry_price = safe_float(result.get("avg_price", 0))
                            await bot.plan_auto_sell(
                                c["pair"],
                                entry_price,
                                base_amount,
                                tp_pct=0.03,
                                sl_pct=0.03,
                                max_cycles=2,
                                reason="pump: trailing_stop",
                            )

                    # 2. Breakout
                    breakout_candidates = bot.detect_breakout_candidates()
                    for c in breakout_candidates:
                        symbol = c["pair"].split("/")[0]
                        if c["pair"] not in bot.pairs_valid:
                            await log_external_crypto_alert(bot, symbol, c)
                            continue
                        if bot.is_long(c["pair"]):
                            print(
                                f"[SKIP] Achat impulsif ignoré sur {c['pair']} — déjà en portefeuille."
                            )
                            continue
                        await bot.telegram.send_message(
                            f"💥 Breakout sur {c['pair']}: close={c['close']:.2f} > {c['breakout_level']:.2f}"
                        )
                        base_amount = 15
                        result = await bot.execute_trade(c["pair"], "BUY", base_amount)
                        bot.safe_update_shared_data(
                            {
                                "breakout_opportunities": [
                                    {
                                        "timestamp": get_current_time_tahiti(),
                                        "pair": c["pair"],
                                        "type": "breakout",
                                        "close": c["close"],
                                        "breakout_level": c["breakout_level"],
                                        "result": result,
                                    }
                                ]
                            },
                            bot.data_file,
                        )
                        if result and result.get("status") == "completed":
                            entry_price = safe_float(result.get("avg_price", 0))
                            await bot.plan_auto_sell(
                                c["pair"],
                                entry_price,
                                base_amount,
                                tp_pct=0.03,
                                sl_pct=0.03,
                                max_cycles=2,
                                reason="breakout: trailing_stop",
                            )

                    # Appel impulsif sur news forte, même hors pairs_valid
                    for news in news_list:
                        if float(news.get("sentiment", 0)) > 0.7:
                            await bot.detect_and_buy_news_impulsif(news)

                    # 3. News
                    news_candidates = bot.detect_news_candidates(news_list)
                    for c in news_candidates:
                        symbol = c["pair"].split("/")[0]
                        if c["pair"] not in bot.pairs_valid:
                            await log_external_crypto_alert(bot, symbol, c)
                            continue
                        if bot.is_long(c["pair"]):
                            print(
                                f"[SKIP] Achat impulsif ignoré sur {c['pair']} — déjà en portefeuille."
                            )
                            continue
                        await bot.telegram.send_message(
                            f"📰 News positive sur {c['pair']}: sentiment={c['sentiment']:.2f}\n{c['title']}"
                        )
                        base_amount = 15
                        result = await bot.execute_trade(c["pair"], "BUY", base_amount)
                        bot.safe_update_shared_data(
                            {
                                "news_opportunities": [
                                    {
                                        "timestamp": get_current_time_tahiti(),
                                        "pair": c["pair"],
                                        "type": "news",
                                        "sentiment": c["sentiment"],
                                        "title": c["title"],
                                        "result": result,
                                    }
                                ]
                            },
                            bot.data_file,
                        )
                        if result and result.get("status") == "completed":
                            entry_price = safe_float(result.get("avg_price", 0))
                            await bot.plan_auto_sell(
                                c["pair"],
                                entry_price,
                                base_amount,
                                tp_pct=0.03,
                                sl_pct=0.03,
                                max_cycles=2,
                                reason="news: trailing_stop",
                            )

                    # 4. Arbitrage
                    arbitrage_candidates = await bot.detect_arbitrage_candidates()
                    for c in arbitrage_candidates:
                        symbol = c["pair"].split("/")[0]
                        if c["pair"] not in bot.pairs_valid:
                            await log_external_crypto_alert(bot, symbol, c)
                            continue
                        if bot.is_long(c["pair"]):
                            print(
                                f"[SKIP] Achat impulsif ignoré sur {c['pair']} — déjà en portefeuille."
                            )
                            continue
                        await bot.telegram.send_message(
                            f"💹 Arbitrage possible sur {c['pair']}: Binance={c['binance_price']} BingX={c['bingx_price']} Diff={c['diff_pct']:.2f}%"
                        )
                        base_amount = 15
                        result = await bot.execute_trade(c["pair"], "BUY", base_amount)
                        bot.safe_update_shared_data(
                            {
                                "arbitrage_opportunities": [
                                    {
                                        "timestamp": get_current_time_tahiti(),
                                        "pair": c["pair"],
                                        "type": "arbitrage",
                                        "binance_price": c["binance_price"],
                                        "bingx_price": c["bingx_price"],
                                        "diff_pct": c["diff_pct"],
                                        "result": result,
                                    }
                                ]
                            },
                            bot.data_file,
                        )
                        if result and result.get("status") == "completed":
                            entry_price = safe_float(result.get("avg_price", 0))
                            await bot.plan_auto_sell(
                                c["pair"],
                                entry_price,
                                base_amount,
                                tp_pct=0.03,
                                sl_pct=0.03,
                                max_cycles=2,
                                reason="arbitrage: trailing_stop",
                            )

                    # Mise à jour des données du bot
                    bot.current_cycle = cycle
                    bot.regime = regime

                    # Calcul des indicateurs techniques
                    bot.indicators = {}
                    for pair in bot.pairs_valid:
                        pair_key = pair.replace("/", "").upper()
                        for tf in bot.config["TRADING"]["timeframes"]:
                            if (
                                pair_key in bot.market_data
                                and tf in bot.market_data[pair_key]
                            ):
                                market_data = bot.market_data[pair_key][tf]
                                df = bot.ws_collector.get_dataframe(pair_key, tf)

                                trend = bot.calculate_trend(market_data)
                                volatility = bot.calculate_volatility(market_data)
                                volume_profile = bot.calculate_volume_profile(
                                    market_data
                                )
                                dominant_signal = bot.get_dominant_signal(pair, tf)

                                indicators = (
                                    bot.add_indicators(df)
                                    if df is not None and not df.empty
                                    else {}
                                )

                                tf_key = f"{tf} | {pair}"
                                bot.indicators[tf_key] = {
                                    "trend": {"trend_strength": trend},
                                    "volatility": {"current_volatility": volatility},
                                    "volume": {"volume_profile": volume_profile},
                                    "dominant_signal": dominant_signal,
                                    "ta": indicators,
                                }

                    # Sauvegarde et dashboard
                    td_dict = {}
                    for pair in bot.pairs_valid:
                        pair_key = pair.replace("/", "").upper()
                        tech_score = 0.5
                        ai_score = 0.5
                        sentiment_score = 0.5

                        if pair_key in bot.market_data:
                            if "1h" in bot.market_data[pair_key]:
                                tf_data = bot.market_data[pair_key]["1h"]
                                if (
                                    "signals" in tf_data
                                    and "technical" in tf_data["signals"]
                                ):
                                    tech_score = safe_float(
                                        tf_data["signals"]["technical"].get(
                                            "score", 0.5
                                        )
                                    )
                            ai_score = safe_float(
                                bot.market_data[pair_key].get("ai_prediction", 0.5)
                            )
                            sentiment_score = safe_float(
                                bot.market_data[pair_key].get("sentiment", 0.5)
                            )

                        td_dict[pair] = {
                            "confidence": 0.5,
                            "action": "neutral",
                            "tech": tech_score,
                            "ai": ai_score,
                            "sentiment": sentiment_score,
                        }

                    for td in trade_decisions:
                        if td and isinstance(td, dict):
                            pair = td.get("pair")
                            signals = (
                                td.get("signals", {})
                                if td
                                and "signals" in td
                                and isinstance(td["signals"], dict)
                                else {}
                            )
                            td_dict[pair].update(
                                {
                                    "confidence": safe_float(td.get("confidence", 0.5)),
                                    "action": str(td.get("action", "neutral")),
                                    "tech": safe_float(
                                        signals.get("technical", {}).get("score", 0.5)
                                    ),
                                    "ai": safe_float(signals.get("ai", 0.5)),
                                    "sentiment": safe_float(
                                        signals.get("sentiment", 0.5)
                                    ),
                                }
                            )

                    cycle_metrics = {
                        "cycle": cycle,
                        "regime": regime,
                        "balance": bot.get_performance_metrics().get("balance", 0.0),
                    }

                    # Sauvegarde dans le fichier partagé
                    bot.safe_update_shared_data(
                        {
                            "trade_decisions": td_dict,
                            "market_data": bot.market_data,
                            "cycle_metrics": cycle_metrics,
                            "active_pauses": bot.get_active_pauses(),
                            "positions_binance": getattr(bot, "positions_binance", {}),
                        },
                        bot.data_file,
                    )

                    bot.trade_decisions = td_dict

                    print("[DEBUG DASHBOARD EXPORT]")
                    print("Trade Decisions:", json.dumps(td_dict, indent=2))
                    print("Cycle Metrics:", json.dumps(cycle_metrics, indent=2))

                    bot.safe_update_shared_data(
                        {
                            "active_pauses": active_pauses,
                            "positions_binance": getattr(bot, "positions_binance", {}),
                            "trade_decisions": td_dict,
                            "cycle_metrics": cycle_metrics,
                            "market_data": bot.market_data,
                        },
                        bot.data_file,
                    )

                    bot.save_shared_data()
                    bot.safe_update_shared_data(
                        {
                            "bot_status": {
                                "regime": bot.regime,
                                "cycle": bot.current_cycle,
                                "last_update": get_current_time(),
                                "performance": bot.get_performance_metrics(),
                            }
                        },
                        bot.data_file,
                    )

                    # Exécution des trades si pas de pause
                    if not trading_paused:
                        await execute_trade_decisions(bot, trade_decisions)
                    else:
                        print(
                            "🚫 [PAUSE] Exécution des trades bloquée, signaux et IA à jour."
                        )

                    # Entraînement IA périodique
                    if cycle % 10 == 0:
                        print(
                            "=== Entraînement automatique IA sur toutes les paires/timeframes ==="
                        )
                        bot.train_cnn_lstm_on_all_live()

                    duration = (datetime.utcnow() - start).total_seconds()
                    print(f"✅ Cycle terminé en {duration:.1f}s")

                    # Envoi du résumé synthétique Telegram
                    await send_telegram_if_needed(
                        bot,
                        cycle,
                        regime,
                        bot.get_performance_metrics(),
                        shared_data.get("sentiment", {}),
                        shared_data.get("alerts", []),
                        shared_data.get("pending_sales", []),
                        shared_data.get("active_pauses", []),
                        bot.get_performance_metrics().get("max_drawdown", 0.0),
                    )

                    # Envoi des rapports complets
                    await send_cycle_reports(
                        bot, trade_decisions, cycle, regime, duration
                    )

                except Exception as e:
                    error_msg = f"⚠️ Erreur cycle {cycle}: {e}"
                    logger.error(error_msg)
                    await bot.telegram.send_message(error_msg)

                bot.get_pending_sales()
                bot.save_shared_data()
                bot.safe_update_shared_data(
                    {
                        "bot_status": {
                            "regime": bot.regime,
                            "cycle": bot.current_cycle,
                            "last_update": get_current_time(),
                            "performance": bot.get_performance_metrics(),
                        }
                    },
                    bot.data_file,
                )
                await bot.handle_auto_sell()
                # Attente avant le prochain cycle
                await bot.send_refused_trades_summary()
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            await handle_shutdown(bot, "👋 Bot arrêté proprement")
        except Exception as e:
            await handle_shutdown(bot, f"💥 Erreur fatale: {e}")

    # Démarrage de la boucle principale
    await main()


def prepare_ohlcv_data(ohlcv_data):
    """Prépare les données OHLCV pour l'analyse"""
    try:
        if not all(
            k in ohlcv_data
            for k in ["open", "high", "low", "close", "volume", "timestamp"]
        ):
            return None

        return pd.DataFrame(
            {
                0: ohlcv_data["timestamp"],
                1: ohlcv_data["open"],
                2: ohlcv_data["high"],
                3: ohlcv_data["low"],
                4: ohlcv_data["close"],
                5: ohlcv_data["volume"],
            }
        )
    except Exception as e:
        logging.error(f"Erreur préparation OHLCV: {e}")
        return None


async def calculate_combined_score(bot, data, signal, pair):
    """Calcule le score combiné des différents signaux"""
    try:
        combined_score = 0

        # Signal technique
        if signal["action"] == "buy":
            combined_score += signal["confidence"] * 0.5
        elif signal["action"] == "sell":
            combined_score -= signal["confidence"] * 0.5

        # Signal IA
        if bot.ai_enabled:
            ai_signal = data.get("ai_prediction", 0.5)
            combined_score += (ai_signal - 0.5) * 2 * bot.ai_weight

        # Analyse des news
        if bot.news_enabled:
            sentiment_score = data.get("sentiment", 0)
            if sentiment_score != 0:
                sentiment_weight = calculate_sentiment_weight(
                    bot, data, sentiment_score
                )
                combined_score += sentiment_score * sentiment_weight

        return combined_score

    except Exception as e:
        logging.error(f"Erreur calcul score: {e}")
        return 0


def calculate_sentiment_weight(bot, data, sentiment_score):
    """Calcule le poids du sentiment en fonction de son intensité et de son âge"""
    try:
        # Poids de base
        sentiment_weight = bot.news_weight * (1 + abs(sentiment_score))

        # Amplification pour sentiments forts
        if abs(sentiment_score) > 0.7:
            sentiment_weight *= 1.5

        # Facteur temporel
        time_factor = 1.0
        if "sentiment_timestamp" in data:
            elapsed_time = time.time() - data["sentiment_timestamp"]
            time_factor = max(0.2, 1.0 - (elapsed_time / (3600 * 12)))

        return sentiment_weight * time_factor

    except Exception as e:
        logging.error(f"Erreur calcul poids sentiment: {e}")
        return bot.news_weight


async def generate_trade_decision(bot, pair, combined_score, data, signal):
    """Génère une décision de trading basée sur le score combiné"""
    try:
        # Détermination de l'action
        final_action = "neutral"
        if combined_score > 0.3:
            final_action = "buy"
        elif combined_score < -0.3:
            final_action = "sell"

        # Calcul de la confiance
        confidence = min(0.99, abs(combined_score) + 0.5)

        # Logging de la décision
        print(f"📡 {pair}: {final_action.upper()} ({confidence:.0%})")
        log_dashboard(
            f"[TRADE-DECISION] {pair} | Action: {final_action.upper()} | Confiance: {confidence:.2f} | Score: {combined_score:.4f} | Tech: {signal['confidence']:.2f} | AI: {data.get('ai_prediction', 0.5):.2f} | Sentiment: {data.get('sentiment',0):.2f}"
        )
        await bot.telegram.send_message(
            f"🔔 <b>Décision de Trade</b>\n"
            f"Pair: {pair}\n"
            f"Action: <b>{final_action.upper()}</b>\n"
            f"Confiance: {confidence:.2f}\n"
            f"Score global: {combined_score:.4f}\n"
            f"Tech: {signal['confidence']:.2f}\n"
            f"AI: {data.get('ai_prediction', 0.5):.2f}\n"
            f"Sentiment: {data.get('sentiment',0):.2f}"
        )
        # Préparation de la décision
        return {
            "pair": pair,
            "action": final_action,
            "confidence": confidence,
            "signals": {
                "technical": signal["confidence"],
                "ai": data.get("ai_prediction", 0.5),
                "sentiment": data.get("sentiment", 0),
            },
        }

    except Exception as e:
        logging.error(f"Erreur génération décision: {e}")
        return None


async def handle_arbitrage_opportunities(bot):
    """Gère la détection et l'exécution des opportunités d'arbitrage"""
    try:
        opportunities = await bot.detect_arbitrage_opportunities()
        if not opportunities:
            return

        print(f"💹 {len(opportunities)} opportunités d'arbitrage détectées")
        # log_dashboard(f"💹 {len(opportunities)} opportunités d'arbitrage détectées")
        for opp in opportunities:
            # Logging de l'opportunité
            print(
                f"  • {opp['pair']}: {opp['diff_percent']:.2f}% entre "
                f"{opp['exchange1']} et {opp['exchange2']}"
            )

            # Notification Telegram
            await bot.telegram.send_arbitrage_alert(opp)

            # Exécution si profitable
            if opp["diff_percent"] > 0.5:
                print(f"🔄 Exécution de l'arbitrage sur {opp['pair']}")
                await bot.execute_arbitrage(opp)

    except Exception as e:
        logging.error(f"Erreur gestion arbitrage: {e}")


async def execute_trade_decisions(bot, trade_decisions):
    """
    Exécute toutes les décisions de trade du cycle.
    Intègre la gestion avancée de pause news par asset/action
    et la validation par le RiskManager.
    Corrigé pour éviter tout bug de type (int + str) sur les montants.
    """
    # Vérification des news du cycle
    news_list = []
    try:
        with open(bot.data_file, "r") as f:
            shared_data = json.load(f)
        news_sentiment = (
            shared_data.get("sentiment", {}) if isinstance(shared_data, dict) else {}
        )
        news_list = (
            news_sentiment.get("scores", []) if isinstance(news_sentiment, dict) else []
        )
    except Exception as e:
        print(f"[WARNING] Erreur chargement news: {e}")
        news_list = []

    # Validation des champs obligatoires des news
    for news in news_list:
        if "symbols" not in news or not news["symbols"]:
            log_dashboard(
                f"[NEWS CHECK] ⚠️ News sans symboles: {news.get('title', '')[:80]}"
            )
        if "sentiment" not in news:
            log_dashboard(
                f"[NEWS CHECK] ⚠️ News sans sentiment: {news.get('title', '')[:80]}"
            )

    # Traitement des décisions de trade
    for decision in trade_decisions:
        try:
            pair = decision.get("pair")
            action = decision.get("action")
            confidence = safe_float(decision.get("confidence", 0))

            # Vérification pause news
            active_pauses = bot.news_pause_manager.get_active_pauses()
            if any(
                p.get("asset") == pair or p.get("asset") == "GLOBAL"
                for p in active_pauses
            ):
                log_dashboard(
                    f"[NEWS PAUSE] Trade {str(action).upper()} sur {pair} bloqué (pause news)"
                )
                await bot.telegram.send_message(
                    f"🚨 Trading {str(action).upper()} sur {pair} bloqué (pause news)"
                )
                continue

            # Validation par le RiskManager
            if not bot.risk_manager.validate_trade(decision.get("signals", {})):
                log_dashboard(
                    f"[RISK] Trade {str(action).upper()} sur {pair} rejeté "
                    "(critères de risque non respectés)"
                )
                continue

            # Calcul de la taille de position
            balance = safe_float(bot.get_performance_metrics().get("balance", 0), 0)
            volatility = safe_float(
                bot.calculate_volatility_advanced(
                    bot.market_data.get(pair.replace("/", "").upper(), {}).get("1h", {})
                ),
                0.02,
            )

            amount = safe_float(
                bot.risk_manager.calculate_position_size(
                    equity=balance, confidence=confidence, volatility=volatility
                ),
                0,
            )

            if amount <= 0:
                log_dashboard(f"[RISK] Taille position nulle pour {pair}, trade ignoré")
                continue

            # Vérification exposition totale
            if not bot.risk_manager.check_exposure_limit(bot.positions, amount):
                log_dashboard(
                    f"[RISK] Limite d'exposition dépassée pour {pair}, trade ignoré"
                )
                continue

            # Log pré-exécution
            log_dashboard(
                f"[EXECUTE] {pair} | {str(action).upper()} | "
                f"Amount: {amount:.6f} | Conf: {confidence:.2f}"
            )

            # Exécution du trade
            trade_result = await bot.execute_trade(
                symbol=pair, side=action, amount=amount
            )

            # Notification du résultat
            if trade_result and trade_result.get("status") == "completed":
                await send_trade_notification(
                    bot=bot, decision=decision, trade_result=trade_result, amount=amount
                )
                log_dashboard(f"[SUCCESS] Trade exécuté sur {pair}")
            else:
                log_dashboard(
                    f"[ERROR] Échec exécution trade sur {pair}: "
                    f"{trade_result.get('reason', 'raison inconnue')}"
                )

        except Exception as e:
            log_dashboard(f"[ERROR] Erreur exécution trade {pair}: {str(e)}")
            continue


def save_best_params(best_params, path="config/best_hyperparams.json"):
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(best_params, f, indent=2)


async def run_automl_tuning(bot, mode="cnn_lstm"):
    """Lance une optimisation AutoML/Optuna complète (manuelle ou auto)"""
    print("🔬 Lancement AutoML/Optuna...")
    import time

    start = time.time()
    if mode == "cnn_lstm":
        from src.optimization.optuna_wrapper import tune_hyperparameters

        best_params = tune_hyperparameters()
        print("✅ Optuna tuning terminé. Meilleurs hyperparams:", best_params)
        save_best_params(best_params)  # <-- Sauvegarde automatique
    elif mode == "full":
        from src.optimization.optuna_wrapper import optimize_hyperparameters_full

        best_trials = optimize_hyperparameters_full()
        print("✅ Optuna full tuning terminé. Résumé:", best_trials)
        # Si besoin, tu peux aussi sauvegarder best_trials ici
    else:
        print("❌ Mode AutoML inconnu")
        return
    duration = time.time() - start
    print(f"Durée optimisation: {duration:.1f}s")
    # (Optionnel) Recharge config/model avec les meilleurs params
    # bot.reload_model(best_params) ou autre logique
    return best_params if mode == "cnn_lstm" else best_trials


def calculate_position_size(bot, decision):
    """
    Sizing intelligent et adaptatif basé sur :
    - Confiance du signal
    - Kelly Criterion
    - Mode SAFE
    - Protection Drawdown
    - NOUVEAU: Ajustement par corrélation
    """
    try:
        # --- Configuration de base ---
        balance = bot.get_performance_metrics().get("balance", 0)
        confidence = safe_float(decision.get("confidence", 0.5))
        MIN_NOTIONAL = 5  # Minimum USDC

        # --- Sizing selon confiance ---
        if confidence > 0.7:
            risk_pct = 0.09  # 9% max
        elif confidence > 0.4:
            risk_pct = 0.04  # 4%
        else:
            risk_pct = 0.02  # 2%

        # --- Ajustement Kelly ---
        perf = bot.get_performance_metrics()
        win_rate = perf.get("win_rate", 0.55)
        profit_factor = perf.get("profit_factor", 1.7)
        kelly = kelly_criterion(win_rate, profit_factor)

        if kelly > 0:
            risk_pct = min(risk_pct + kelly * 0.5, 0.12)

        # --- NOUVEAU: Ajustement par corrélation ---
        pair = decision.get("pair")
        if pair:
            correlations = bot.calculate_correlation_matrix()
            corr_factor = max(
                [v for k, v in correlations.items() if pair in k], default=0.5
            )
            # Réduit le sizing si forte corrélation
            risk_pct *= 1 - corr_factor * 0.5

        # --- Mode SAFE ---
        try:
            with open(bot.data_file, "r") as f:
                data = json.load(f)

            recent_trades = data.get("trade_history", [])[-5:]
            losses = [t for t in recent_trades if t.get("pnl_usd", 0) < 0]
            wins = [t for t in recent_trades if t.get("pnl_usd", 0) > 0]

            mode_safe = len(losses) >= 3 and len(wins) == 0
            if mode_safe and wins:
                mode_safe = False

            bot.safe_update_shared_data({"safe_mode": mode_safe}, bot.data_file)

            if mode_safe:
                risk_pct *= 0.25
                print("[SAFE MODE] Sizing -75%")

        except Exception as e:
            print(f"[WARNING] Erreur mode safe: {e}")

        # --- Protection Drawdown ---
        try:
            with open(bot.data_file, "r") as f:
                data = json.load(f)

            equity_history = data.get("equity_history", [])
            if equity_history and len(equity_history) >= 30:
                balances = [pt["balance"] for pt in equity_history if "balance" in pt]
                peak = max(balances)
                trough = min(balances)
                drawdown = (trough - peak) / peak if peak > 0 else 0

                if drawdown < -0.15:
                    risk_pct *= 0.5
                    print(f"[DRAWDOWN] Sizing -50% (DD: {drawdown:.1%})")

        except Exception as e:
            print(f"[WARNING] Erreur drawdown: {e}")

        # --- Calcul final ---
        size = balance * risk_pct
        size = max(MIN_NOTIONAL, round(size, 2))

        print(f"[SIZING] {size:.2f} USDC ({risk_pct*100:.1f}% du capital)")
        return size

    except Exception as e:
        import logging

        logging.error(f"Erreur sizing: {e}")
        return MIN_NOTIONAL


async def send_trade_notification(bot, decision, trade_result, amount):
    """
    Envoie une notification Telegram centralisée et lisible pour un trade exécuté.
    Affiche tous les signaux clés et la confiance de la décision.
    """
    try:
        # Détermination de l'emoji selon l'action
        action = decision.get("action", "").lower()
        emoji = "🟢" if action == "buy" else "🔴" if action == "sell" else "⚪️"

        # Construction du message
        message = (
            f"{emoji} <b>TRADE EXÉCUTÉ</b>\n\n"
            f"📊 Paire : {decision.get('pair', '?')}\n"
            f"Action : <b>{action.upper()}</b>\n"
            f"Montant : {amount}\n"
            f"Prix : {trade_result.get('avg_price', 'N/A')}\n"
            f"Total : {float(amount) * float(trade_result.get('avg_price', 0)):.2f} USDT\n"
            f"Confiance : {decision.get('confidence', 0):.0%}\n"
            f"Signaux : Tech {decision.get('signals', {}).get('technical', 0):.0%} | "
            f"IA {decision.get('signals', {}).get('ai', 0):.2f} | "
            f"Sentiment {decision.get('signals', {}).get('sentiment', 0):.2f}\n"
        )
        await bot.telegram.send_message(message)

    except Exception as e:
        logging.error(f"Erreur envoi notification: {e}")


def should_send_summary(latest, previous):
    """Détermine si le résumé doit être envoyé selon le changement des données."""
    # Envoi si :
    # - Changement de régime
    # - Drawdown critique
    # - Pause active
    # - Sentiment news extrême
    # - Position à risque
    # - Au moins toutes les 6h
    last_sent = previous.get("timestamp", 0)
    now = time.time()
    if now - last_sent > 21600:  # 6h
        return True
    keys_to_check = ["regime", "drawdown", "pauses", "sentiment", "risky_positions"]
    for k in keys_to_check:
        if latest.get(k) != previous.get(k):
            return True
    return False


def build_telegram_summary(
    cycle, regime, perf, news_sentiment, alerts, trades, pauses, drawdown
):
    summary = f"🟢 <b>Résumé du cycle Trading</b>\n"
    summary += f"Cycle: <b>{cycle}</b> | Régime: <b>{regime}</b>\n"
    summary += f"Balance: <b>${safe_float(perf.get('balance', 0)):,.0f}</b> | Win Rate: <b>{safe_float(perf.get('win_rate', 0))*100:.1f}%</b>\n"

    # Drawdown critique
    if drawdown is not None and drawdown < -0.15:
        summary += f"🚨 <b>Drawdown:</b> {drawdown:.1%}\n"

    # Pauses actives (si présentes)
    if pauses:
        summary += "\n⏸️ <b>Pauses actives</b> :\n"
        for p in pauses:
            asset = p.get("asset", "GLOBAL")
            reason = p.get("reason", "Indéterminée")
            cycles_left = p.get("cycles_left", "N/A")
            summary += f"• {asset} | Raison: {reason} | Restant: {cycles_left}\n"

    # Alertes critiques (rare)
    if alerts:
        summary += "\n🚨 <b>Alertes critiques</b> :\n"
        for a in alerts:
            summary += f"• {a['message']} ({a['timestamp']})\n"

    # News sentiment (si extrême)
    if news_sentiment and abs(news_sentiment.get("overall_sentiment", 0)) > 0.7:
        summary += (
            f"\n📰 <b>Sentiment News</b>: "
            f"{news_sentiment.get('overall_sentiment', 0):.2%} ({news_sentiment.get('n_news', 0)} news)\n"
        )
        if news_sentiment.get("major_events"):
            summary += f"Événement majeur: {news_sentiment.get('major_events')}\n"

    # Positions à risque
    risky_trades = [t for t in trades if "Perte latente" in t.get("reason", "")]
    if risky_trades:
        summary += "\n📋 <b>Positions à risque</b> :\n"
        for t in risky_trades:
            summary += f"• {t['symbol']} | PnL: {t['% Gain/Perte']} | {t['reason']} | Pause: {t['pause_blocage']}\n"

    return summary


# --- Usage dans ton bot (exemple d'intégration dans le cycle principal) ---
# Charger les infos du cycle
async def send_telegram_if_needed(
    bot, cycle, regime, perf, news_sentiment, alerts, trades, pauses, drawdown
):
    try:
        with open("src/last_telegram_summary.json", "r") as f:
            previous = json.load(f)
    except Exception:
        previous = {}

    latest = {
        "regime": regime,
        "drawdown": drawdown,
        "pauses": pauses,
        "sentiment": (
            news_sentiment.get("overall_sentiment", 0) if news_sentiment else 0
        ),
        "risky_positions": [
            t["symbol"] for t in trades if "Perte latente" in t.get("reason", "")
        ],
        "timestamp": time.time(),
    }

    if should_send_summary(latest, previous):
        summary = build_telegram_summary(
            cycle, regime, perf, news_sentiment, alerts, trades, pauses, drawdown
        )
        # Envoi Telegram
        await bot.telegram.send_message(summary)
        # Sauvegarde du dernier résumé envoyé
        with open("src/last_telegram_summary.json", "w") as f:
            json.dump(latest, f, indent=2)


async def send_cycle_reports(bot, trade_decisions, current_cycle, regime, duration):
    """Envoi des rapports de cycle AVEC PAUSES, SIGNALS 1H, POSITIONS À RISQUE"""
    try:
        # 1. Rapport des trades
        await send_trade_summary(bot, trade_decisions)

        # 2. Préparation des données d'analyse
        analysis_data = await prepare_analysis_data(bot, trade_decisions)
        analysis_data.update(
            {"cycle": current_cycle, "regime": regime, "duration": duration}
        )

        # 3. Génération et envoi du rapport principal
        report = await bot.generate_market_analysis_report(cycle=current_cycle)
        await bot.telegram.send_message(report)

        # --- AJOUT : Pauses actives et leur raison ---
        pauses = bot.get_active_pauses()
        if pauses:
            msg = "⏸️ <b>Pauses actives</b> :\n"
            for pause in pauses:
                asset = pause.get("asset", "GLOBAL")
                reason = pause.get("reason", "Indéterminée")
                cycles_left = pause.get("cycles_left", "N/A")
                typ = pause.get("type", "pause")
                msg += (
                    f"• Paire: <b>{asset}</b> | Type: {typ} | Raison: <i>{reason}</i> | "
                    f"Restant: {cycles_left} cycles\n"
                )
            await bot.telegram.send_message(msg)

        # --- AJOUT : Signaux uniquement sur 1h ---
        message_1h = "🔎 <b>Signaux 1h par paire</b> :\n"
        for pair in bot.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            signals_1h = bot.market_data.get(pair_key, {}).get("1h", {})
            tech = signals_1h.get("signals", {}).get("technical", {}).get("score", 0.5)
            ai = bot.market_data.get(pair_key, {}).get("ai_prediction", 0.5)
            sentiment = bot.market_data.get(pair_key, {}).get("sentiment", 0.5)
            message_1h += (
                f"{pair}: Tech={tech:.2f} | AI={ai:.2f} | Sentiment={sentiment:.2f}\n"
            )
        await bot.telegram.send_message(message_1h)

        # --- AJOUT : Positions à risque ou en attente ---
        pending = bot.get_pending_sales()
        if pending:
            msg = "📋 <b>Positions à risque ou en attente</b> :\n"
            for pos in pending:
                msg += (
                    f"• {pos['symbol']} | {pos['reason']} | PnL: {pos['% Gain/Perte']} | "
                    f"Pause: {pos['pause_blocage']} | Note: {pos['note']}\n"
                )
            await bot.telegram.send_message(msg)

        # 4. Sauvegarde des données
        await save_cycle_data(bot, analysis_data)

        # 5. Vérification des alertes
        await check_risk_alerts(bot, analysis_data)

    except Exception as e:
        print(f"❌ Erreur envoi rapports: {e}")


async def send_trade_summary(bot, trade_decisions):
    """Envoi du résumé des trades"""
    try:
        if trade_decisions:
            trade_report = "💹 <b>Résumé des trades du cycle</b>\n\n"
            for trade in trade_decisions:
                emoji = (
                    "🟢"
                    if trade["action"] == "buy"
                    else "🔴" if trade["action"] == "sell" else "⚪️"
                )
                pair = trade.get("pair", "INCONNU")
                signals = trade.get("signals", {})

                trade_report += (
                    f"{emoji} {pair}: {trade['action'].upper()} "
                    f"({trade.get('confidence', 0):.0%}) | "
                    f"Tech {signals.get('technical', 0):.0%} | "
                    f"IA {signals.get('ai', 0):.2f} | "
                    f"Sent {signals.get('sentiment', 0):.2f}\n"
                )
            await bot.telegram.send_message(trade_report)
    except Exception as e:
        print(f"❌ Erreur envoi résumé trades: {e}")


async def prepare_analysis_data(bot, trade_decisions):
    """Préparation des données d'analyse"""
    # Construction des analyses par timeframe/paire
    indicators_analysis = {}
    trade_decisions_dict = {}

    # Analyse des indicateurs
    for pair in bot.pairs_valid:
        pair_key = pair.replace("/", "").upper()
        for tf in bot.config["TRADING"]["timeframes"]:
            tf_key = f"{tf} | {pair}"
            indics = bot.indicators.get(pair_key, {}).get(tf, {})
            indicators_analysis[tf_key] = indics if indics else {}

    # Organisation des décisions de trade
    for decision in trade_decisions:
        tf = decision.get("tf", "1h")
        pair = decision.get("pair", "")
        tf_key = f"{tf} | {pair}"

        trade_decisions_dict[tf_key] = {
            "pair": pair,
            "tf": tf,
            "action": decision.get("action", "NEUTRAL").upper(),
            "confidence": decision.get("confidence", 0),
            "tech": decision.get("signals", {}).get("technical", 0),
            "ai": decision.get("signals", {}).get("ai", 0),
            "sentiment": decision.get("signals", {}).get("sentiment", 0),
        }

    # Métriques avancées (intégration de track_advanced_metrics)
    advanced_metrics = bot.track_advanced_metrics()

    return {
        "indicators": indicators_analysis,
        "decisions": trade_decisions_dict,
        "metrics": advanced_metrics,
        "perf": bot.get_performance_metrics(),
    }


async def save_cycle_data(bot, analysis_data):
    """Sauvegarde des données du cycle"""
    try:
        with open(bot.data_file, "r") as f:
            data = json.load(f)

        # Mise à jour des données
        equity_history = data.get("equity_history", [])
        equity_history.append(
            {
                "timestamp": get_current_time_tahiti(),
                "balance": analysis_data["perf"].get("balance", 0),
                "metrics": analysis_data["metrics"],  # Ajout des métriques avancées
            }
        )

        # Limitation de l'historique
        if len(equity_history) > 1000:
            equity_history = equity_history[-1000:]

        # Sauvegarde sécurisée
        bot.safe_update_shared_data(
            {
                "trade_decisions": analysis_data["decisions"],
                "equity_history": equity_history,
                "positions_binance": getattr(bot, "positions_binance", {}),
                "advanced_metrics": analysis_data["metrics"],
            },
            bot.data_file,
        )

    except Exception as e:
        logging.error(f"Erreur sauvegarde données: {e}")


async def check_risk_alerts(bot, analysis_data):
    """Vérification et envoi des alertes de risque"""
    try:
        equity_history = analysis_data.get("equity_history", [])
        perf = analysis_data["perf"]

        # 1. Alerte Kelly
        kelly = kelly_criterion(
            win_rate=perf.get("win_rate", 0), payoff_ratio=perf.get("profit_factor", 1)
        )
        if abs(kelly) > 0.5:
            await bot.telegram.send_message(
                f"⚠️ Kelly fraction élevée: {kelly:.2f}\n"
                f"Réduction recommandée du sizing!"
            )

        # 2. Alerte Drawdown
        equity_curve = [
            pt.get("balance", 0) for pt in equity_history if pt.get("balance", 0) > 0
        ]
        if equity_curve and len(equity_curve) > 10:
            max_dd = calculate_max_drawdown(np.array(equity_curve))
            if max_dd < -0.15:
                await bot.telegram.send_message(
                    f"🚨 Drawdown critique: {max_dd:.2%}\n"
                    f"Actions recommandées:\n"
                    f"- Réduction du sizing\n"
                    f"- Pause trading conseillée"
                )

        # 3. Alerte VaR
        if len(equity_curve) > 10:
            try:
                equity_curve_np = np.array(equity_curve)
                returns = np.diff(equity_curve_np) / equity_curve_np[:-1]
                var95 = calculate_var(returns, 0.05)
                if var95 < -0.05:
                    await bot.telegram.send_message(
                        f"🛑 VaR(95%) critique: {var95:.2%}\n"
                        f"Risque de perte important!"
                    )
            except Exception:
                pass

    except Exception as e:
        logging.error(f"Erreur alertes risque: {e}")


async def handle_shutdown(bot, message):
    """Gère l'arrêt propre du bot"""
    try:
        print(f"\n{message}")
        await bot.telegram.send_message(message)
        await bot.ws_collector.stop()
        bot.save_shared_data()
        bot.safe_update_shared_data(
            {
                "bot_status": {
                    "regime": bot.regime,
                    "cycle": bot.current_cycle,
                    "last_update": get_current_time(),
                    "performance": bot.get_performance_metrics(),
                }
            },
            bot.data_file,
        )
    except Exception as e:
        logging.error(f"Erreur arrêt bot: {e}")


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    batch_size = int(batch_size)  # <-- PATCH FONDAMENTAL
    n_epochs = 5
    pairs = load_config()
    scores = []
    for pair in pairs:
        try:
            print(f"[Optuna] TRAIN sur {pair}…")
            model = HybridAI(
                pair=pair,
                window=30,
                interval="1h",
                start_str="1 Jan, 2023",
                end_str="now",
                cache_dir="data_cache",
            )
            acc = model.validate(lr=lr, batch_size=batch_size, n_epochs=n_epochs)
            print(f"[Optuna] {pair} | Accuracy={acc:.4f}")
            scores.append(acc)
        except Exception as e:
            print(f"[Optuna] Erreur sur {pair}: {e}")
    if not scores:
        print("[Optuna] Aucune paire dispo pour ce trial !")
    return float(sum(scores)) / len(scores) if scores else 0.0


if __name__ == "__main__":

    # --- 1. Argument parsing avancé
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtest", action="store_true", help="Lancer un backtest quantitatif"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/historical/BTCUSDT_1h.csv",
        help="Chemin du CSV market data",
    )
    parser.add_argument("--capital", type=float, default=0, help="Capital initial")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sma",
        choices=["sma", "breakout", "arbitrage"],
        help="Stratégie à utiliser",
    )
    parser.add_argument(
        "--auto-strategy",
        action="store_true",
        help="Active l'auto-stratégie (recherche + utilisation)",
    )
    parser.add_argument(
        "--auto-pair",
        type=str,
        default="BTCUSDT",
        help="Paire à utiliser pour l'auto-stratégie",
    )
    parser.add_argument(
        "--auto-timeframe",
        type=str,
        default="1h",
        help="Timeframe à utiliser pour l'auto-stratégie",
    )
    parser.add_argument(
        "--auto-days",
        type=int,
        default=30,
        help="Nombre de jours d'historique pour l'auto-stratégie",
    )
    parser.add_argument(
        "--auto-n", type=int, default=50, help="Nombre de stratégies à générer/tester"
    )
    parser.add_argument(
        "--optuna-signal-fusion",
        action="store_true",
        help="Lance l'optimisation AutoML des pondérations de signaux",
    )
    args, unknown = parser.parse_known_args()

    # --- 2. Mode AutoML/Tuning (prioritaire sur tout le reste)
    if "automl" in sys.argv or "tune" in sys.argv:
        asyncio.run(run_automl_tuning(None, mode="cnn_lstm"))

    # --- 2bis. Mode Optuna signal fusion
    elif args.optuna_signal_fusion:
        print("=== Lancement Optuna signal fusion (diagnostic print) ===")
        from src.optimization.signal_fusion_automl import optimize_signal_fusion_and_mm

        optimize_signal_fusion_and_mm(n_trials=100)
        exit(0)

    # --- 3. Mode auto-strategy (AUTO-ML stratégies)
    elif "auto-strategy" in sys.argv:
        # Paramètres pour Binance
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        symbol = args.auto_pair.upper()
        tf_str = args.auto_timeframe.lower()
        interval = getattr(Client, f"KLINE_INTERVAL_{tf_str.upper()}")
        nb_days = args.auto_days

        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=nb_days)
        start_str = start_dt.strftime("%d %b %Y")
        end_str = end_dt.strftime("%d %b %Y")

        # Récupère les données Binance
        df = fetch_binance_ohlcv(
            symbol,
            interval,
            start_str,
            end_str,
            api_key=api_key,
            api_secret=api_secret,
        )
        if df is None or len(df) == 0:
            print("Aucune donnée récupérée sur Binance, impossible d’auto-stratégie.")
            sys.exit(1)

        df.columns = [col.lower() for col in df.columns]  # Sécurité
        best_config, best_score = auto_generate_and_backtest(df, n_strats=args.auto_n)
        print("Meilleure stratégie trouvée :", best_config)
        print("Score (profit brut sur l'historique):", best_score)

        # Sauvegarde pour usage live
        if not os.path.exists("config"):
            os.makedirs("config", exist_ok=True)
        with open("config/auto_strategy.json", "w") as f:
            json.dump(
                {
                    "pair": symbol,
                    "timeframe": tf_str,
                    "config": best_config,
                    "score": best_score,
                    "date": datetime.utcnow().isoformat(),
                },
                f,
                indent=4,
            )

        # Envoi rapport Telegram
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            from src.bot_runner import TelegramNotifier, get_current_time, CURRENT_USER
            import asyncio

            notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            rapport = (
                f"🔬 <b>Auto-Strategy Report</b>\n\n"
                f"Paire: <b>{symbol}</b>\nTimeframe: <b>{tf_str}</b>\n"
                f"Meilleure config trouvée : <code>{best_config}</code>\n"
                f"Score (profit brut): <b>{best_score:.2f}</b>\n"
                f"Date: {get_current_time()}\n"
                f"Utilisateur: {CURRENT_USER}"
            )
            asyncio.run(notifier.send_message(rapport))

        sys.exit(0)

    # --- 4. Mode backtest CLI
    elif args.backtest:
        print("=== Lancement du backtesting quantitatif ===")
        # 1. Charge les paires depuis la config
        config_path = "config/trading_pairs.json"
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            pairs = config.get("valid_pairs", ["BTC/USDT"])
        except Exception as e:
            print("Impossible de charger la config, on utilise BTC/USDT.")
            pairs = ["BTC/USDT"]

        # 2. Définis la période à backtester
        nb_days = 30
        end_dt = pd.Timestamp.utcnow()
        start_dt = end_dt - pd.Timedelta(days=nb_days)
        interval = Client.KLINE_INTERVAL_1HOUR

        # 3. Stratégies
        strategy_map = {
            "sma": sma_strategy,
            "breakout": breakout_strategy,
            "arbitrage": arbitrage_strategy,
        }
        strategy_func = strategy_map.get(args.strategy, sma_strategy)

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        for pair in pairs:
            symbol = pair.replace("/", "")
            print(f"Téléchargement des données pour {symbol}...")
            df = fetch_binance_ohlcv(
                symbol,
                interval,
                start_dt.strftime("%d %b %Y"),
                end_dt.strftime("%d %b %Y"),
                api_key=api_key,
                api_secret=api_secret,
            )
            if df is None or len(df) == 0:
                print(f"Données manquantes pour {pair}, backtest ignoré.")
                continue

            engine = BacktestEngine(initial_capital=args.capital)
            print(f"Backtest sur {pair} ({len(df)} lignes)...")
            results = engine.run_backtest(df, strategy_func)
            print(f"Résultats du backtest pour {pair} :")
            print(results)
        sys.exit(0)

    # --- 5. Entraînement IA live
    elif "train-cnn-lstm" in sys.argv:
        bot = TradingBotM4()
        # Préchargement historique pour chaque paire/timeframe avant entraînement IA
        if hasattr(bot, "ws_collector") and hasattr(bot, "binance_client"):
            for symbol in bot.pairs_valid:
                symbol_binance = symbol.replace("/", "").upper()
                for tf in bot.config["TRADING"]["timeframes"]:
                    try:
                        bot.ws_collector.preload_historical(
                            bot.binance_client, symbol_binance, tf, limit=2000
                        )
                        print(f"Préchargement {symbol_binance} {tf} OK")
                    except Exception as e:
                        print(f"Erreur préchargement {symbol_binance} {tf} : {e}")
        # Lancement de l'entraînement IA sur les données chargées
        bot.train_cnn_lstm_on_all_live()
        sys.exit(0)

    # --- 6. Lancement du bot de trading en mode normal
    else:
        asyncio.run(run_clean_bot())
