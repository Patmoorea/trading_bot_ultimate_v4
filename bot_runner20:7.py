import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import warnings
import logging
import json
import asyncio
import aiohttp
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import argparse
import pandas as pd
import pandas_ta as pta
import pyarrow as pa
import pyarrow.parquet as pq

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
from src.analysis.news.sentiment_analyzer import NewsSentimentAnalyzer
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

# Charger les variables d'environnement depuis .env
load_dotenv()

LOG_FILE = "src/bot_logs.txt"


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

    major_news = news_sentiment.get("latest_news", []) if news_sentiment else []
    if major_news:
        report.append("Dernières news :")
        for news in major_news[:3]:
            report.append(f"- {news}")

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
    parser.add_argument("--capital", type=float, default=10000, help="Capital initial")
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

# Supprimer TOUS les warnings Python
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

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
    utc_now = datetime.utcnow()
    polynesie_offset = timedelta(hours=-10)
    local_dt = utc_now + polynesie_offset
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")


# Constantes
CURRENT_TIME = get_current_time()
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
        return self.N_FEATURES * self.N_STEPS * len(self.pairs_valid)

    async def send_message(self, message):
        """Envoie un message sur Telegram"""
        if not self.bot_token or not self.chat_id:
            print("⚠️ Message non envoyé: Configuration Telegram manquante")
            return

        header = (
            f"🕒 {get_current_time()}\n"
            f"👤 {CURRENT_USER}\n"
            "------------------------\n"
        )
        full_message = header + message

        MAX_TELEGRAM_LENGTH = 4000
        if len(full_message) > MAX_TELEGRAM_LENGTH:
            full_message = (
                full_message[: MAX_TELEGRAM_LENGTH - 20]
                + "\n... (troncature automatique)"
            )

        url = f"{self.base_url}/sendMessage"
        data = {"chat_id": self.chat_id, "text": full_message, "parse_mode": "HTML"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    result = await response.json()
                    if not result.get("ok"):
                        print(f"⚠️ Erreur Telegram: {result.get('description')}")
                    return result
        except Exception as e:
            print(f"⚠️ Erreur envoi Telegram: {e}")

    async def send_performance_update(self, performance_data):
        message = (
            "🤖 <b>Trading Bot Status Update</b>\n\n"
            f"💰 Balance: ${safe(performance_data.get('balance'))}\n"
            f"📊 Win Rate: {safe(performance_data.get('win_rate', 0)*100, 'N/A', '{:.1f}')}%\n"
            f"📈 Profit Factor: {safe(performance_data.get('profit_factor'))}\n"
            f"🔄 Total Trades: {safe(performance_data.get('total_trades'), 'N/A', '{:d}')}\n"
        )
        await self.send_message(message)

    async def send_cycle_update(self, cycle, regime, duration):
        """Envoie une mise à jour du cycle"""
        message = (
            "🔄 <b>Cycle Update</b>\n\n"
            f"📊 Cycle: {cycle}\n"
            f"🎯 Régime: {regime}\n"
            f"⏱️ Durée: {duration:.1f}s\n"
        )
        await self.send_message(message)

    async def send_trade_alert(self, trade_data):
        """Envoie un message unique et lisible pour chaque trade exécuté"""
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
        """Envoie une alerte d'opportunité d'arbitrage"""
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

        # Mapping emoji par source
        source_emoji = {
            "CoinDesk": "📰",
            "Cointelegraph": "🟣",
            "Decrypt": "🟦",
            "Binance": "🟡",
            "Twitter": "🐦",
            "default": "🗞️",
        }

        # Filtrage avancé selon symboles ou volatilité
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

        # Si rien ne passe le filtre, utilise tout
        if not filtered_news:
            filtered_news = news_data

        message = "📰 <b>Dernières Nouvelles Importantes</b>\n\n"

        # Ajoute le résumé IA si fourni
        if ai_summary:
            message += f"🤖 <b>Résumé IA:</b>\n{ai_summary}\n\n"

        def real_translate_title(title):
            try:
                return GoogleTranslator(source="auto", target="fr").translate(title)
            except Exception:
                return title

        def translate_title(title):
            original = title
            dico = {
                "Bitcoin": "Bitcoin",
                "Ethereum": "Ethereum",
                "price": "prix",
                "update": "mise à jour",
                "reaches": "atteint",
                "falls": "chute",
                "surges": "explose",
                "network": "réseau",
                "record": "record",
                "launch": "lancement",
                "approval": "approbation",
                "hack": "piratage",
                "coin": "jeton",
                "exchange": "plateforme",
                "regulation": "réglementation",
                "ETF": "ETF",
                "market": "marché",
                "crash": "effondrement",
                "rise": "hausse",
                "buy": "achat",
                "sell": "vente",
                "token": "jeton",
                "trading": "trading",
                "volume": "volume",
                "support": "support",
                "resistance": "résistance",
            }
            for en, fr in dico.items():
                title = title.replace(en, fr)

            if title == original:
                try:
                    from deep_translator import GoogleTranslator

                    return GoogleTranslator(source="auto", target="fr").translate(title)
                except Exception:
                    return title

            return title

        # Remplacer [:5] par rien pour prendre tous les titres
        for news in filtered_news:
            src = news.get("source", "default")
            emoji = source_emoji.get(src, source_emoji["default"])
            title = news.get("title", "NO_TITLE")
            url = news.get("url", "")
            # Traduction simplifiée
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

    def write(self, message):
        if any(
            word in message()
            for word in [
                "warning",
                "scriptruncontext",
                "missing",
                "streamlit",
                "pair",
                "not available",
                "skipping",
            ]
        ):
            return
        self.original_stderr.write(message)

    def flush(self):
        self.original_stderr.flush()


sys.stderr = WarningFilter(sys.stderr)


def get_sentiment_summary_from_batch(sentiment_scores, top_n=5):
    import numpy as np

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
    sentiments = [item["sentiment"] for item in valid]
    sentiment_global = float(np.mean(sentiments))
    # Top news (par score absolu)
    top_news = sorted(valid, key=lambda x: abs(x["sentiment"]), reverse=True)[:top_n]
    top_news_titles = [news["title"] for news in top_news if "title" in news]
    # Top symbols (fréquence + score fort)
    symbol_scores = {}
    for item in valid:
        for s in item.get("symbols", []):
            symbol_scores.setdefault(s, []).append(item["sentiment"])
    top_symbols = sorted(
        symbol_scores.items(), key=lambda kv: abs(np.mean(kv[1])), reverse=True
    )
    top_symbols = [s for s, scores in top_symbols[:top_n]]
    return {
        "sentiment_global": sentiment_global,
        "n_news": len(valid),
        "top_symbols": top_symbols,
        "top_news": top_news_titles,
    }


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
                "low_watermark_ratio": 0.2,
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
                },
            },
        }
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
        self.data_file = SHARED_DATA_PATH
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
        self.risk_manager = RiskManager()

        # Configuration de l'arbitrage
        self.arbitrage_config = {
            "min_profit": 0.5,
            "max_exposure": 10000,
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

    def is_short(self, symbol):
        return self.positions.get(symbol, {}).get("side") == "short"

    # Ajoute cette méthode pour savoir si on est long
    def is_long(self, symbol):
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
        Récupère le carnet d'ordres (bid/ask) depuis le ws_collector (WebSocket) ou via Binance API.
        - symbol : exemple 'BTCUSDC'
        Retourne : tuple (bid, ask) ou (None, None) si non dispo.
        """
        try:
            # Méthode ws_collector (stub ou réelle)
            if hasattr(self, "ws_collector") and self.ws_collector is not None:
                bid, ask = self.ws_collector.get_orderbook(symbol)
                # Si les valeurs existent et sont numériques, retourne-les
                if bid is not None and ask is not None:
                    return float(bid), float(ask)
                # Sinon, tente la récupération via Binance API si disponible
            # Fallback sur Binance API réelle (live trading uniquement)
            if (
                getattr(self, "is_live_trading", False)
                and hasattr(self, "binance_client")
                and self.binance_client is not None
            ):
                try:
                    ob = self.binance_client.get_order_book(symbol=symbol, limit=5)
                    best_bid = float(ob["bids"][0][0]) if ob["bids"] else None
                    best_ask = float(ob["asks"][0][0]) if ob["asks"] else None
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
                self.dl_model_last_mtime = mtime
                print(f"♻️ Nouveau modèle IA chargé automatiquement ({path})")

    async def _news_analysis_loop(self):
        log_dashboard("[NEWS] Lancement boucle d'analyse des news…")
        while True:
            try:
                if not self.news_enabled or not self.news_analyzer:
                    await asyncio.sleep(self.news_update_interval)
                    continue

                self.logger.info("Fetching latest news for sentiment analysis")
                news_data = await self.news_analyzer.fetch_all_news()

                sentiment_analysis = {}
                try:
                    sentiment_analysis = await self.news_analyzer.update_analysis()
                except Exception:
                    self.logger.error("Erreur update_analysis", exc_info=True)
                    # sentiment_analysis reste {}

                # Extract the items list from the analysis result
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
                    await self.telegram.send_news_summary(news_data[:5])
                except Exception:
                    pass

                # === LOG SENTIMENT GLOBAL ===
                try:
                    with open(self.data_file, "r") as f:
                        shared_data = json.load(f)
                    sentiment_data = shared_data.get("sentiment", {})
                    avg_sentiment = sentiment_data.get("overall_sentiment", 0)
                    impact_score = sentiment_data.get("impact_score", 0)
                    major_events = sentiment_data.get("major_events", "")

                    log_dashboard(
                        f"[NEWS] Score sentiment global: {avg_sentiment:.2f} | Impact: {impact_score:.2f} | Événements: {major_events}"
                    )
                except Exception as e:
                    print(
                        f"[NEWS] Impossible d'afficher le score sentiment global: {e}"
                    )

            except Exception as e:
                self.logger.error(f"News analysis error: {e}")

            await asyncio.sleep(self.news_update_interval)

    async def analyze_signals(self, symbol, ohlcv_df, indicators):
        """
        Analyse la paire, retourne la décision réelle (plus aucun patch de test IA)
        """
        if hasattr(self, "auto_strategy_config") and self.auto_strategy_config:
            auto_cfg = self.auto_strategy_config
            log_dashboard(f"[STRATEGY] Stratégie AUTO-GÉNÉRÉE appliquée pour {symbol}")
        else:
            log_dashboard(f"[STRATEGY] Stratégie STANDARD appliquée")

        # Indicateurs techniques
        close = ohlcv_df["close"].iloc[-1] if "close" in ohlcv_df else None
        prev_close = (
            ohlcv_df["close"].iloc[-2]
            if "close" in ohlcv_df and len(ohlcv_df) >= 2
            else None
        )
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        ema_20 = indicators.get("ema_20")
        rsi_14 = indicators.get("rsi_14")
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        macd_hist = indicators.get("macd_hist")
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        psar = indicators.get("psar")
        momentum_10 = indicators.get("momentum_10")
        zscore_20 = indicators.get("zscore_20")

        tech_score = 0
        tech_factors = 0

        # Calcul des scores techniques avec amplification
        if close and sma_20:
            tech_factors += 1
            pct_diff = (close - sma_20) / sma_20 * 100  # Pourcentage de différence
            tech_score += np.clip(pct_diff * 2, -1, 1)  # Amplifier et limiter

        if close and sma_50:
            tech_factors += 1
            pct_diff = (close - sma_50) / sma_50 * 100
            tech_score += np.clip(pct_diff * 1.5, -1, 1)

        if close and ema_20:
            tech_factors += 1
            pct_diff = (close - ema_20) / ema_20 * 100
            tech_score += np.clip(pct_diff * 2.5, -1, 1)

        if rsi_14:
            tech_factors += 1
            # RSI plus réactif : suracheté/survendu plus marqué
            if rsi_14 > 70:
                tech_score -= 0.8  # Signal de vente fort
            elif rsi_14 < 30:
                tech_score += 0.8  # Signal d'achat fort
            else:
                tech_score += (rsi_14 - 50) / 25  # Signal graduel

        if macd and macd_signal:
            tech_factors += 1
            macd_diff = macd - macd_signal
            tech_score += np.clip(macd_diff * 10, -1, 1)  # Amplifier MACD

        if macd_hist:
            tech_factors += 1
            tech_score += np.clip(macd_hist * 15, -1, 1)  # Amplifier histogramme

        if bb_upper and bb_lower and close:
            tech_factors += 1
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)
            if bb_position < 0.2:
                tech_score += 0.6  # Proche de la bande inférieure = achat
            elif bb_position > 0.8:
                tech_score -= 0.6  # Proche de la bande supérieure = vente

        if psar and prev_close and close:
            tech_factors += 1
            if prev_close < psar and close > psar:
                tech_score += 0.8  # Signal d'achat fort
            elif prev_close > psar and close < psar:
                tech_score -= 0.8  # Signal de vente fort

        if momentum_10 and close:
            tech_factors += 1
            momentum_pct = momentum_10 / close * 100
            tech_score += np.clip(momentum_pct * 5, -1, 1)

        if zscore_20:
            tech_factors += 1
            tech_score += np.clip(zscore_20 * 0.5, -1, 1)

        # Moyenne pondérée au lieu de simple division
        if tech_factors > 0:
            tech_score = tech_score / tech_factors
            # Amplifier le score final si plusieurs indicateurs convergent
            if abs(tech_score) > 0.3:
                tech_score *= 1.2

        # === IA réelle uniquement ===
        ai_score = 0
        if self.ai_enabled and hasattr(self, "dl_model") and self.dl_model:
            log_dashboard(f"[AI] Prédiction IA sollicitée pour {symbol}")
            try:
                features = await self._prepare_features_for_ai(symbol)
                # log_dashboard(f"[DEBUG AI FEATURES] features: {features}")
                if features is not None:
                    ai_score = float(self.dl_model.predict(features))
            except Exception as e:
                self.logger.warning(f"Erreur IA analyse_signals: {e}")

        sentiment_score = 0
        pair_key = symbol.replace("/", "").upper()
        if getattr(self, "news_enabled", False) and hasattr(self, "news_analyzer"):
            # Utilise le score par symbole si disponible
            try:
                sentiment_score = await self.news_analyzer.get_symbol_sentiment(
                    pair_key
                )
                # Fallback sur le global si rien de spécifique
                if sentiment_score == 0:
                    sentiment_score = self.news_analyzer.get_sentiment_summary().get(
                        "sentiment_global", 0.0
                    )
            except Exception as e:
                self.logger.error(f"Erreur récupération sentiment pour {pair_key}: {e}")
                sentiment_score = self.news_analyzer.get_sentiment_summary().get(
                    "sentiment_global", 0.0
                )
        log_dashboard(
            f"[DEBUG SENTIMENT] symbol={symbol} | sentiment_score={sentiment_score} | news_enabled={getattr(self, 'news_enabled', False)}"
        )

        arbitrage_score = 0

        # Amplifier les scores pour avoir des décisions plus marquées
        total_score = (
            0.4 * tech_score * 1.2  # Amplification réduite
            + 0.3 * ai_score * 1.0  # Pas d'amplification
            + 0.3 * sentiment_score * 1.5  # Amplification réduite + poids augmenté
            + 0.05 * arbitrage_score
        )

        decision = {
            "action": "neutral",
            "confidence": abs(total_score),
            "signals": {
                "technical": tech_score,
                "ai": ai_score,
                "sentiment": sentiment_score,
            },
        }
        if total_score > 0.2:  # Seuil abaissé
            decision["action"] = "buy"
        elif total_score < -0.2:  # Seuil abaissé
            decision["action"] = "sell"

        # Log détaillé pour debug
        log_dashboard(
            f"[ANALYZE_SIGNALS] {symbol} | "
            f"Tech: {tech_score:.3f} | AI: {ai_score:.3f} | Sentiment: {sentiment_score:.3f} | "
            f"Total: {total_score:.3f} | Action: {decision['action'].upper()} | "
            f"Confidence: {decision['confidence']:.3f}"
        )
        log_dashboard(
            f"[DEBUG SENTIMENT] symbol={symbol} | sentiment_score={sentiment_score} | news_enabled={getattr(self, 'news_enabled', False)}"
        )
        print(f"### SENTIMENT CHECK {symbol} = {sentiment_score}")
        return decision

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
        Détecte les opportunités d'arbitrage cross-quote USDC/USDT :
        Compare par exemple BTC/USDC sur Binance à BTC/USDT sur les autres brokers,
        avec adaptation du format des symboles selon chaque broker.
        """

        def get_broker_symbol(coin, quote, broker):
            # Adapter au format attendu par chaque broker
            if broker == "binance":
                return f"{coin}{quote}"
            elif broker in ["okx", "bingx"]:
                return f"{coin}-{quote}"
            elif broker == "gateio":
                return f"{coin}_{quote}"
            elif broker == "blofin":
                return f"{coin}{quote}"  # Si différent, adapter ici !
            else:
                return f"{coin}/{quote}"  # Fallback

        if not self.is_live_trading:
            log_dashboard("[ARBITRAGE] Pas en mode live trading, détection annulée.")
            return []
        log_dashboard("[ARBITRAGE] Démarrage détection arbitrage USDC/USDT…")
        opportunities = []
        pairs_to_check = [pair] if pair else self.pairs_valid
        MIN_PROFIT_THRESHOLD = 0.15
        MIN_VOLUME_USD = 10000
        MAX_SPREAD = 0.5

        try:
            for current_pair in pairs_to_check:
                try:
                    coin = current_pair.split("/")[0]
                    # Symboles pour chaque broker
                    binance_symbol = get_broker_symbol(coin, "USDC", "binance")

                    # Prix Binance (USDC)
                    binance_ticker = self.binance_client.get_ticker(
                        symbol=binance_symbol
                    )
                    binance_price = float(binance_ticker.get("lastPrice") or 0)
                    binance_volume = float(binance_ticker.get("volume", 0))

                    # Liste des brokers à comparer (USDT)
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
                            # Récupération du prix sur l'autre broker (USDT)
                            ticker = await exchange["client"].fetch_ticker(other_symbol)
                            exchange_price = float(ticker["last"])
                            if not exchange_price or not binance_price:
                                continue

                            # Calcul du spread
                            price_diff = exchange_price - binance_price
                            profit_pct = (
                                (price_diff / binance_price) * 100
                                if binance_price > 0
                                else 0
                            )

                            # Opportunité cross-quote
                            if profit_pct > MIN_PROFIT_THRESHOLD:
                                opportunity = {
                                    "pair": coin,
                                    "exchange1": "Binance (USDC)",
                                    "exchange2": f"{broker} (USDT)",
                                    "price1": binance_price,
                                    "price2": exchange_price,
                                    "diff_percent": profit_pct,
                                    "volume_24h": binance_volume * binance_price,
                                    "estimated_profit": profit_pct - 0.2,  # Après frais
                                    "route": f"Buy {coin}/USDC (Binance) -> Transfer {coin} -> Sell {coin}/USDT ({broker})",
                                }
                                log_dashboard(
                                    f"[ARBITRAGE] OPPORTUNITÉ: {coin}: {binance_price} (Binance USDC) <> {exchange_price} ({broker} USDT) | Diff: {profit_pct:.2f}%"
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
            # Utiliser l'ArbitrageExecutor existant
            result = await self.arbitrage_executor.execute(
                opportunity=opportunity,
                max_slippage=0.1,  # 0.1% de slippage maximum
                timeout=5,  # 5 secondes maximum
            )

            if result["success"]:
                profit = result["realized_profit"]
                message = (
                    f"✅ Arbitrage réussi!\n"
                    f"💰 Profit: {profit:.2f} USDT\n"
                    f"📊 Paire: {opportunity['pair']}\n"
                    f"🔄 Route: {opportunity['route']}"
                )
                await self.telegram.send_message(message)

                # Mettre à jour les statistiques
                self._update_performance_metrics(
                    {"type": "arbitrage", "profit": profit, "pair": opportunity["pair"]}
                )
            else:
                self.logger.warning(f"Échec arbitrage: {result['error']}")

        except Exception as e:
            self.logger.error(f"Erreur exécution arbitrage: {e}")

        def secure_withdraw(self, address, amount, asset):
            # Cette fonction serait appelée avant tout transfert sortant
            # Demande la signature de l'opération
            message = f"{address}|{amount}|{asset}|{get_current_time()}"
            signature = self.key_manager.sign_message(message)
            # Ici, tu pourrais envoyer la requête à l'exchange avec la signature pour logs/sécurité
            print(
                f"Retrait sécurisé demandé : {amount} {asset} vers {address}, signature: {signature}"
            )
            # (A compléter: appel API exchange avec signature, ou log d'audit)
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

    # Ajoute la méthode get_input_dim à ta classe TradingBotM4 si ce n'est pas déjà fait :
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
        Vérifie si le stop-loss doit être déclenché pour la position SPOT sur le symbole.
        - Retourne True si le stop doit être déclenché (perte dépassant le seuil)
        - Le seuil par défaut est self.stop_loss_pct (ex : 0.03 pour -3%)
        - Utilise le prix d'entrée mémorisé dans self.positions[symbol]['entry_price']
        - Utilise le dernier prix marché si price n'est pas fourni
        """
        try:
            # Vérifie s'il y a une position ouverte (long)
            pos = self.positions.get(symbol)
            if not pos or pos.get("side") != "long":
                return False
            entry = pos.get("entry_price")
            if entry is None:
                return False

            # Détermination du prix courant
            if price is None:
                # Essaye de récupérer le prix via WebSocket collector s'il existe
                price = None
                if hasattr(self, "ws_collector") and self.ws_collector is not None:
                    price = self.ws_collector.get_last_price(symbol)
                # Fallback sur market_data
                if (
                    price is None
                    and symbol in self.market_data
                    and "1h" in self.market_data[symbol]
                ):
                    closes = self.market_data[symbol]["1h"].get("close", [])
                    if closes:
                        price = closes[-1]
            if price is None:
                self.logger.error(
                    f"[STOPLOSS] Impossible de récupérer le prix courant pour {symbol}"
                )
                return False

            # Calcul de la perte latente
            loss = (price - entry) / entry
            if loss < -self.stop_loss_pct:
                self.logger.warning(
                    f"[STOPLOSS] Déclenché sur {symbol}: perte = {loss:.2%} (entrée {entry}, prix actuel {price})"
                )
                return True
            return False
        except Exception as e:
            self.logger.error(f"[STOPLOSS] Erreur vérification stop-loss: {e}")
            return False

    async def execute_trade(
        self, symbol, side, amount, price=None, iceberg=False, iceberg_visible_size=0.1
    ):
        """
        Exécute un ordre de trading avec logs détaillés.
        - BUY sur Binance spot (quoteOrderQty)
        - SELL sur Binance spot (revente, si déjà long)
        - SHORT sur BingX (futures)
        - Gère le suivi de position SPOT et le stop-loss automatique
        """
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
                    "entry_price": price or 0,
                    "amount": amount,
                }
            elif side.upper() == "SELL":
                if not self.is_long(symbol):
                    log_dashboard(
                        f"[ORDER] Pas en position long sur {symbol}, vente ignorée (simu)"
                    )
                    return {"status": "skipped", "reason": "not in position"}
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
                        "entry_price": result.get("avg_price", price),
                        "amount": result.get("filled_amount", amount),
                    }

            # ----- VENTE SPOT -----
            elif side.upper() == "SELL" and symbol.endswith("USDC"):
                if not self.is_long(symbol):
                    log_dashboard(
                        f"[ORDER] Pas en position long sur {symbol}, vente ignorée."
                    )
                    return {"status": "skipped", "reason": "not in position"}
                bid, ask = self.get_ws_orderbook(symbol)
                if bid is None or ask is None:
                    log_dashboard(
                        f"[ORDER] Orderbook WS non dispo pour {symbol}, annulation de l'ordre."
                    )
                    return {"status": "error", "reason": "Orderbook WS not available"}
                orderbook = {"bids": [[bid, 1.0]], "asks": [[ask, 1.0]]}
                pos = self.positions[symbol]
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
                    quoteOrderQty=pos["amount"],
                    orderbook=orderbook,
                    market_data=market_data,
                    iceberg=iceberg,
                    iceberg_visible_size=iceberg_visible_size,
                )
                if result.get("status") == "completed":
                    self.positions.pop(symbol, None)

            # ----- SHORT BINGX -----
            elif side.upper() == "SHORT":
                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                ticker = await self.bingx_client.fetch_ticker(symbol_bingx)
                price_bingx = float(ticker["last"])
                qty = amount / price_bingx
                result = await self.bingx_executor.short_order(
                    symbol_bingx, qty, leverage=3
                )

            else:
                return {"status": "rejected", "reason": "unsupported side"}

            # ----- LOGS & NOTIF -----
            if result["status"] == "completed":
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
                    f"💵 Total: ${float(result.get('filled_amount', amount)) * float(result.get('avg_price', price) or 0):.2f}"
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

    def _update_performance_metrics(self, trade_result):
        """Met à jour les métriques de performance après un trade réel"""
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)

            performance = data["bot_status"]["performance"]

            # Mise à jour des statistiques
            performance["total_trades"] += 1

            # Calcul du profit/perte
            filled_amount = float(trade_result["filled_amount"])
            avg_price = float(trade_result["avg_price"])
            side = trade_result["side"]  # <-- side est une string

            if side == "buy":
                # Pour un achat, on ne sait pas encore si c'est gagnant
                pass
            elif side == "sell":
                # Pour une vente, on peut calculer le profit par rapport au prix d'achat moyen
                entry_price = trade_result.get("entry_price", 0)
                if entry_price > 0:
                    profit_pct = (
                        (avg_price / entry_price - 1) * 100
                        if side == "sell"
                        else (1 - avg_price / entry_price) * 100
                    )
                    profit_amount = filled_amount * avg_price * profit_pct / 100

                    # Mise à jour de la balance
                    performance["balance"] += profit_amount

                    # Mise à jour du win_rate
                    if profit_amount > 0:
                        performance["wins"] = performance.get("wins", 0) + 1
                    else:
                        performance["losses"] = performance.get("losses", 0) + 1

                    performance["win_rate"] = (
                        performance.get("wins", 0) / performance["total_trades"]
                    )

                    # Mise à jour du profit factor
                    performance["total_profit"] = performance.get(
                        "total_profit", 0
                    ) + max(0, profit_amount)
                    performance["total_loss"] = performance.get("total_loss", 0) + max(
                        0, -profit_amount
                    )

                    if performance["total_loss"] > 0:
                        performance["profit_factor"] = (
                            performance["total_profit"] / performance["total_loss"]
                        )

            # Sauvegarde des données mises à jour
            data["bot_status"]["performance"] = performance
            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _prepare_features_for_ai(self, symbol):
        """
        Prépare les features pour les modèles d'IA (adapté pour PPO et DL).
        ATTENTION: Retourne TOUJOURS un dict avec les clés
        'close', 'high', 'low', 'volume', 'rsi', 'macd', 'volatility'.
        Si besoin pour PPO, ajoute aussi 'vol_ratio'.
        """
        try:
            N_STEPS = self.N_STEPS

            ohlcv = self.market_data.get(symbol, {}).get("1h", {})
            if not ohlcv or not isinstance(ohlcv, dict) or "close" not in ohlcv:
                return None

            closes = np.array(ohlcv.get("close", []))
            highs = np.array(ohlcv.get("high", []))
            lows = np.array(ohlcv.get("low", []))
            volumes = np.array(ohlcv.get("volume", []))

            # --- Vérification stricte sur la taille
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

            # RSI (14)
            delta = np.diff(closes)
            gain = (delta > 0) * delta
            loss = (delta < 0) * -delta
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0.001
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))

            # MACD: EMA12 - EMA26
            ema12 = np.mean(closes[-12:]) if len(closes) >= 12 else closes[-1]
            ema26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
            macd = ema12 - ema26

            # Volatility: std des returns
            if len(closes) >= N_STEPS:
                returns = np.diff(np.log(closes))
                volatility = float(np.std(returns[-14:])) if len(returns) >= 14 else 0
            else:
                volatility = 0

            avg_volume = np.mean(volumes) if np.mean(volumes) > 0 else 1
            vol_ratio = float(volumes[-1]) / avg_volume if avg_volume > 0 else 1
            vol_ratio = min(1, vol_ratio / 3)

            features = {
                "close": closes / closes[0],
                "high": highs / highs[0] if highs[0] > 0 else highs,
                "low": lows / lows[0] if lows[0] > 0 else lows,
                "volume": volumes / volumes[0] if volumes[0] > 0 else volumes,
                "rsi": float(rsi) / 100,
                "macd": float(macd) / 100,
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
            # 1. Vérification et initialisation des poids
            if not hasattr(self, "ai_weight"):
                self.ai_weight = 0.4  # Valeur par défaut

            # Conversion robuste du ai_weight si nécessaire
            try:
                ai_weight = float(self.ai_weight)
            except (TypeError, ValueError):
                self.logger.error(
                    "ai_weight invalide, utilisation de la valeur par défaut 0.4"
                )
                ai_weight = 0.4

            technical_weight = 1.0 - ai_weight

            # 2. Initialisation des structures
            if symbol not in self.market_data:
                self.market_data[symbol] = {}

            default_signals = {"trend": 0.0, "momentum": 0.0, "volatility": 0.0}

            current_signals = self.market_data[symbol].get(
                "signals", default_signals.copy()
            )

            # 3. Fonction de conversion universelle
            def safe_float(value, context=""):
                """Convertit n'importe quelle entrée en float de manière sécurisée"""
                if isinstance(value, (float, int)):
                    return float(value)

                if isinstance(value, dict):
                    # Extraction depuis les dictionnaires
                    for key in ["value", "action", "score", "prediction", "weight"]:
                        if key in value:
                            try:
                                return float(value[key])
                            except (TypeError, ValueError):
                                continue

                    # Fallback: premier float trouvé
                    for v in value.values():
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            continue

                # Fallback final
                self.logger.warning(
                    f"Conversion impossible pour {context}, utilisation de 0.0"
                )
                return 0.0

            # 4. Conversion des entrées
            dl_value = safe_float(dl_prediction, "dl_prediction")
            ppo_value = safe_float(ppo_action, "ppo_action")
            ai_signal = dl_value * 0.7 + ppo_value * 0.3

            # 5. Nettoyage des signaux existants
            clean_signals = {
                k: safe_float(v, f"signal {k}")
                for k, v in current_signals.items()
                if k in default_signals
            }

            # 6. Fusion finale
            merged_signals = {
                k: (v * technical_weight + ai_signal * ai_weight)
                for k, v in clean_signals.items()
            }

            # 7. Sauvegarde des résultats
            self.market_data[symbol]["signals"] = merged_signals
            self.market_data[symbol]["ai_prediction"] = ai_signal

            return merged_signals

        except Exception as e:
            self.logger.error(f"ERREUR dans _merge_signals: {str(e)}", exc_info=True)
            return default_signals.copy()

    async def _news_analysis_loop(self):
        log_dashboard("[NEWS] Lancement boucle d'analyse des news…")
        """Boucle d'analyse des news (version propre sans print/debug)"""
        while True:
            try:
                if not self.news_enabled or not self.news_analyzer:
                    await asyncio.sleep(self.news_update_interval)
                    continue

                self.logger.info("Fetching latest news for sentiment analysis")
                news_data = await self.news_analyzer.fetch_all_news()

                sentiment_analysis = {}
                try:
                    sentiment_analysis = await self.news_analyzer.update_analysis()
                except Exception:
                    self.logger.error("Erreur update_analysis", exc_info=True)
                    # sentiment_analysis reste {}

                # Extract the items list from the analysis result
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
                    await self.telegram.send_news_summary(news_data[:5])
                except Exception:
                    pass

                # === LOG SENTIMENT GLOBAL ===
                try:
                    with open(self.data_file, "r") as f:
                        shared_data = json.load(f)
                    sentiment_data = shared_data.get("sentiment", {})
                    avg_sentiment = sentiment_data.get("overall_sentiment", 0)
                    impact_score = sentiment_data.get("impact_score", 0)
                    major_events = sentiment_data.get("major_events", "")

                    log_dashboard(
                        f"[NEWS] Score sentiment global: {avg_sentiment:.2f} | Impact: {impact_score:.2f} | Événements: {major_events}"
                    )
                except Exception as e:
                    print(
                        f"[NEWS] Impossible d'afficher le score sentiment global: {e}"
                    )

            except Exception as e:
                self.logger.error(f"News analysis error: {e}")

            await asyncio.sleep(self.news_update_interval)

    async def _update_sentiment_data(self, sentiment_scores):
        """
        Met à jour les données de marché avec le sentiment :
        - Calcule la moyenne pondérée du sentiment par symbole sur toutes les news du cycle.
        - Applique le score global sinon.
        - Enregistre tout dans shared_data.json pour usage persistant.
        """
        from collections import defaultdict

        # 1. Agrégation pondérée des scores par symbole
        symbol_sentiments = defaultdict(list)
        for item in sentiment_scores:
            symbols = item.get("symbols", [])
            score = item.get("sentiment", 0)
            if not symbols:
                # PATCH: Appliquer le score global à toutes les paires
                for key in self.market_data:
                    self.market_data[key]["sentiment"] = score
                    self.market_data[key]["sentiment_timestamp"] = time.time()
                continue
            for symbol in symbols:
                symbol = symbol.upper()
                for key in self.market_data:
                    if symbol in key.upper():
                        self.market_data[key]["sentiment"] = score
                        self.market_data[key]["sentiment_timestamp"] = time.time()
                        print(
                            f"[DEBUG SENTIMENT FUZZY ASSIGN] {key} <- {score} via symbol={symbol}"
                        )

        # 2. Applique la moyenne pondérée à chaque paire
        for key in self.market_data:
            # Extrait le ticker principal, ex: "BTCUSDT" -> "BTC", "ETHUSDT" -> "ETH"
            ticker = key.replace("USDT", "").replace("USD", "")
            values = symbol_sentiments.get(ticker, [])
            if values:
                total = sum(s * i for s, i in values)
                total_weight = sum(i for _, i in values)
                avg = total / total_weight if total_weight else 0
                self.market_data[key]["sentiment"] = avg
                self.market_data[key]["sentiment_timestamp"] = time.time()
                print(
                    f"[DEBUG AGG SENTIMENT] {key} <- {avg:.4f} via {len(values)} news (pondérée)"
                )

        # 3. Récupère la valeur globale du sentiment depuis le fichier partagé
        try:
            with open(self.data_file, "r") as f:
                shared_data = json.load(f)
            news_sentiment = shared_data.get("sentiment", None)
            if news_sentiment:
                global_sentiment = news_sentiment.get("overall_sentiment", 0)
            else:
                global_sentiment = 0
        except Exception as e:
            print(f"[DEBUG ERROR] Could not read global sentiment from file: {e}")
            global_sentiment = 0

        print(f"[DEBUG SENTIMENT GLOBAL FINAL] avg_sentiment={global_sentiment}")

        # 4. Applique le score global si aucune news spécifique
        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            if pair_key not in self.market_data:
                self.market_data[pair_key] = {}

            if (
                "sentiment" not in self.market_data[pair_key]
                or self.market_data[pair_key]["sentiment"] == 0
            ):
                self.market_data[pair_key]["sentiment"] = global_sentiment
                self.market_data[pair_key]["sentiment_timestamp"] = time.time()
                print(
                    f"[DEBUG PROPAG GLOBAL SENTIMENT] {pair_key} <- {global_sentiment}"
                )

        # 5. Sauvegarde tous les sentiments dans shared_data.json
        symbol_sentiments_out = {
            key: data.get("sentiment", 0) for key, data in self.market_data.items()
        }
        try:
            with open(self.data_file, "r") as f:
                shared_data = json.load(f)
        except Exception:
            shared_data = {}
        shared_data["last_sentiment_update"] = time.time()
        shared_data["sentiment_by_symbol"] = symbol_sentiments_out
        try:
            with open(self.data_file, "w") as f:
                json.dump(shared_data, f, indent=2)
            print("[SENTIMENT SAVE] shared_data.json mis à jour avec les sentiments")
        except Exception as e:
            print(f"[SENTIMENT SAVE ERROR] {e}")

    async def _save_sentiment_data(self, sentiment_scores, news_data=None):
        """
        Enregistre les données de sentiment du marché (scores, news, global) dans le fichier partagé.
        Correction : le score global est calculé sur les scores déjà assignés à chaque paire,
        et sinon fallback sur sentiment_scores si jamais.
        """
        headlines = []
        if news_data is None:
            news_data = sentiment_scores
        if isinstance(news_data, list):
            for item in news_data[:10]:
                if isinstance(item, dict) and "title" in item:
                    headlines.append(
                        str(item["title"])
                    )  # Toujours str pour éviter erreur

        # Correction : on prend les scores assignés dans market_data
        valid_scores = [
            data.get("sentiment")
            for key, data in self.market_data.items()
            if data.get("sentiment") is not None
        ]
        print(
            f"[DEBUG _save_sentiment_data] valid_scores from market_data={valid_scores}"
        )

        # Fallback sur sentiment_scores si jamais
        if not valid_scores:
            valid_scores = [
                item.get("sentiment")
                for item in sentiment_scores
                if isinstance(item, dict) and item.get("sentiment") is not None
            ]
            print(
                f"[DEBUG _save_sentiment_data] fallback valid_scores from sentiment_scores={valid_scores}"
            )

        # === PATCH : Utilise le nouveau résumé pour remplir le sentiment_data ===
        summary = get_sentiment_summary_from_batch(sentiment_scores)
        sentiment_global = summary["sentiment_global"]
        impact_score = float(
            np.mean(
                [
                    abs(item.get("sentiment", 0))
                    for item in sentiment_scores
                    if isinstance(item, dict)
                ]
            )
            if sentiment_scores
            else 0.0
        )
        major_events = (
            "; ".join(summary["top_news"][:3]) if summary["top_news"] else "Aucun"
        )

        print(
            f"[DEBUG SENTIMENT GLOBAL] sentiment_global={sentiment_global} impact={impact_score} major_events={major_events}"
        )

        sentiment_data = {
            "timestamp": datetime.now().isoformat(),
            "scores": sentiment_scores,
            "latest_news": summary["top_news"],
            "overall_sentiment": sentiment_global,
            "impact_score": impact_score,
            "major_events": major_events,
            "top_symbols": summary["top_symbols"],
            "n_news": summary["n_news"],
        }

        try:
            with open(self.data_file, "r") as f:
                shared_data = json.load(f)
            shared_data["sentiment"] = sentiment_data
            with open(self.data_file, "w") as f:
                json.dump(shared_data, f, indent=4)
            self.logger.info(
                f"[SENTIMENT] Data written successfully to {self.data_file}"
            )
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")

    async def generate_market_analysis_report(self, cycle=None):
        debug_market_data_structure(
            self.market_data, self.pairs_valid, ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
        report = (
            f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {get_current_time()}\n"
            f"Cycle: {cycle if cycle is not None else self.current_cycle}\n"
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
                    ai_score = self.market_data[pair_key]["ai_prediction"]
                    ai_signal = (
                        "ACHAT"
                        if ai_score > 0.6
                        else "VENTE" if ai_score < 0.4 else "NEUTRE"
                    )
                    report += f"""
        🤖 {pair} :
        └─ Prédiction: {ai_signal} ({ai_score:.2f})
        """

        # --- AJOUT : Section news/sentiment globale détaillée ---
        try:
            with open(self.data_file, "r") as f:
                shared_data = json.load(f)
            news_sentiment = shared_data.get("sentiment", None)
        except Exception:
            news_sentiment = None

        if news_sentiment and isinstance(news_sentiment, dict):
            try:
                sentiment = float(news_sentiment.get("overall_sentiment", 0) or 0)
            except Exception:
                sentiment = 0.0
            try:
                impact = float(news_sentiment.get("impact_score", 0) or 0)
            except Exception:
                impact = 0.0
            major_events = news_sentiment.get("major_events", "Aucun")
            report += (
                "\n📰 Analyse des News:\n"
                f"Sentiment: {sentiment:.2%}\n"
                f"Impact estimé: {impact:.2%}\n"
                f"Événements majeurs: {major_events}\n"
            )
            # Ajout des dernières news si dispo
            major_news = news_sentiment.get("latest_news", [])
            if major_news:
                report += "Dernières news :\n"
                for news in major_news[:3]:
                    report += f"- {news}\n"
        else:
            report += "\n📰 Analyse des News: Aucune donnée disponible.\n"

        # Ajout des informations de sentiment par paire si disponibles
        if self.news_enabled:
            report += "\n    📰 Analyse de Sentiment :\n"
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if (
                    pair_key in self.market_data
                    and "sentiment" in self.market_data[pair_key]
                ):
                    sentiment_score = self.market_data[pair_key]["sentiment"]
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
        """
        Ajoute les prédictions des modèles d'IA aux données de marché.
        Corrige dynamiquement le shape de ppo_features selon le nombre de paires.
        """
        # PATCH: Définit les constantes locales nécessaires !
        N_STEPS = self.N_STEPS
        N_FEATURES = self.N_FEATURES

        if not self.ai_enabled or not self.dl_model or not self.ppo_strategy:
            return

        expected_shape = (self.get_input_dim(),)
        num_pairs = len(self.pairs_valid)

        ppo_features_list = []
        dl_predictions = {}

        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            features = await self._prepare_features_for_ai(pair_key)
            if features is not None:
                try:
                    # Prédiction du CNN-LSTM
                    dl_prediction = self.dl_model.predict(features)
                    dl_predictions[pair_key] = dl_prediction

                    # Correction NaN/inf
                    for k in features:
                        arr = features[k]
                        if isinstance(arr, np.ndarray):
                            if np.isnan(arr).any() or np.isinf(arr).any():
                                print(
                                    f"[WARN] NaN/inf détecté dans {k}, correction appliquée"
                                )
                                features[k] = np.nan_to_num(arr)
                        else:
                            if np.isnan(features[k]) or np.isinf(features[k]):
                                print(
                                    f"[WARN] NaN/inf détecté dans {k}, correction appliquée"
                                )
                                features[k] = float(np.nan_to_num(features[k]))

                    # Construction du vecteur feature
                    vec = np.concatenate(
                        [
                            (
                                features[k]
                                if isinstance(features[k], np.ndarray)
                                else np.full(N_STEPS, features[k])
                            )
                            for k in [
                                "close",
                                "high",
                                "low",
                                "volume",
                                "rsi",
                                "macd",
                                "volatility",
                                "vol_ratio",
                            ]
                        ]
                    )
                    if vec.shape != (N_FEATURES * N_STEPS,):
                        print(
                            f"[SKIP PPO] {pair_key}, shape {vec.shape}, pas assez de data"
                        )
                        continue
                    ppo_features_list.append(vec)
                except Exception as e:
                    self.logger.error(f"Error preparing AI features for {pair}: {e}")

        if not ppo_features_list:
            print("[SKIP PPO] Aucun vecteur de features disponible pour PPO.")
            return
        ppo_features = np.concatenate(ppo_features_list)
        expected_shape = (N_FEATURES * N_STEPS * num_pairs,)
        print(
            f"[DEBUG] Shape du vecteur features PPO : {ppo_features.shape}, attendu : {expected_shape}"
        )
        if ppo_features.shape != expected_shape:
            print(f"[SKIP PPO] Shape {ppo_features.shape}, attendu: {expected_shape}")
            return

        print("PPO features shape:", ppo_features.shape)

        try:
            ppo_action = self.ppo_strategy.get_action(ppo_features)
            for i, pair in enumerate(self.pairs_valid):
                pair_key = pair.replace("/", "").upper()
                dl_pred = dl_predictions.get(pair_key, 0)
                await self._merge_signals(pair_key, dl_pred, ppo_action)
        except Exception as e:
            self.logger.error(f"Error getting PPO action: {e}")

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
        """Initialise le fichier de données partagées"""
        data = {
            "timestamp": get_current_time(),
            "user": CURRENT_USER,
            "bot_status": {
                "regime": self.regime,
                "cycle": self.current_cycle,
                "last_update": get_current_time(),
                "performance": {
                    "total_trades": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "balance": 10000,
                    "wins": 0,
                    "losses": 0,
                    "total_profit": 0,
                    "total_loss": 0,
                },
            },
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=4)

    def save_shared_data(self):
        """Met à jour les données partagées sans effacer la clé 'sentiment'"""
        try:
            # Charger les données existantes pour préserver 'sentiment'
            if os.path.exists(self.data_file):
                with open(self.data_file, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            # MAJ des sections
            data.update(
                {
                    "timestamp": get_current_time(),
                    "user": CURRENT_USER,
                    "bot_status": {
                        "regime": self.regime,
                        "cycle": self.current_cycle,
                        "last_update": get_current_time(),
                        "performance": self.get_performance_metrics(),
                    },
                    "market_data": self.market_data,
                    "indicators": self.indicators,
                }
            )

            # Ajoute les prédictions IA si besoin
            if self.ai_enabled:
                ai_predictions = {}
                for pair in self.pairs_valid:
                    pair_key = pair.replace("/", "").upper()
                    if (
                        pair_key in self.market_data
                        and "ai_prediction" in self.market_data[pair_key]
                    ):
                        ai_predictions[pair] = self.market_data[pair_key][
                            "ai_prediction"
                        ]
                data["ai_predictions"] = ai_predictions

            # NE PAS EFFACER 'sentiment' si déjà présent
            # (on ne touche pas à data["sentiment"])

            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving shared data: {e}")

    def get_performance_metrics(self):
        """Récupère les métriques de performance actuelles"""
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)

            # AJOUT ICI : récupération du solde réel Binance
            real_balance = self.get_binance_real_balance("USDC")
            if real_balance is not None:
                data["bot_status"]["performance"]["balance"] = real_balance

            return data["bot_status"]["performance"]
        except:
            # Retourne des métriques simulées si le fichier n'existe pas
            return {
                "total_trades": self.current_cycle * 2,
                "win_rate": 0.62 + (self.current_cycle * 0.001),
                "profit_factor": 1.85 + (self.current_cycle * 0.01),
                "balance": 10000 + (self.current_cycle * 100),
                "wins": int(self.current_cycle * 1.2),
                "losses": self.current_cycle - int(self.current_cycle * 1.2),
                "total_profit": self.current_cycle * 150,
                "total_loss": self.current_cycle * 50,
            }

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

    def add_indicators(self, df):
        """
        Calcule tous les indicateurs nécessaires pour les stratégies du dossier 'strategies'.
        Retourne un dictionnaire {nom_indicateur: dernière_valeur non-NaN ou None}
        (Version enrichie avec indicateurs avancés)
        """
        try:
            # Gestion entrée : DataFrame, liste de dicts, liste de listes
            if isinstance(df, list):
                if len(df) == 0:
                    self.logger.error("add_indicators: Liste reçue vide")
                    return None
                if isinstance(df[0], dict):
                    df = pd.DataFrame(df)
                elif isinstance(df[0], (list, tuple)):
                    columns = ["timestamp", "open", "high", "low", "close", "volume"]
                    df = pd.DataFrame(df, columns=columns)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    self.logger.error(
                        "add_indicators: Format de liste non pris en charge"
                    )
                    return None
            if not isinstance(df, pd.DataFrame):
                self.logger.error("add_indicators: df n'est pas un DataFrame")
                return None

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

            # Tri et conversion du timestamp pour tous les indicateurs ET VWAP
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.drop_duplicates(subset="timestamp", keep="last")
                df = df.sort_values("timestamp")
                df = df.reset_index(drop=True)

            if df.empty:
                self.logger.warning(
                    "DataFrame vide, impossible de calculer les indicateurs"
                )
                print("[DEBUG add_indicators] DataFrame vide après tri/formatage")
                return None

            try:
                df_ta = df.copy()

                # Calcul des indicateurs classiques
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

                # Indicateurs avancés supplémentaires
                try:
                    # PATCH : on reforce ici le tri strict et datetime juste avant VWMA/VWAP
                    if "timestamp" in df_ta.columns:
                        df_ta["timestamp"] = pd.to_datetime(df_ta["timestamp"])
                        df_ta = df_ta.drop_duplicates(subset="timestamp", keep="last")
                        df_ta = df_ta.sort_values("timestamp")
                        df_ta = df_ta.reset_index(drop=True)
                    vwma = df_ta.ta.vwma(length=20)
                    df_ta["vwma_20"] = vwma
                except Exception:
                    df_ta["vwma_20"] = np.nan
                try:
                    obv = df_ta.ta.obv()
                    df_ta["obv"] = obv
                except Exception:
                    df_ta["obv"] = np.nan
                try:
                    # PATCH : reforce tri juste avant VWAP !
                    if "timestamp" in df_ta.columns:
                        df_ta["timestamp"] = pd.to_datetime(df_ta["timestamp"])
                        df_ta = df_ta.drop_duplicates(subset="timestamp", keep="last")
                        df_ta = df_ta.sort_values("timestamp")
                        df_ta = df_ta.reset_index(drop=True)
                    vwap = df_ta.ta.vwap()
                    df_ta["vwap"] = vwap
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

            except Exception as e:
                self.logger.warning(f"Erreur pandas-ta indicateurs principaux : {e}")
                indicators = {}

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
            return None

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

            # 3. Préchargement historique (optionnel, sécurisé)
            if hasattr(bot, "ws_collector") and hasattr(bot, "binance_client"):
                for symbol in bot.config["TRADING"]["pairs"]:
                    symbol_binance = symbol.replace("/", "").upper()
                    for tf in bot.config["TRADING"]["timeframes"]:
                        try:
                            bot.ws_collector.preload_historical(
                                bot.binance_client, symbol_binance, tf, limit=2000
                            )
                            print(f"Préchargement {symbol_binance} {tf} OK")
                        except Exception as e:
                            print(f"Erreur préchargement {symbol_binance} {tf} : {e}")

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
        try:
            pair_key = pair.replace("/", "").upper()
            if not market_data or pair_key not in market_data:
                return None

            ohlcv_df = bot.ws_collector.get_dataframe(pair_key, tf)
            if ohlcv_df is None or len(ohlcv_df) < 20:
                return None

            indicators_data = bot.add_indicators(ohlcv_df)

            # === PATCH AUTO-STRATEGIE ===
            if hasattr(bot, "auto_strategy_config") and bot.auto_strategy_config:
                auto_cfg = bot.auto_strategy_config
                if (
                    pair_key.upper() == auto_cfg["pair"].upper()
                    and tf == auto_cfg["timeframe"]
                ):
                    action = appliquer_config_strategy(ohlcv_df, auto_cfg["config"])
                    signal = {"action": action, "confidence": 1.0}
                else:
                    # Appel standard
                    signal = await bot.analyze_signals(
                        pair_key, ohlcv_df, indicators_data
                    )
                    signal["pair"] = pair
                    signal["tf"] = tf
                    return signal
            else:
                # Appel standard
                signal = await bot.analyze_signals(pair_key, ohlcv_df, indicators_data)
                signal["pair"] = pair
                signal["tf"] = tf
                return signal
            # === FIN PATCH AUTO-STRATEGIE ===

            return signal

        except Exception as e:
            logger.error(f"Erreur analyse {pair}: {e}")
            return None

    async def execute_trading_cycle(bot, valid_pairs):
        """Exécute un cycle complet de trading"""
        try:
            # 0. Import avancé des indicateurs orderflow (à placer en haut du fichier !)
            try:
                from src.analysis.technical.advanced.advanced_indicators import (
                    AdvancedIndicators,
                )

                orderflow_indicators = AdvancedIndicators()
            except Exception as e:
                orderflow_indicators = None
                print("[Orderflow] Impossible d'importer AdvancedIndicators:", e)

            # 1. Injection des données live WS dans market_data (remplace le fetch market_data historique !)
            for pair in bot.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if pair_key not in bot.market_data:
                    bot.market_data[pair_key] = {}
                for tf in bot.config["TRADING"]["timeframes"]:
                    df = bot.ws_collector.get_dataframe(pair_key, tf)

                    if df is not None and not df.empty:
                        bot.market_data[pair_key][tf] = {
                            "open": df["open"].tolist(),
                            "high": df["high"].tolist(),
                            "low": df["low"].tolist(),
                            "close": df["close"].tolist(),
                            "volume": df["volume"].tolist(),
                            "timestamp": [
                                int(pd.Timestamp(t).timestamp())
                                for t in df["timestamp"]
                            ],
                        }
                        # 1bis. Calcul et injection des indicateurs orderflow avancés
                        if orderflow_indicators is not None:
                            try:
                                bid_ask = None
                                liquidity_wave = None
                                smart_money = None
                                if hasattr(orderflow_indicators, "_bid_ask_ratio"):
                                    bid_ask = orderflow_indicators._bid_ask_ratio(df)
                                if hasattr(orderflow_indicators, "_liquidity_wave"):
                                    liquidity_wave = (
                                        orderflow_indicators._liquidity_wave(df)
                                    )
                                if hasattr(orderflow_indicators, "_smart_money_index"):
                                    smart_money = (
                                        orderflow_indicators._smart_money_index(df)
                                    )
                                bot.market_data[pair_key][tf]["orderflow"] = {
                                    "bid_ask_ratio": bid_ask,
                                    "liquidity_wave": liquidity_wave,
                                    "smart_money_index": smart_money,
                                }
                            except Exception as e:
                                print(f"[Orderflow] Erreur calcul {pair_key} {tf}: {e}")
                    else:
                        # Ajout d'un log utile pour debug volume/DF vide
                        print(f"[DEBUG] DataFrame vide pour {pair_key} {tf}")

            # Log debug sur la structure finale (optionnel)
            for sym in bot.market_data:
                print(f"[DEBUG] {sym}: {list(bot.market_data[sym].keys())}")

            # 2. Analyse de marché
            regime, market_data, indicators = await bot.study_market("7d")
            strategy = bot.choose_strategy(regime, indicators)
            log_dashboard(f"🎯 Stratégie active: {strategy}")

            # 3. Détection d'arbitrage
            await handle_arbitrage_opportunities(bot)

            # 4. Analyse des paires pour CHAQUE timeframe
            trade_decisions = []
            for pair in valid_pairs:
                for tf in bot.config["TRADING"]["timeframes"]:
                    decision = await market_analysis_cycle(
                        bot, pair, bot.market_data, tf=tf
                    )
                    if decision:
                        trade_decisions.append(decision)

            # === AJOUT : LOG DES DÉCISIONS FINALES ===
            for decision in trade_decisions:
                signals = decision.get("signals", {})
                log_dashboard(
                    f"[TRADE DECISION] {decision['pair']} | "
                    f"Action: {decision['action'].upper()} | Confiance: {decision['confidence']:.2f} | "
                    f"Tech: {signals.get('technical', 0):.2f} | "
                    f"IA: {signals.get('ai', 0):.2f} | "
                    f"Sentiment: {signals.get('sentiment', 0):.2f}"
                )
            # === FIN AJOUT ===

            # 5. Exécution des trades
            await execute_trade_decisions(bot, trade_decisions)

            return trade_decisions, regime

        except Exception as e:
            logger.error(f"Erreur cycle trading: {e}")
            raise

    # Fonction principale
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
                cycle += 1
                start = datetime.utcnow()
                try:
                    print(f"\n🔄 Cycle {cycle} - {start.strftime('%H:%M:%S')}")
                    # Hot reload IA
                    bot.check_reload_dl_model()

                    # === PATCH : Déclenchement automatique du stop-loss SPOT ===
                    for symbol, pos in list(bot.positions.items()):
                        if bot.is_long(symbol) and bot.check_stop_loss(symbol):
                            print(
                                f"[STOPLOSS] Déclenchement automatique du stop-loss pour {symbol}"
                            )
                            await bot.execute_trade(symbol, "SELL", pos["amount"])

                    # === PATCH : Déclenchement du stop-loss ET trailing stop SHORT BingX ===
                    for symbol, pos in list(bot.positions.items()):
                        if bot.is_short(symbol):
                            try:
                                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                                ticker = await bot.bingx_client.fetch_ticker(
                                    symbol_bingx
                                )
                                price = float(ticker["last"])
                            except Exception:
                                continue
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
                                await bot.execute_trade(symbol, "BUY", pos["amount"])

                    # Exécution du cycle de trading
                    trade_decisions, regime = await execute_trading_cycle(
                        bot, valid_pairs
                    )

                    # Mise à jour des données du bot
                    bot.current_cycle = cycle
                    bot.regime = regime

                    # Calcul et stockage des indicateurs pour chaque paire/timeframe
                    bot.indicators = {}
                    for pair in bot.pairs_valid:
                        pair_key = pair.replace("/", "").upper()
                        for tf in bot.config["TRADING"]["timeframes"]:
                            if (
                                pair_key in bot.market_data
                                and tf in bot.market_data[pair_key]
                            ):
                                trend = bot.calculate_trend(
                                    bot.market_data[pair_key][tf]
                                )
                                volatility = bot.calculate_volatility(
                                    bot.market_data[pair_key][tf]
                                )
                                volume_profile = bot.calculate_volume_profile(
                                    bot.market_data[pair_key][tf]
                                )
                                dominant_signal = bot.get_dominant_signal(pair, tf)
                                # Ajoute les indicateurs techniques bruts
                                df = bot.ws_collector.get_dataframe(pair_key, tf)
                                indics = (
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
                                    "ta": indics if indics else {},
                                }

                    # Sauvegarde de l'état du bot à chaque cycle
                    bot.save_shared_data()

                    # === Entraînement automatique IA toutes les 50 itérations ===
                    if cycle % 50 == 0:
                        print(
                            "=== Entraînement automatique IA sur toutes les paires/timeframes ==="
                        )
                        bot.train_cnn_lstm_on_all_live()

                    # === Fin entraînement IA auto ===
                    bot.train_cnn_lstm_on_all_live()
                    print(
                        "=== Entraînement MANUEL IA sur toutes les paires/timeframes ==="
                    )
                    # Calcul de la durée du cycle et affichage
                    duration = (datetime.utcnow() - start).total_seconds()
                    print(f"✅ Cycle terminé en {duration:.1f}s")

                    # Envoi des mises à jour et rapports Telegram
                    await send_cycle_reports(
                        bot, trade_decisions, cycle, regime, duration
                    )

                except Exception as e:
                    error_msg = f"⚠️ Erreur cycle {cycle}: {e}"
                    logger.error(error_msg)
                    await bot.telegram.send_message(error_msg)

                # Attente avant le prochain cycle
                await asyncio.sleep(30)

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
        log_dashboard(f"💹 {len(opportunities)} opportunités d'arbitrage détectées")
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
    """
    for decision in trade_decisions:
        pair = decision.get("pair")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        amount = calculate_position_size(
            bot, decision
        )  # Utilise la fonction déjà présente
        # Log avant exécution
        log_dashboard(
            f"[EXECUTE TRADE] {pair} | Action: {action.upper()} | Amount: {amount} | Confidence: {confidence}"
        )
        # Exécution réelle
        trade_result = await bot.execute_trade(pair, action, amount)
        # Notification Telegram
        await send_trade_notification(bot, decision, trade_result, amount)


async def execute_trade(
    self, symbol, side, amount, price=None, iceberg=False, iceberg_visible_size=0.1
):
    """
    Exécute un ordre de trading avec logs détaillés.
    - BUY sur Binance spot (quoteOrderQty)
    - SELL sur Binance spot (revente, si déjà long)
    - SHORT sur BingX (futures)
    - BUY sur BingX pour rachat short
    - Gère le suivi de position SPOT et le stop-loss automatique
    """
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
                log_dashboard(f"[ORDER] Déjà long sur {symbol}, achat ignoré (simu)")
                return {"status": "skipped", "reason": "already long"}
            self.positions[symbol] = {
                "side": "long",
                "entry_price": price or 0,
                "amount": amount,
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
                log_dashboard(f"[ORDER] Déjà short sur {symbol}, short ignoré (simu)")
                return {"status": "skipped", "reason": "already short"}
            self.positions[symbol] = {
                "side": "short",
                "entry_price": price or 0,
                "amount": amount,
                "min_price": price or 0,
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
                    "entry_price": result.get("avg_price", price),
                    "amount": result.get("filled_amount", amount),
                }

        # ----- VENTE SPOT -----
        elif side.upper() == "SELL" and symbol.endswith("USDC"):
            if not self.is_long(symbol):
                log_dashboard(
                    f"[ORDER] Pas en position long sur {symbol}, vente ignorée."
                )
                return {"status": "skipped", "reason": "not in position"}
            bid, ask = self.get_ws_orderbook(symbol)
            if bid is None or ask is None:
                log_dashboard(
                    f"[ORDER] Orderbook WS non dispo pour {symbol}, annulation de l'ordre."
                )
                return {"status": "error", "reason": "Orderbook WS not available"}
            orderbook = {"bids": [[bid, 1.0]], "asks": [[ask, 1.0]]}
            pos = self.positions[symbol]
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
                quoteOrderQty=pos["amount"],
                orderbook=orderbook,
                market_data=market_data,
                iceberg=iceberg,
                iceberg_visible_size=iceberg_visible_size,
            )
            if result.get("status") == "completed":
                self.positions.pop(symbol, None)

        # ----- OUVERTURE SHORT BINGX -----
        elif side.upper() == "SHORT":
            if self.is_short(symbol):
                log_dashboard(f"[ORDER] Déjà short sur {symbol}, short ignoré.")
                return {"status": "skipped", "reason": "already short"}
            symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
            ticker = await self.bingx_client.fetch_ticker(symbol_bingx)
            price_bingx = float(ticker["last"])
            qty = amount / price_bingx
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
            qty = pos["amount"]
            # Il faut avoir une méthode close_short_order côté BingXOrderExecutor, sinon utiliser un BUY ordinaire sur futures
            result = await self.bingx_executor.close_short_order(symbol_bingx, qty)
            if result.get("status") == "completed":
                self.positions.pop(symbol, None)

        else:
            return {"status": "rejected", "reason": "unsupported side"}

        # ----- LOGS & NOTIF -----
        if result["status"] == "completed":
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
                f"💵 Total: ${float(result.get('filled_amount', amount)) * float(result.get('avg_price', price) or 0):.2f}"
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
    """Calcule le montant en USDC à investir (et non la quantité de BTC)"""
    try:
        # Par exemple, on investit de 10 à 100 USDC selon la confiance et la volatilité
        base_usdc = 5  # minimum à investir (doit être > minNotional Binance, ici 5)
        max_usdc = 10  # maximum à investir

        volatility_factor = decision.get("signals", {}).get("volatility", 0.5)
        confidence = decision.get("confidence", 0.5)

        # Plus la volatilité est faible et la confiance élevée, plus on investit
        invest_usdc = base_usdc + (max_usdc - base_usdc) * confidence * (
            1 - volatility_factor
        )

        # Securité : arrondi à 2 décimales et respect du minimum
        invest_usdc = max(base_usdc, round(invest_usdc, 2))

        return invest_usdc  # <<< Montant USDC à investir

    except Exception as e:
        logging.error(f"Erreur calcul montant USDC: {e}")
        return 10  # fallback


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


def build_telegram_summary(bot, trade_decisions, news_sentiment):
    summary = "🟢 <b>Résumé du cycle</b>\n"
    # Régime
    summary += f"📊 Régime de marché : {bot.regime}\n"
    # Paires principales (top 5)
    top_pairs = (
        ", ".join([d["pair"] for d in trade_decisions[:5]])
        if trade_decisions
        else "N/A"
    )
    summary += f"📈 Paires principales : {top_pairs}\n"
    # Décisions de trade principales (top 5)
    for d in trade_decisions[:5]:
        emoji = (
            "🟢" if d["action"] == "buy" else "🔴" if d["action"] == "sell" else "⚪️"
        )
        conf = int(d["confidence"] * 100)
        summary += f"{emoji} {d['pair']} : {d['action'].upper()} ({conf}%)\n"
    # News principales (top 3)
    if news_sentiment and "latest_news" in news_sentiment:
        summary += "\n📰 News principales :\n"
        for title in news_sentiment["latest_news"][:3]:
            summary += f"• {title}\n"
    return summary


async def send_cycle_reports(bot, trade_decisions, cycle, regime, duration):
    """Envoie les rapports de fin de cycle"""
    try:
        # 1. Rapport des trades si nécessaire
        if trade_decisions:
            trade_report = "💹 <b>Résumé des trades du cycle</b>\n\n"
            for trade in trade_decisions:
                emoji = (
                    "🟢"
                    if trade["action"] == "buy"
                    else "🔴" if trade["action"] == "sell" else "⚪️"
                )
                pair = trade.get("pair", "INCONNU")
                conf = f"{trade.get('confidence', 0):.0%}"
                tech = f"{trade.get('signals', {}).get('technical', 0):.0%}"
                ia = f"{trade.get('signals', {}).get('ai', 0):.2f}"
                sent = f"{trade.get('signals', {}).get('sentiment', 0):.2f}"
                trade_report += f"{emoji} {pair}: {trade['action'].upper()} ({conf}) | Tech {tech} | IA {ia} | Sent {sent}\n"
            await bot.telegram.send_message(trade_report)
        else:
            await bot.telegram.send_cycle_update(cycle, regime, duration)

        # 2. ====== RAPPORT ANALYSE COMPLET (news + décisions + multi-TF/paire) ======

        # Construction d'un dict { "1m | BTC/USDT": {...}, ... }
        indicators_analysis = {}
        for pair in bot.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            for tf in bot.config["TRADING"]["timeframes"]:
                tf_key = f"{tf} | {pair}"
                indics = bot.indicators.get(pair_key, {}).get(tf, {})
                indicators_analysis[tf_key] = indics if indics else {}

        # Charger le sentiment/news si dispo
        news_sentiment = None
        try:
            with open(bot.data_file, "r") as f:
                shared_data = json.load(f)
            news_sentiment = shared_data.get("sentiment", None)
        except Exception:
            news_sentiment = None

        # Générer les décisions de trade par TF/paire (clé = f"{tf} | {pair}")
        trade_decisions_dict = {}
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

        # PATCH : Sauvegarde des décisions dans shared_data.json pour dashboard Streamlit
        try:
            with open(bot.data_file, "r") as f:
                shared_data = json.load(f)
            shared_data["trade_decisions"] = trade_decisions_dict
            with open(bot.data_file, "w") as f:
                json.dump(shared_data, f, indent=4)
        except Exception as e:
            logging.error(f"Erreur sauvegarde trade_decisions : {e}")

        regime_name = bot.regime if hasattr(bot, "regime") else "Indéterminé"

        rapport = _generate_analysis_report(
            indicators_analysis,
            regime_name,
            news_sentiment=news_sentiment,
            trade_decisions=trade_decisions_dict,
        )

        # Envoi sur Telegram
        await bot.telegram.send_message(
            build_telegram_summary(bot, trade_decisions, news_sentiment)
        )

    except Exception as e:
        logging.error(f"Erreur envoi rapports: {e}")


async def handle_shutdown(bot, message):
    """Gère l'arrêt propre du bot"""
    try:
        print(f"\n{message}")
        await bot.telegram.send_message(message)
        await bot.ws_collector.stop()
        bot.save_shared_data()
    except Exception as e:
        logging.error(f"Erreur arrêt bot: {e}")


def objective(trial):
    try:
        # Hyperparamètres
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # Initialisation et entraînement
        model = HybridAI()
        acc = model.train_and_validate(lr, batch_size)

        return acc

    except Exception as e:
        print(f"Erreur critique: {str(e)}")
        return 0.0


if __name__ == "__main__":

    # --- 1. Argument parsing avancé
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--capital", type=float, default=10000, help="Capital initial")
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
    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args()

    # Ajoute ici d'autres paramètres si besoin...
    args, unknown = parser.parse_known_args()

    # --- 2. Mode AutoML/Tuning (prioritaire sur tout le reste)
    if "automl" in sys.argv or "tune" in sys.argv:
        asyncio.run(run_automl_tuning(None, mode="cnn_lstm"))

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

    # --- 3. Mode backtest CLI
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
    elif "train-cnn-lstm" in sys.argv:
        bot = TradingBotM4()
        # Modifie ici la paire/timeframe si besoin
        bot.train_cnn_lstm_on_all_live()
        sys.exit(0)

    else:
        asyncio.run(run_clean_bot())
