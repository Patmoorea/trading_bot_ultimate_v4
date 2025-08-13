# 1. Import et configuration Streamlit (DOIT ÊTRE EN PREMIER)
import streamlit as st
# --- Ajout: Hack JavaScript pour autorefresh sans st_autorefresh ---
def auto_refresh(interval_ms=2000, key="js_autorefresh"):
    """Inject JS code for auto-refresh in Streamlit."""
    js_code = f"""
    <script>
        if (!window.{key}) {{
            window.{key} = setInterval(function() {{
                window.location.reload();
            }}, {interval_ms});
        }}
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)
# Initialisation des flags de protection
for flag, default in [
    ("prevent_cleanup", True),
    ("keep_alive", True),
    ("force_cleanup", False),
    ("cleanup_allowed", False),
]:
    if flag not in st.session_state:
        st.session_state[flag] = default
st.set_page_config(
    page_title="Trading Bot Ultimate v4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
# 2. Imports système
import os
import sys
import logging
import json
import re
import time
import signal
from datetime import timedelta
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import AsyncExitStack
from asyncio import TimeoutError, AbstractEventLoop
import asyncio
import nest_asyncio
import aiohttp
# 3. Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# 4. Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
# 5. Imports des bibliothèques externes
import plotly.graph_objects as go
import numpy as np
import pandas as pd
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"
import torch
import telegram
import ccxt
import ta
from dotenv import load_dotenv
import gymnasium as gym
from gymnasium import spaces
from binance import AsyncClient, BinanceSocketManager
# 6. Imports des modules internes (exchanges, core, etc.)
from src.exchanges.binance_exchange import BinanceExchange
from src.exchanges.binance.binance_client import BinanceClient
from src.core.exchange import ExchangeInterface as Exchange
from src.core.buffer.circular_buffer import CircularBuffer
from src.connectors.binance import BinanceConnector
from src.portfolio.real_portfolio import RealPortfolio
from src.regime_detection.hmm_kmeans import MarketRegimeDetector
from src.monitoring.streamlit_ui import TradingDashboard
from src.data.realtime.websocket.client import MultiStreamManager, StreamConfig
from src.indicators.advanced.multi_timeframe import (
    MultiTimeframeAnalyzer,
    TimeframeConfig,
)
from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators
from src.analysis.indicators.momentum.momentum import MomentumIndicators
from src.analysis.indicators.volume.volume_analysis import VolumeAnalysis
from src.analysis.indicators.trend.indicators import TrendIndicators
from src.analysis.indicators.orderflow.orderflow_analysis import (
    OrderFlowAnalysis,
    OrderFlowConfig,
)
from src.analysis.indicators.volatility.volatility import VolatilityIndicators
from src.ai.cnn_lstm import CNNLSTM
from src.ai.ppo_gtrxl import PPOGTrXL
from src.ai.hybrid_model import HybridAI
from src.quantum.qsvm import QuantumTradingModel as QuantumSVM
from src.risk_management.circuit_breakers import CircuitBreaker
from src.risk_management.position_manager import PositionManager
from src.notifications.telegram_bot import TelegramBot
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import (
    ArbitrageScanner as ArbitrageEngine,
)
from src.liquidity_heatmap.visualization import generate_heatmap
from web_interface.app.services.news_analyzer import NewsAnalyzer
from src.backtesting.advanced.quantum_backtest import QuantumBacktester, BacktestConfig
from src.backtesting.core.backtest_engine import BacktestEngine
# 7. Constantes de nettoyage
cleanup_lock = asyncio.Lock()
cleanup_in_progress = False
last_cleanup_time = 0
CLEANUP_COOLDOWN = 5
# 8. Constantes WebSocket globales
WEBSOCKET_CONFIG = {
    "RECONNECT_DELAY": 1.0,
    "MESSAGE_TIMEOUT": 30.0,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 5.0,
    "STREAM_TYPES": ["ticker", "depth", "kline"],
}
def setup_asyncio():
    """Configure l'environnement asyncio pour Streamlit"""
    try:
        if not st.session_state.get("loop"):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            nest_asyncio.apply()
            st.session_state.loop = loop
        return st.session_state.loop
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║             ASYNCIO SETUP ERROR                  ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
╚═════════════════════════════════════════════════╝
        """
        )
        return None
class StreamlitSessionManager:
    """Gestionnaire de session Streamlit avec protection et logging améliorés"""
    def __init__(self):
        """Initialisation du gestionnaire de session"""
        self.logger = logging.getLogger(__name__)
        # Initialisation immédiate de la session
        if "session_initialized" not in st.session_state:
            if self._initialize_session_state():
                self._log_initialization()
    def _initialize_session_state(self):
        """Initialise l'état de la session avec des valeurs sûres"""
        try:
            # États par défaut avec horodatage
            default_state = {
                # États de base
                "session_id": self.session_id,
                "initialized": True,
                "session_initialized": True,
                # États du bot
                "bot_running": False,
                "portfolio": None,
                "latest_data": {},
                "indicators": None,
                "refresh_count": 0,
                # États de la boucle événementielle
                "loop": None,
                "error_count": 0,
                # États WebSocket
                "ws_status": "disconnected",
                "ws_initialized": False,
                "ws_connection_status": "disconnected",
                # Protections
                "keep_alive": True,
                "prevent_cleanup": True,
                "force_cleanup": False,
                "cleanup_allowed": False,
            }
            # Initialisation des états manquants uniquement
            for key, value in default_state.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            return True
        except Exception as e:
            self._log_error("Session state initialization error", e)
            return False
    def _log_initialization(self):
        """Log de l'initialisation de la session"""
        self.logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION INITIALIZED                    ║
╠═════════════════════════════════════════════════╣
║ Session ID: {self.session_id}
║ Status: Active
╚═════════════════════════════════════════════════╝
        """
        )
    def _log_error(self, message, error):
        """Log unifié des erreurs"""
        self.logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION ERROR                          ║
╠═════════════════════════════════════════════════╣
║ Error: {message}
║ Details: {str(error)}
║ Type: {type(error).__name__}
║ Session ID: {self.session_id}
╚═════════════════════════════════════════════════╝
        """
        )
        # Incrément du compteur d'erreurs
        st.session_state.error_count = st.session_state.get("error_count", 0) + 1
    def _log_protection(self):
        """Log de la protection de session"""
        self.logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION PROTECTED                      ║
╠═════════════════════════════════════════════════╣
║ Session ID: {self.session_id}
║ Last Action: {st.session_state.get('last_action_time')}
╚═════════════════════════════════════════════════╝
        """
        )
    def protect_session(self):
        """Protection renforcée de la session"""
        try:
            # Éviter les protections multiples
            if st.session_state.get("session_protected"):
                return True
            # Mise à jour du timestamp
                "%Y-%m-%d %H:%M:%S"
            )
            # Activation des protections
            st.session_state.prevent_cleanup = True
            st.session_state.keep_alive = True
            st.session_state.force_cleanup = False
            st.session_state.cleanup_allowed = False
            st.session_state.session_protected = True
            self._log_protection()
            return True
        except Exception as e:
            self._log_error("Session protection error", e)
            return False
    def allow_cleanup(self):
        """Autorisation sécurisée du nettoyage"""
        try:
            # Vérification de l'état du bot
            if st.session_state.get("bot_running", False):
                logger.warning("Cannot allow cleanup while bot is running")
                return False
            # Configuration du nettoyage
            st.session_state.cleanup_allowed = True
            st.session_state.force_cleanup = True
            st.session_state.prevent_cleanup = False
            st.session_state.keep_alive = False
            self._log_cleanup_authorization()
            return True
        except Exception as e:
            self._log_error("Cleanup authorization error", e)
            return False
    def get_session_info(self):
        """Récupération des informations de session"""
        try:
            info = {
                "session_id": self.session_id,
                "last_action": st.session_state.get("last_action_time"),
                "session_initialized": st.session_state.get(
                    "session_initialized", False
                ),
                "bot_running": st.session_state.get("bot_running", False),
                "ws_initialized": st.session_state.get("ws_initialized", False),
                "error_count": st.session_state.get("error_count", 0),
            }
            return info
        except Exception as e:
            self._log_error("Session info retrieval error", e)
            return None
def _setup_and_verify_event_loop():
    """Configure et vérifie la boucle d'événements avec gestion d'erreur améliorée"""
    try:
        # Vérification de l'existence d'une boucle
        if not st.session_state.get("loop"):
            # Création et configuration de la nouvelle boucle
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply()
            # Sauvegarde dans la session
            st.session_state.loop = loop
            # Log de succès d'initialisation
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP INITIALIZED              ║
╠═════════════════════════════════════════════════╣
║ Status: Successfully configured
║ Loop ID: {id(loop)}
╚═════════════════════════════════════════════════╝
            """
            )
            return loop
        # Vérification de la boucle existante
        existing_loop = st.session_state.loop
        if existing_loop.is_closed():
            logger.warning(
                f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP CLOSED                   ║
╠═════════════════════════════════════════════════╣
║ Status: Creating new loop
║ Previous Loop ID: {id(existing_loop)}
╚═════════════════════════════════════════════════╝
            """
            )
            # Création d'une nouvelle boucle
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            nest_asyncio.apply()
            st.session_state.loop = new_loop
            return new_loop
        # Retour de la boucle existante
        logger.debug(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP VERIFIED                 ║
╠═════════════════════════════════════════════════╣
║ Status: Using existing loop
║ Loop ID: {id(existing_loop)}
╚═════════════════════════════════════════════════╝
        """
        )
        return existing_loop
    except Exception as e:
        # Log d'erreur détaillé
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP ERROR                    ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ Details: {traceback.format_exc()}
╚═════════════════════════════════════════════════╝
        """
        )
        # Incrément du compteur d'erreurs
        st.session_state.error_count = st.session_state.get("error_count", 0) + 1
        return None
    finally:
        # Mise à jour du timestamp
# Création de l'instance globale avec vérification
try:
    session_manager = StreamlitSessionManager()
    logger.info("✅ Session manager initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize session manager: {e}")
    session_manager = None
class WebSocketManager:
    def __init__(self, bot):
        self.bot = bot
        self.streams = {}
        self.running = False
        self.lock = asyncio.Lock()
        # Correction des valeurs par défaut
        self.pairs = bot.config.get("TRADING", {}).get(
            "pairs", ["BTC/USDT", "ETH/USDT"]
        )
        self.timeframes = bot.config.get("TRADING", {}).get(
            "timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
        self.retry_count = 0
        self.max_retries = WEBSOCKET_CONFIG["MAX_RETRIES"]
        self.retry_delay = WEBSOCKET_CONFIG["RETRY_DELAY"]
    async def start(self):
        """Démarre les WebSockets"""
        async with self.lock:
            if self.running:
                return True
            try:
                # Initialisation du client Binance
                self.bot.binance_ws = await AsyncClient.create(
                    api_key=os.getenv("BINANCE_API_KEY"),
                    api_secret=os.getenv("BINANCE_API_SECRET"),
                )
                # Initialisation du socket manager
                self.bot.socket_manager = BinanceSocketManager(self.bot.binance_ws)
                # Configuration des streams
                if not await self._setup_streams():
                    raise Exception("Failed to setup streams")
                self.running = True
                return True
            except Exception as e:
                logger.error(f"WebSocket start error: {e}")
                await self.cleanup()
                return False
    async def _setup_streams(self):
        """Configure les streams"""
        try:
            for pair in self.pairs:
                # Stream de trades
                ts = self.bot.socket_manager.trade_socket(pair)
                self.streams[f"{pair}_trades"] = asyncio.create_task(
                    self._handle_stream(ts, "trade", pair)
                )
                # Stream d'orderbook
                ds = self.bot.socket_manager.depth_socket(pair)
                self.streams[f"{pair}_depth"] = asyncio.create_task(
                    self._handle_stream(ds, "depth", pair)
                )
                # Stream de klines
                for tf in self.timeframes:
                    ks = self.bot.socket_manager.kline_socket(pair, tf)
                    self.streams[f"{pair}_kline_{tf}"] = asyncio.create_task(
                        self._handle_stream(ks, "kline", pair, tf)
                    )
            return True
        except Exception as e:
            logger.error(f"Stream setup error: {e}")
            return False
    async def _handle_stream(self, socket, stream_type, pair, timeframe=None):
        """Gère un stream WebSocket"""
        while self.running:
            try:
                async with socket as sock:
                    msg = await sock.recv()
                    if msg:
                        # Traitement selon le type
                        if stream_type == "trade":
                            await self.bot._handle_trade(msg)
                        elif stream_type == "depth":
                            await self.bot._handle_orderbook(msg)
                        elif stream_type == "kline":
                            await self.bot._handle_kline(msg)
            except Exception as e:
                if "shutdown" not in str(e).lower() and "closed" not in str(e).lower():
                    logger.error(f"Stream error ({stream_type}-{pair}): {e}")
                    if self.running:
                        await asyncio.sleep(self.retry_delay)
                        continue
                return
    # Dans la méthode cleanup()
async def cleanup(self):
    """Nettoie les ressources WebSocket"""
    try:
        self.running = False
        # Annulation des tâches
        for stream in self.streams.values():
            if not stream.done():
                stream.cancel()
                try:
                    await stream
                except asyncio.CancelledError:
                    pass
        self.streams.clear()
        # Fermeture du socket manager
        if hasattr(self.bot, "socket_manager") and self.bot.socket_manager:
            try:
                # Modification pour utiliser stop_socket
                for socket in self.bot.socket_manager.sockets:
                    await self.bot.socket_manager.stop_socket(socket)
                self.bot.socket_manager = None
            except Exception as e:
                self.logger.warning(f"Error closing socket manager: {e}")
        # Fermeture du client Binance
        if hasattr(self.bot, "binance_ws") and self.bot.binance_ws:
            try:
                await self.bot.binance_ws.close_connection()
            except Exception as e:
                self.logger.warning(f"Error closing Binance client: {e}")
            self.bot.binance_ws = None
    except Exception as e:
        self.logger.error(f"Cleanup error: {e}")
# Définition de la classe SessionManager
class SessionManager:
    def __init__(self):
        self.sessions = set()
    def register(self, session):
        self.sessions.add(session)
        logging.getLogger(__name__).info(
            f"New session registered (active: {len(self.sessions)})"
        )
    def unregister(self, session):
        self.sessions.discard(session)
        logging.getLogger(__name__).info(
            f"Session unregistered (remaining: {len(self.sessions)})"
        )
class RegimeDetector:
    """Détecteur de régimes de marché"""
    def __init__(self):
        self.current_regime = None
        self.logger = logging.getLogger(__name__)
    def predict(self, indicators_analysis):
        try:
            regime = "Unknown"
            if indicators_analysis:
                trend_strength = 0
                volatility = 0
                volume = 0
                for timeframe_data in indicators_analysis.values():
                    if "trend" in timeframe_data:
                        trend_strength += timeframe_data["trend"].get(
                            "trend_strength", 0
                        )
                    if "volatility" in timeframe_data:
                        volatility += timeframe_data["volatility"].get(
                            "current_volatility", 0
                        )
                    if "volume" in timeframe_data:
                        volume += float(
                            timeframe_data["volume"]
                            .get("volume_profile", {})
                            .get("strength", 0)
                        )
                if trend_strength > 0.7:
                    regime = "Trending"
                elif volatility > 0.7:
                    regime = "Volatile"
                elif volume > 0.7:
                    regime = "High Volume"
                else:
                    regime = "Ranging"
            self.current_regime = regime
            self.logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           MARKET REGIME DETECTION                ║
╠═════════════════════════════════════════════════╣
║ Régime: {regime}
╚═════════════════════════════════════════════════╝
            """
            )
            return regime
        except Exception as e:
            self.logger.error(f"❌ Erreur détection régime: {e}")
            return "Error"
def setup_event_loop() -> AbstractEventLoop:
    """Configure l'event loop pour Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply()
    return loop
def init_session_state():
    """Initialize session state variables with strong defaults"""
    session_vars = {
        "initialized": False,
        "bot_running": False,
        "portfolio": None,
        "latest_data": {},
        "indicators": None,
        "refresh_count": 0,
        "loop": None,
        "ws_status": "disconnected",
        "error_count": 0,
        "keep_alive": True,  # Force à True
        "prevent_cleanup": True,  # Force à True
        "force_cleanup": False,  # Force à False
        "ws_initialized": False,
        "cleanup_allowed": False,  # Nouveau flag
    }
    for var, default in session_vars.items():
        # Ne pas écraser les valeurs existantes pour keep_alive et prevent_cleanup
        if var in ["keep_alive", "prevent_cleanup"]:
            st.session_state.setdefault(var, True)
        else:
            st.session_state[var] = default
# Configuration du bot
load_dotenv()
config = {
    "NEWS": {"enabled": True, "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", "")},
    "BINANCE": {
        "API_KEY": os.getenv("BINANCE_API_KEY"),
        "API_SECRET": os.getenv("BINANCE_API_SECRET"),
    },
    "ARBITRAGE": {
        "exchanges": ["binance", "bitfinex", "kraken"],
        "min_profit": 0.001,
        "max_trade_size": 1000,
        "pairs": ["BTC/USDC", "ETH/USDC"],
        "timeout": 5,
        "volume_filter": 1000,
        "price_check": True,
        "max_slippage": 0.0005,
    },
    "TRADING": {
        "base_currency": "USDC",
        "pairs": ["BTC/USDC", "ETH/USDC"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "study_period": "7d",
    },
    "RISK": {
        "max_drawdown": 0.05,
        "daily_stop_loss": 0.02,
        "position_sizing": "volatility_based",
        "circuit_breaker": {
            "market_crash": True,
            "liquidity_shock": True,
            "black_swan": True,
        },
    },
    "AI": {
        "confidence_threshold": 0.75,
        "min_training_size": 1000,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "n_epochs": 10,
        "gtrxl_layers": 6,
        "embedding_dim": 512,
        "dropout": 0.1,
        "gradient_clip": 0.5,
    },
    "INDICATORS": {
        "trend": {
            "supertrend": {"period": 10, "multiplier": 3},
            "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52},
            "ema_ribbon": [5, 10, 20, 50, 100, 200],
        },
        "momentum": {
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "stoch_rsi": {"period": 14, "k": 3, "d": 3},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
        },
        "volatility": {
            "bbands": {"period": 20, "std_dev": 2},
            "keltner": {"period": 20, "atr_mult": 2},
            "atr": {"period": 14},
        },
        "volume": {
            "vwap": {"anchor": "session"},
            "obv": {"signal": 20},
            "volume_profile": {"price_levels": 100},
        },
        "orderflow": {
            "delta": {"window": 100},
            "cvd": {"smoothing": 20},
            "imbalance": {"threshold": 0.2},
        },
    },
}
@st.cache_resource(ttl=None)
def get_bot():
    """Create or get the bot instance with lifecycle protection"""
    # 1. Vérifier si le bot existe déjà
    if "bot_instance" in st.session_state and st.session_state.bot_instance is not None:
        return st.session_state.bot_instance
    try:
        # 2. Log de création
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║             CREATING BOT INSTANCE                ║
╠═════════════════════════════════════════════════╣
╚═════════════════════════════════════════════════╝
        """
        )
        # 3. Configuration de l'event loop
        if not st.session_state.get("loop"):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                nest_asyncio.apply()
                st.session_state.loop = loop
                logger.info("✅ Event loop configured successfully")
            except Exception as loop_error:
                logger.error(
                    f"""
╔═════════════════════════════════════════════════╗
║             EVENT LOOP ERROR                     ║
╠═════════════════════════════════════════════════╣
║ Error: {str(loop_error)}
╚═════════════════════════════════════════════════╝
                """
                )
                raise
        # 4. Création et initialisation du bot
        bot = TradingBotM4()
        async def initialize_bot():
            try:
                if not await bot.start():
                    raise Exception("Bot initialization failed")
                bot._initialized = True
                logger.info("✅ Bot initialization successful")
                return bot
            except Exception as init_error:
                logger.error(
                    f"""
╔═════════════════════════════════════════════════╗
║             INITIALIZATION ERROR                 ║
╠═════════════════════════════════════════════════╣
║ Error: {str(init_error)}
╚═════════════════════════════════════════════════╝
                """
                )
                raise
        # 5. Exécution de l'initialisation
        try:
            loop = st.session_state.loop
            bot = loop.run_until_complete(initialize_bot())
            if not bot or not getattr(bot, "_initialized", False):
                raise Exception("Bot initialization incomplete")
            # 6. Sauvegarde dans session state
            st.session_state.bot_instance = bot
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║             BOT INSTANCE READY                   ║
╠═════════════════════════════════════════════════╣
║ Status: {bot.ws_connection.get('status', 'initializing')}
║ Trading Mode: {getattr(bot, 'trading_mode', 'production')}
╚═════════════════════════════════════════════════╝
            """
            )
            return bot
        except RuntimeError as e:
            if "This event loop is already running" in str(e):
                logger.warning("⚠️ Event loop already running, applying nest_asyncio")
                nest_asyncio.apply()
                bot = loop.run_until_complete(initialize_bot())
                return bot
            raise
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║             BOT CREATION ERROR                   ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
╚═════════════════════════════════════════════════╝
        """
        )
        # 7. Nettoyage en cas d'erreur
        if hasattr(bot, "_cleanup"):
            try:
                st.session_state.loop.run_until_complete(bot._cleanup())
            except:
                pass
        if "bot_instance" in st.session_state:
            del st.session_state.bot_instance
        if "loop" in st.session_state:
            del st.session_state.loop
        return None
# Fonction d'aide pour la configuration asyncio
def setup_asyncio():
    """Configure l'environnement asyncio pour Streamlit"""
    try:
        if not st.session_state.get("loop"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            st.session_state.loop = loop
            nest_asyncio.apply()
        return st.session_state.loop
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║             ASYNCIO SETUP ERROR                  ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
╚═════════════════════════════════════════════════╝
        """
        )
        return None
async def setup_streams(bot):
    """Configure and setup WebSocket streams"""
    try:
        tasks = []
        async def setup_single_stream(
            stream_type, setup_func, symbol="BTCUSDT", interval="1m"
        ):
            retry_count = 0
            while retry_count < WEBSOCKET_CONFIG["MAX_RETRIES"]:
                try:
                    logger.info(
                        f"Setting up {stream_type} stream (attempt {retry_count + 1}/{WEBSOCKET_CONFIG['MAX_RETRIES']})..."
                    )
                    # Création du socket avec le bon symbole
                    socket = None
                    if stream_type == "ticker":
                        socket = bot.socket_manager.trade_socket(symbol)
                    elif stream_type == "depth":
                        socket = bot.socket_manager.depth_socket(symbol)
                    elif stream_type == "kline":
                        socket = bot.socket_manager.kline_socket(symbol, interval)
                    if socket:
                        task = asyncio.create_task(
                            handle_socket_message(bot, socket, stream_type)
                        )
                        task.set_name(f"{stream_type}_stream_{symbol}")
                        return task
                    retry_count += 1
                    await asyncio.sleep(WEBSOCKET_CONFIG["RETRY_DELAY"])
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error setting up {stream_type} stream: {e}")
                    if retry_count < WEBSOCKET_CONFIG["MAX_RETRIES"]:
                        await asyncio.sleep(WEBSOCKET_CONFIG["RETRY_DELAY"])
                    else:
                        logger.error(
                            f"Failed to setup {stream_type} stream after {WEBSOCKET_CONFIG['MAX_RETRIES']} attempts"
                        )
                        return None
        # Configuration des streams avec le bon ordre
        ticker_task = await setup_single_stream(
            "ticker", bot.socket_manager.trade_socket
        )
        depth_task = await setup_single_stream("depth", bot.socket_manager.depth_socket)
        kline_task = await setup_single_stream("kline", bot.socket_manager.kline_socket)
        # Collecte des tâches réussies
        tasks = [t for t in [ticker_task, depth_task, kline_task] if t is not None]
        if len(tasks) > 0:
            logger.info(
                f"✅ Successfully setup {len(tasks)}/{len(WEBSOCKET_CONFIG['STREAM_TYPES'])} streams"
            )
            return tasks
        else:
            logger.error("❌ Failed to setup any streams")
            return None
    except Exception as e:
        logger.error(f"❌ Stream setup error: {e}")
        return None
async def cleanup_existing_connections(bot):
    """Nettoie les connexions WebSocket existantes"""
    try:
        # Fermeture du socket manager
        if hasattr(bot, "socket_manager") and bot.socket_manager:
            try:
                # Fermeture des connexions WebSocket individuelles
                for socket_name in dir(bot.socket_manager):
                    if socket_name.startswith("_socket_"):
                        socket = getattr(bot.socket_manager, socket_name)
                        if hasattr(socket, "close"):
                            await socket.close()
                # Fermeture du socket manager lui-même
                if hasattr(bot.socket_manager, "close_connection"):
                    await bot.socket_manager.close_connection()
            except Exception as e:
                logger.warning(f"⚠️ Error closing socket manager: {e}")
            finally:
                bot.socket_manager = None
        # Fermeture du client WebSocket
        if hasattr(bot, "binance_ws") and bot.binance_ws:
            try:
                await bot.binance_ws.close_connection()
            except Exception as e:
                logger.warning(f"⚠️ Error closing Binance client: {e}")
            finally:
                bot.binance_ws = None
        return True
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        return False
async def create_binance_client(bot):
    """
    Crée une nouvelle instance du client Binance
    Args:
        bot: Instance du bot de trading
    """
    try:
        # Création du client avec les credentials
        bot.binance_ws = await AsyncClient.create(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
        )
        # Création du socket manager
        bot.socket_manager = BinanceSocketManager(bot.binance_ws)
        return True
    except Exception as e:
        logger.error(f"❌ Error creating Binance client: {e}")
        return False
async def setup_websocket_streams(bot):
    """Configure les streams WebSocket"""
    try:
        tasks = []
        # Configuration des streams par paire
        for pair in bot.config["TRADING"]["pairs"]:
            # Stream de trades en temps réel
            trade_socket = bot.socket_manager.trade_socket(pair)
            tasks.append(
                asyncio.create_task(handle_socket_message(bot, trade_socket, "trade"))
            )
            # Stream d'orderbook
            depth_socket = bot.socket_manager.depth_socket(pair)
            tasks.append(
                asyncio.create_task(handle_socket_message(bot, depth_socket, "depth"))
            )
            # Stream de klines pour chaque timeframe
            for timeframe in bot.config["TRADING"]["timeframes"]:
                kline_socket = bot.socket_manager.kline_socket(pair, timeframe)
                tasks.append(
                    asyncio.create_task(
                        handle_socket_message(bot, kline_socket, "kline")
                    )
                )
        # Mise à jour du statut de connexion
        bot.ws_connection.update(
            {
                "enabled": True,
                "status": "connected",
                "tasks": tasks,
                "start_time": time.time(),
            }
        )
        # Attendre que tous les streams soient initialisés
        await asyncio.gather(*[asyncio.shield(task) for task in tasks])
        return True
    except Exception as e:
        logger.error(f"❌ Stream setup error: {e}")
        return False
async def initialize_websocket(bot):
    """
    Initialise la connexion WebSocket avec gestion améliorée des erreurs et des reconnexions.
    """
    try:
        # Vérification du statut d'initialisation
        if getattr(bot, "_ws_initializing", False):
            logger.warning("⚠️ Initialisation WebSocket déjà en cours")
            return False
        bot._ws_initializing = True
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║         INITIALISATION WEBSOCKET                ║
╠═════════════════════════════════════════════════╣
╚═════════════════════════════════════════════════╝
        """
        )
        # 1. Vérification des credentials
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            logger.error(
                """
╔═════════════════════════════════════════════════╗
║         ERREUR CREDENTIALS                      ║
╠═════════════════════════════════════════════════╣
║ API Key ou Secret manquants                    ║
╚═════════════════════════════════════════════════╝
            """
            )
            return False
        # 2. Nettoyage des connexions existantes si nécessaire
        if hasattr(bot, "binance_ws") and bot.binance_ws:
            try:
                await bot.binance_ws.close_connection()
                bot.binance_ws = None
            except Exception as cleanup_error:
                logger.warning(
                    f"⚠️ Erreur nettoyage connexion existante: {cleanup_error}"
                )
        # 3. Création du client avec timeout et retry
        try:
            bot.binance_ws = await AsyncClient.create(
                api_key=api_key, api_secret=api_secret, tld="com"
            )
            logger.info("✅ Client Binance initialisé")
        except Exception as client_error:
            logger.error(f"❌ Erreur création client: {client_error}")
            return False
        # 4. Configuration du socket manager avec paramètres optimisés
        try:
            bot.socket_manager = BinanceSocketManager(
                bot.binance_ws,
            )
            logger.info("✅ Socket Manager configuré")
        except Exception as manager_error:
            logger.error(f"❌ Erreur configuration socket manager: {manager_error}")
            return False
        # 5. Configuration des streams avec gestion d'erreur
        try:
            # Définition des streams
            streams = [
                "btcusdt@trade",  # Stream de trades
                "btcusdt@depth",  # Stream d'orderbook
                "btcusdt@kline_1m",  # Stream de klines 1m
            ]
            # Réinitialisation des tâches
            bot.ws_tasks = []
            # Création du socket multiplexé avec retry
            multiplex_socket = bot.socket_manager.multiplex_socket(streams)
            # Création de la tâche principale avec gestion d'erreur
            main_task = asyncio.create_task(
                handle_socket_message(bot, multiplex_socket, "market_data")
            )
            main_task.set_name("main_market_data_stream")
            bot.ws_tasks.append(main_task)
            # Ajout d'un heartbeat pour maintenir la connexion
            heartbeat_task = asyncio.create_task(websocket_heartbeat(bot))
            heartbeat_task.set_name("websocket_heartbeat")
            bot.ws_tasks.append(heartbeat_task)
            logger.info("✅ Streams configurés")
        except Exception as stream_error:
            logger.error(f"❌ Erreur configuration streams: {stream_error}")
            return False
        # 6. Mise à jour du statut de connexion
        bot.ws_connection = {
            "enabled": True,
            "status": "connected",
            "tasks": bot.ws_tasks,
            "reconnect_count": 0,
            "max_reconnects": 3,
        }
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║         WEBSOCKET INITIALISÉ                    ║
╠═════════════════════════════════════════════════╣
║ Status: Connected
║ Streams: {len(streams)}
║ Tasks: {len(bot.ws_tasks)}
╚═════════════════════════════════════════════════╝
        """
        )
        return True
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║         ERREUR INITIALISATION                   ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
╚═════════════════════════════════════════════════╝
        """
        )
        # Nettoyage en cas d'erreur
        try:
            if hasattr(bot, "binance_ws") and bot.binance_ws:
                await bot.binance_ws.close_connection()
            if hasattr(bot, "socket_manager"):
                bot.socket_manager = None
        except:
            pass
        return False
    finally:
        bot._ws_initializing = False
        # Vérification finale de la connexion
        if not bot.ws_connection.get("enabled", False):
            logger.warning("⚠️ WebSocket non initialisé correctement")
async def websocket_heartbeat(bot):
    """Maintient la connexion WebSocket active"""
    while True:
        try:
            if not bot.ws_connection["enabled"]:
                break
            # Update heartbeat timestamp
            await asyncio.sleep(30)  # Heartbeat toutes les 30 secondes
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(5)
async def handle_socket_message(bot, socket, stream_name):
    """Gestion des messages avec meilleure gestion des erreurs"""
    async with socket as tscm:
        while True:
            try:
                msg = await asyncio.wait_for(
                    tscm.recv(), timeout=60  # Timeout plus long pour la réception
                )
                if msg:
                    # Mise à jour des données
                    if "data" not in bot.latest_data:
                        bot.latest_data["data"] = {}
                    bot.latest_data["data"][stream_name] = msg
                    # Mise à jour du timestamp
            except asyncio.TimeoutError:
                # Au lieu de se déconnecter, on continue
                continue
            except Exception as e:
                logger.error(f"Socket error ({stream_name}): {e}")
                await asyncio.sleep(1)
                continue
async def cleanup_websocket(bot):
    """Clean WebSocket resources"""
    try:
        logger.info("🔄 Closing WebSocket...")
        if hasattr(bot, "ws_tasks"):
            for task in bot.ws_tasks:
                task.cancel()
            bot.ws_tasks = []
        if hasattr(bot, "socket_manager"):
            await bot.socket_manager.close()
        if hasattr(bot, "binance_ws"):
            await bot.binance_ws.close_connection()
        bot.ws_connection = {"enabled": False, "status": "disconnected", "tasks": []}
        logger.info("✅ WebSocket closed successfully")
    except Exception as e:
        logger.error(f"❌ WebSocket cleanup error: {e}")
async def cleanup_resources(bot):
    """
    Nettoyage sécurisé des ressources avec protection de session et logging détaillé.
    Args:
        bot: Instance du bot de trading à nettoyer
    Returns:
        bool: True si le nettoyage a réussi, False sinon
    """
    # Log de début de tentative de nettoyage
    logger.info(
        f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP ATTEMPT STARTED                ║
╠═════════════════════════════════════════════════╣
║ Bot Status: {'Running' if st.session_state.get('bot_running') else 'Stopped'}
╚═════════════════════════════════════════════════╝
    """
    )
    # Vérification des conditions de protection
    protection_conditions = {
        "prevent_cleanup": st.session_state.get("prevent_cleanup", True),
        "keep_alive": st.session_state.get("keep_alive", True),
        "bot_running": st.session_state.get("bot_running", False),
        "ws_initializing": getattr(bot, "_ws_initializing", False),
        "bot_initialized": getattr(bot, "_initialized", False),
        "cleanup_in_progress": getattr(bot, "cleanup_in_progress", False),
        "force_cleanup": not st.session_state.get("force_cleanup", False),
        "cleanup_allowed": not st.session_state.get("cleanup_allowed", False),
    }
    # Si une condition de protection est active
    if any(protection_conditions.values()):
        # Log détaillé des conditions qui empêchent le nettoyage
        active_protections = [k for k, v in protection_conditions.items() if v]
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP PREVENTED                      ║
╠═════════════════════════════════════════════════╣
║ Active Protections: {', '.join(active_protections)}
║ Session ID: {st.session_state.get('session_id', 'Unknown')}
╚═════════════════════════════════════════════════╝
        """
        )
        # Renforcer la protection
        session_manager.protect_session()
        return False
    try:
        # Marquer le début du nettoyage
        bot.cleanup_in_progress = True
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP STARTED                        ║
╠═════════════════════════════════════════════════╣
║ WebSocket Status: {bot.ws_connection.get('status', 'unknown')}
╚═════════════════════════════════════════════════╝
        """
        )
        # Fermeture du WebSocket
        await close_websocket(bot)
        # Log de succès
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP SUCCESSFUL                     ║
╠═════════════════════════════════════════════════╣
║ Resources Cleaned: WebSocket, Buffer, Data
╚═════════════════════════════════════════════════╝
        """
        )
        return True
    except Exception as e:
        # Log d'erreur détaillé
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP ERROR                          ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        return False
    finally:
        # Nettoyage final et restauration de la protection
        try:
            bot.cleanup_in_progress = False
            session_manager.protect_session()
            # Log final
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP FINALIZED                      ║
╠═════════════════════════════════════════════════╣
║ Protection Restored: True
║ Session Status: Protected
╚═════════════════════════════════════════════════╝
            """
            )
        except Exception as final_error:
            logger.error(f"Final cleanup error: {final_error}")
async def check_websocket_health(bot):
    """Vérifie l'état du WebSocket et le réinitialise si nécessaire"""
    try:
        # Vérifier si les streams sont actifs
        if not bot.ws_connection.get("tasks"):
            return await reset_websocket(bot)
        # Vérifier l'état des tâches
        active_tasks = [t for t in bot.ws_connection["tasks"] if not t.done()]
        if not active_tasks:
            return await reset_websocket(bot)
        # Vérifier si on reçoit des données
        if not bot.latest_data:
            return await reset_websocket(bot)
        return True
    except Exception as e:
        logger.error(f"❌ WebSocket health check error: {e}")
        await reset_websocket(bot)
        return False
async def close_websocket(bot):
    """Ferme proprement la connexion WebSocket"""
    try:
        logger.info("🔄 Closing WebSocket...")
        # Fermeture des tâches
        if bot.ws_connection and bot.ws_connection.get("tasks"):
            for task in bot.ws_connection["tasks"]:
                try:
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                except:
                    pass
        # Fermeture du socket manager
        if hasattr(bot, "socket_manager") and bot.socket_manager:
            try:
                await asyncio.wait_for(bot.socket_manager.close(), timeout=5.0)
            except:
                pass
            finally:
                bot.socket_manager = None
        # Fermeture du client websocket
        if hasattr(bot, "binance_ws") and bot.binance_ws:
            try:
                await asyncio.wait_for(bot.binance_ws.close_connection(), timeout=5.0)
            except:
                pass
            finally:
                bot.binance_ws = None
        # Fermeture explicite de la session client
        if hasattr(bot, "client_session") and bot.client_session:
            if not bot.client_session.closed:
                await bot.client_session.close()
                await asyncio.sleep(0.1)  # Petit délai pour assurer la fermeture
            bot.client_session = None
        # Réinitialisation de l'état
        bot.ws_connection = {"enabled": False, "status": "disconnected", "tasks": []}
        logger.info("✅ WebSocket closed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ WebSocket close error: {e}")
        return False
async def update_trading_data(bot):
    """Mise à jour des données de trading"""
    try:
        # Récupération des données BTC/USDC
        logger.info("📊 Récupération données pour BTC/USDC")
        btc_data = await fetch_market_data(bot, "BTCUSDT")
        if btc_data:
            bot.latest_data["BTCUSDT"] = btc_data
        # Récupération des données ETH/USDC
        logger.info("📊 Récupération données pour ETH/USDC")
        eth_data = await fetch_market_data(bot, "ETHUSDT")
        if eth_data:
            bot.latest_data["ETHUSDT"] = eth_data
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour données: {e}")
async def handle_ticker_message(bot, msg):
    """Gestion des messages de ticker"""
    try:
        if "s" in msg and "p" in msg:
            symbol = msg["s"]
            price = float(msg["p"])
            # Mise à jour des données
            if not hasattr(bot, "latest_prices"):
                bot.latest_prices = {}
            bot.latest_prices[symbol] = price
            # Mise à jour du timestamp
            bot.ws_connection["last_message"] = time.time()
    except Exception as e:
        logger.error(f"❌ Ticker message error: {e}")
async def handle_kline_message(bot, msg):
    """Gestion des messages de klines"""
    try:
        if "k" in msg:
            kline = msg["k"]
            if all(k in kline for k in ["t", "o", "h", "l", "c", "v"]):
                candle = {
                    "timestamp": kline["t"],
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                }
                if not hasattr(bot, "latest_klines"):
                    bot.latest_klines = []
                bot.latest_klines.append(candle)
                if len(bot.latest_klines) > 1000:
                    bot.latest_klines.pop(0)
    except Exception as e:
        logger.error(f"❌ Kline message error: {e}")
async def handle_depth_message(bot, msg):
    """Gestion des messages d'orderbook"""
    try:
        if "a" in msg and "b" in msg:
            orderbook = {
                "asks": [[float(price), float(qty)] for price, qty in msg["a"]],
                "bids": [[float(price), float(qty)] for price, qty in msg["b"]],
                "timestamp": time.time(),
            }
            if not hasattr(bot, "latest_orderbook"):
                bot.latest_orderbook = {}
            bot.latest_orderbook = orderbook
    except Exception as e:
        logger.error(f"❌ Depth message error: {e}")
async def fetch_market_data(bot, symbol):
    """Récupère les données de marché de manière asynchrone"""
    try:
        # Configuration du timeframe par défaut si non défini
        if not hasattr(bot.config, "timeframe"):
            bot.config["timeframe"] = "1m"  # timeframe par défaut
        # Récupération des données via l'API Binance
        klines = await bot.binance_ws.get_klines(
            symbol=symbol, interval=bot.config["timeframe"]
        )
        # Conversion en format utilisable
        data = []
        for k in klines:
            candle = {
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            data.append(candle)
        return data
    except Exception as e:
        logger.error(f"❌ Erreur récupération données {symbol}: {e}")
        return None
async def update_market_data(bot):
    """Met à jour les données de marché"""
    try:
        data_received = False
        # Récupération BTC/USDC
        logger.info("📊 Récupération données pour BTC/USDC")
        btc_data = await fetch_market_data(bot, "BTCUSDT")
        if btc_data:
            bot.latest_data["BTCUSDT"] = btc_data
            data_received = True
        # Récupération ETH/USDC
        logger.info("📊 Récupération données pour ETH/USDC")
        eth_data = await fetch_market_data(bot, "ETHUSDT")
        if eth_data:
            bot.latest_data["ETHUSDT"] = eth_data
            data_received = True
        if not data_received:
            logger.warning("⚠️ Aucune donnée reçue")
        return data_received
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour données: {e}")
        return False
async def process_market_data(bot, symbol):
    """Traite les données de marché pour un symbole"""
    try:
        data = bot.latest_data[symbol]
        if not data:
            return
        # Calcul des indicateurs
        if not hasattr(bot, "indicators"):
            bot.indicators = {}
        if symbol not in bot.indicators:
            bot.indicators[symbol] = {}
        # Mise à jour des indicateurs
        await update_indicators(bot, symbol, data)
        # Vérification des signaux
        await check_signals(bot, symbol)
    except Exception as e:
        logger.error(f"❌ Erreur traitement données {symbol}: {e}")
async def cleanup_session(bot):
    """Nettoyage d'une session avec verrou et cooldown"""
    global cleanup_in_progress, last_cleanup_time
    try:
        # Vérification du cooldown
        current_time = time.time()
        if current_time - last_cleanup_time < CLEANUP_COOLDOWN:
            return
        # Utilisation d'un verrou pour éviter les nettoyages simultanés
        async with cleanup_lock:
            if cleanup_in_progress:
                return
            cleanup_in_progress = True
            last_cleanup_time = current_time
            try:
                # Nettoyage des ressources
                await cleanup_resources(bot)
                # Un seul message de log
                logger.info("✅ Session cleaned successfully")
                logger.info(
                    """
╔═════════════════════════════════════════════════╗
║              CLEANUP COMPLETED                   ║
╠═════════════════════════════════════════════════╣
║ All resources cleaned successfully              ║
╚═════════════════════════════════════════════════╝
                """
                )
            finally:
                cleanup_in_progress = False
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
async def process_ws_message(bot, msg):
    """Process WebSocket messages"""
    try:
        if not msg:
            logger.warning("Empty message received")
            return
        if "e" not in msg:
            logger.warning(f"Invalid message format: {msg}")
            return
        if msg["e"] == "ticker":
            # Mise à jour du prix
            bot.latest_data["price"] = float(msg["c"])
            bot.latest_data["volume"] = float(msg["v"])
            logger.debug(f"💰 Price updated: {bot.latest_data['price']}")
        elif msg["e"] == "depth":
            # Mise à jour de l'orderbook
            bot.latest_data["orderbook"] = {"bids": msg["b"][:5], "asks": msg["a"][:5]}
            logger.debug("📚 Orderbook updated")
        elif msg["e"] == "kline":
            # Mise à jour des klines
            k = msg["k"]
            bot.latest_data["klines"] = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
            }
            logger.debug("📊 Klines updated")
        # Mise à jour du timestamp
        bot.latest_data["timestamp"] = msg.get("E", int(time.time() * 1000))
        bot.ws_connection["last_message"] = time.time()
    except Exception as e:
        logger.error(f"❌ Message processing error: {e}")
class TradingEnv(gym.Env):
    """Environment d'apprentissage par renforcement pour le trading"""
    def __init__(self, trading_pairs, timeframes):
        super().__init__()
        self.trading_pairs = trading_pairs
        self.timeframes = timeframes
        # Espace d'observation: 42 features par paire/timeframe
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(trading_pairs) * len(timeframes) * 42,),
            dtype=np.float32,
        )
        # Espace d'action: allocation par paire entre 0 et 1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(len(trading_pairs),), dtype=np.float32
        )
        # Paramètres d'apprentissage
        self.reward_scale = 1.0
        self.position_history = []
        self.done_penalty = -1.0
        # Initialisation des métriques
        self.metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "positions": [],
            "actions": [],
        }
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.observation_space.shape)
        self.position_history.clear()
        return self.state, {}
    def step(self, action):
        # Validation de l'action
        if not self.action_space.contains(action):
            logger.warning(f"Action invalide: {action}")
            action = np.clip(action, self.action_space.low, self.action_space.high)
        # Calcul de la récompense
        reward = self._calculate_reward(action)
        # Mise à jour de l'état
        self._update_state()
        # Vérification des conditions de fin
        done = self._check_done()
        truncated = False
        # Mise à jour des métriques
        self._update_metrics(action, reward)
        return self.state, reward, done, truncated, self._get_info()
    def _calculate_reward(self, action):
        """Calcule la récompense basée sur le PnL et le risque"""
        try:
            # Calcul du PnL
            pnl = self._calculate_pnl(action)
            # Pénalité pour le risque
            risk_penalty = self._calculate_risk_penalty(action)
            # Reward final
            reward = (pnl - risk_penalty) * self.reward_scale
            return float(reward)
        except Exception as e:
            logger.error(f"Erreur calcul reward: {e}")
            return None
    def _update_state(self):
        """Mise à jour de l'état avec les dernières données de marché"""
        try:
            # Mise à jour des features techniques
            technical_features = self._calculate_technical_features()
            # Mise à jour des features de marché
            market_features = self._calculate_market_features()
            # Combinaison des features
            self.state = np.concatenate([technical_features, market_features])
        except Exception as e:
            logger.error(f"Erreur mise à jour state: {e}")
            return None
    def _check_done(self):
        """Vérifie les conditions de fin d'épisode"""
        # Vérification du stop loss
        if self._check_stop_loss():
            return True
        # Vérification de la durée max
        if len(self.position_history) >= self.max_steps:
            return True
        return False
    def _update_metrics(self, action, reward):
        """Mise à jour des métriques de l'épisode"""
        self.metrics["episode_rewards"].append(reward)
        self.metrics["portfolio_values"].append(self._get_portfolio_value())
        self.metrics["positions"].append(self.position_history[-1])
        self.metrics["actions"].append(action)
    def _get_info(self):
        """Retourne les informations additionnelles"""
        return {
            "portfolio_value": self._get_portfolio_value(),
            "current_positions": (
                self.position_history[-1] if self.position_history else None
            ),
            "metrics": self.metrics,
        }
    def render(self):
        """Affichage de l'environnement"""
        # Affichage des métriques principales
        print(f"\nPortfolio Value: {self._get_portfolio_value():.2f}")
        print(f"Total Reward: {sum(self.metrics['episode_rewards']):.2f}")
        print(f"Number of Trades: {len(self.position_history)}")
class MultiStreamManager:
    def __init__(self, pairs=None, config=None):
        """Initialise le gestionnaire de flux multiples"""
        self.pairs = pairs or []
        self.config = config
        self.exchange = None  # Initialisé plus tard
        self.buffer = CircularBuffer()
    def setup_exchange(self, exchange_id="binance"):
        """Configure l'exchange"""
        self.exchange = Exchange(exchange_id=exchange_id)
class TradingBotM4:
    """Classe principale du bot de trading v4"""
    async def tick(self):
        """Effectue une itération de trading (une fois par refresh)"""
        try:
            # Récupération des données
            market_data = await self.get_latest_data()
            if market_data:
                for pair in self.config["TRADING"]["pairs"]:
                    indicators = await self.calculate_indicators(pair)
                    if indicators:
                        signals = await self.analyze_signals(market_data, indicators)
                        # Ici tu peux gérer l’exécution réelle du trade si besoin
                        # if signals and signals.get('should_trade', False):
                        #     await self.execute_real_trade(signals)
                portfolio = await self.get_real_portfolio()
                if portfolio:
                    st.session_state.portfolio = portfolio
                    st.session_state.latest_data = market_data
                    st.session_state.indicators = indicators
                # Appel périodique de l’analyseur de news
                now = time.time()
                news_result = None
                if now - self.last_news_check > self.news_refresh_interval:
                    news_result = await self.news_analyzer.analyze_news()
                    self.last_news_check = now
                if news_result and news_result.get("status") == "success":
                    st.session_state["news_score"] = news_result["sentiment_summary"]
                    st.session_state["important_news"] = news_result["important_news"]
                    self.logger.info(
                        f"News sentiment: {news_result['sentiment_summary']}"
                    )
                elif news_result is not None:
                    st.session_state["news_score"] = None
                    st.session_state["important_news"] = []
        except Exception as e:
            logger.error(f"Erreur tick: {e}")
    def __init__(self):
        """Initialisation du bot avec gestion améliorée des états"""
        # Flags de contrôle
        self._ws_initializing = False
        self._cleanup_requested = False
        self._initialized = False
        self._reconnecting = False
        # Configuration de la session
        self.session_config = {
            "keep_alive": True,
            "timeout": 60,
            "ping_interval": 20,
            "ping_timeout": 10,
            "reconnect_on_error": True,
            "max_reconnect_attempts": 3,
        }
        # Configuration des streams
        self.stream_config = StreamConfig(
            max_connections=12, reconnect_delay=1.0, buffer_size=10000
        )
        # État du WebSocket
        self.ws_connection = {
            "enabled": False,
            "status": "disconnected",
            "reconnect_count": 0,
            "last_message": None,
            "last_heartbeat": None,
            "tasks": [],
        }
        # Initialisation des composants
        self.buffer = CircularBuffer(maxlen=1000)
        self.indicators = {}
        self.latest_data = {}
        # Configuration des streams (DOIT ÊTRE EN PREMIER)
        self.stream_config = StreamConfig(
            max_connections=12, reconnect_delay=1.0, buffer_size=10000
        )
        self.cleanup_in_progress = False
        self.shutdown_requested = False
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        # Configuration principale
        self.config = {
            "NEWS": {
                "enabled": True,
                "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", ""),
            },
            "BINANCE": {
                "API_KEY": os.getenv("BINANCE_API_KEY", ""),
                "API_SECRET": os.getenv("BINANCE_API_SECRET", ""),
            },
            "TRADING": {
                "pairs": ["BTC/USDT", "ETH/USDT"],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            },
            "ARBITRAGE": {
                "exchanges": ["binance", "bitfinex", "kraken"],
                "pairs": ["BTC/USDT", "ETH/USDT"],
                "min_profit": 0.002,
                "max_trade_size": 1000,
                "timeout": 5,
                "volume_filter": 100000,
                "price_check": True,
                "max_slippage": 0.001,
            },
        }
        api_key = self.config["BINANCE"]["API_KEY"]
        api_secret = self.config["BINANCE"]["API_SECRET"]
        self.exchange = BinanceExchange(api_key, api_secret)
        # Initialisation du WebSocket Manager (AJOUT ICI)
        self.ws_manager = WebSocketManager(self)
        # Initialisation des buffers et données
        self.buffer = CircularBuffer(maxlen=1000)
        self.indicators = {}
        self.latest_data = {}
        # Initialisation du client Binance
        try:
            self.spot_client = BinanceClient(
                api_key=self.config["BINANCE"]["API_KEY"],
                api_secret=self.config["BINANCE"]["API_SECRET"],
            )
            self.logger.info("✅ Spot client initialisé avec succès")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation spot client: {e}")
            self.spot_client = None
        # Configuration du WebSocket
        self.websocket = MultiStreamManager(
            pairs=self.config["TRADING"]["pairs"], config=self.stream_config
        )
        self.ws_connection = {
            "enabled": False,
            "reconnect_count": 0,
            "max_reconnects": 3,
            "last_connection": None,
            "status": "disconnected",
            "last_message": None,
            "last_error": None,
        }
        # Mode de trading et composants
        self.trading_mode = os.getenv("TRADING_MODE", "production")
        self.testnet = False
        self.news_enabled = True
        self.arbitrage_enabled = True
        self.telegram_enabled = True
        # Configuration risque
        self.max_drawdown = 0.05  # 5% max
        self.daily_stop_loss = 0.02  # 2% par jour
        self.max_position_size = 1000  # USDC
        # Interface et monitoring
        self.dashboard = TradingDashboard()
        # Composants principaux
        self.arbitrage_engine = ArbitrageEngine(
            exchanges=self.config["ARBITRAGE"]["exchanges"],
            pairs=self.config["ARBITRAGE"]["pairs"],
            min_profit=self.config["ARBITRAGE"]["min_profit"],
            max_trade_size=self.config["ARBITRAGE"]["max_trade_size"],
            timeout=self.config["ARBITRAGE"]["timeout"],
            volume_filter=self.config["ARBITRAGE"]["volume_filter"],
            price_check=self.config["ARBITRAGE"]["price_check"],
            max_slippage=self.config["ARBITRAGE"]["max_slippage"],
        )
        # Configuration Telegram
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.telegram = TelegramBot()
        # IA et analyse
        self.hybrid_model = HybridAI()
        self.env = TradingEnv(
            trading_pairs=self.config["TRADING"]["pairs"],
            timeframes=self.config["TRADING"]["timeframes"],
        )
        # Gestionnaires de trading
        self.position_manager = PositionManager(
            account_balance=10000,
            max_positions=5,
            max_leverage=3.0,
            min_position_size=0.001,
        )
        self.circuit_breaker = CircuitBreaker(
            crash_threshold=0.1, liquidity_threshold=0.5, volatility_threshold=0.3
        )
        # Configuration timeframes
        self.timeframe_config = TimeframeConfig(
            timeframes=self.config["TRADING"]["timeframes"],
            weights={
                "1m": 0.1,
                "5m": 0.15,
                "15m": 0.2,
                "1h": 0.25,
                "4h": 0.15,
                "1d": 0.15,
            },
        )
        # Composants d'analyse
        self.news_analyzer = NewsAnalyzer()
        self.last_news_check = 0
        self.news_refresh_interval = int(
            os.getenv("NEWS_REFRESH_INTERVAL", 60)
        )  # secondes, configurable
        self.regime_detector = RegimeDetector()
        self.qsvm = QuantumSVM()
        self.client_session = None
    def get_latest_price(self, symbol):
        """Récupère le dernier prix pour un symbole"""
        try:
            # Utiliser directement le symbole USDC
            if not symbol.endswith("USDC"):
                symbol = f"{symbol}USDC"
            # Vérifier si le symbole est supporté
            try:
                price_data = self.spot_client.get_ticker_price(symbol)
                if price_data and "price" in price_data:
                    return float(price_data["price"])
            except Exception as e:
                logger.debug(f"🔎 {symbol} not listed on Binance, skipping.")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return None
    async def start(self):
        """Démarre le bot"""
        # Déplacer la configuration ici, avant le try
        self.config = {
            "NEWS": {
                "enabled": True,
                "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", ""),
            },
            "BINANCE": {
                "API_KEY": os.getenv("BINANCE_API_KEY", ""),
                "API_SECRET": os.getenv("BINANCE_API_SECRET", ""),
            },
            "ARBITRAGE": {  # Ajout de la config manquante
                "exchanges": ["binance"],
                "pairs": ["BTC/USDT", "ETH/USDT"],
                "min_profit": 0.001,
                "max_trade_size": 1000,
                "timeout": 5,
                "volume_filter": 100000,
                "price_check": True,
                "max_slippage": 0.001,
            },
            "TRADING": {  # Ajout de la config manquante
                "pairs": ["BTC/USDT", "ETH/USDT"],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            },
        }
        self.spot_client = None
        self.ws_manager = None
        self.news_analyzer = None
        self.initialized = False
        # Mode de trading
        self.trading_mode = os.getenv("TRADING_MODE", "production")
        self.testnet = False
        # Activation des composants
        self.news_enabled = True
        self.arbitrage_enabled = True
        self.telegram_enabled = True
        # Configuration risque
        self.max_drawdown = 0.05  # 5% max
        self.daily_stop_loss = 0.02  # 2% par jour
        self.max_position_size = 1000  # USDC
        # Configuration des streams
        self.stream_config = StreamConfig(
            max_connections=12, reconnect_delay=1.0, buffer_size=10000
        )
        # Initialisation du MultiStreamManager
        self.ws_manager = WebSocketManager(self)
        # Configuration de l'exchange
        self.websocket.setup_exchange("binance")
        self.buffer = CircularBuffer()
        # Interface et monitoring
        self.dashboard = TradingDashboard()
        try:
            self.logger.info("Starting bot initialization...")
            # Composants principaux
            self.arbitrage_engine = ArbitrageEngine(
                exchanges=self.config["ARBITRAGE"][
                    "exchanges"
                ],  # Correction: self.config au lieu de config
                pairs=self.config["ARBITRAGE"]["pairs"],
                min_profit=self.config["ARBITRAGE"]["min_profit"],
                max_trade_size=self.config["ARBITRAGE"]["max_trade_size"],
                timeout=self.config["ARBITRAGE"]["timeout"],
                volume_filter=self.config["ARBITRAGE"]["volume_filter"],
                price_check=self.config["ARBITRAGE"]["price_check"],
                max_slippage=self.config["ARBITRAGE"]["max_slippage"],
            )
            # Configuration Telegram
            self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
            self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
            self.telegram = TelegramBot()
            # IA et analyse
            self.hybrid_model = HybridAI()
            self.env = TradingEnv(
                trading_pairs=self.config["TRADING"][
                    "pairs"
                ],  # Correction: self.config au lieu de config
                timeframes=self.config["TRADING"]["timeframes"],
            )
            # Gestionnaires de trading
            self.position_manager = PositionManager(
                account_balance=10000,
                max_positions=5,
                max_leverage=3.0,
                min_position_size=0.001,
            )
            self.circuit_breaker = CircuitBreaker(
                crash_threshold=0.1, liquidity_threshold=0.5, volatility_threshold=0.3
            )
            # Configuration timeframes
            self.timeframe_config = TimeframeConfig(
                timeframes=self.config["TRADING"][
                    "timeframes"
                ],  # Correction: self.config au lieu de config
                weights={
                    "1m": 0.1,
                    "5m": 0.15,
                    "15m": 0.2,
                    "1h": 0.25,
                    "4h": 0.15,
                    "1d": 0.15,
                },
            )
            self.news_analyzer = NewsAnalyzer()
            self.regime_detector = RegimeDetector()
            self.client_session = None
            # Démarrage du WebSocket Manager
            if not await self.ws_manager.start():
                raise Exception("Failed to start WebSocket manager")
            # Configuration des composants
            if not await self._setup_components():
                raise Exception("Failed to setup components")
            # Mise à jour du statut
            self.initialized = True
            self.logger.info("✅ Bot initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Bot initialization error: {e}")
            await self._cleanup()
            return False
    def _generate_recommendation(self, trend, momentum, volatility, volume):
        try:
            # Compteurs pour les signaux buy/sell (ancienne logique)
            buy_signals = 0
            sell_signals = 0
            # Système de points (nouvelle logique)
            points = 0
            # --- Analyse de la tendance ---
            if trend["primary_trend"] == "bullish":
                buy_signals += 1
                points += 2
            elif trend["primary_trend"] == "bearish":
                sell_signals += 1
            if trend.get("trend_strength", 0) > 25:
                points += 1
            if trend.get("trend_direction", 0) == 1:
                points += 1
            # --- Momentum ---
            if momentum.get("rsi_signal") == "oversold":
                buy_signals += 1
                points += 2
            elif momentum.get("rsi_signal") == "overbought":
                sell_signals += 1
            if momentum.get("stoch_signal") == "buy":
                points += 1
            if momentum.get("stoch_signal") == "buy":
                buy_signals += 1
            if momentum.get("stoch_signal") == "sell":
                sell_signals += 1
            if momentum.get("ultimate_signal") == "buy":
                points += 1
            # --- Volatilité ---
            if volatility.get("bb_signal") == "oversold":
                points += 1
                buy_signals += 1
            elif volatility.get("bb_signal") == "overbought":
                sell_signals += 1
            if volatility.get("kc_signal") == "breakout":
                points += 1
            # --- Volume ---
            if volume.get("mfi_signal") == "buy":
                buy_signals += 1
                points += 1
            elif volume.get("mfi_signal") == "sell":
                sell_signals += 1
            if volume.get("cmf_trend") == "positive":
                points += 1
                buy_signals += 1
            if volume.get("obv_trend") == "up":
                points += 1
                buy_signals += 1
            elif volume.get("obv_trend") == "down":
                sell_signals += 1
            # --- Génération de la recommandation finale ---
            # Par points (plus fin)
            if points >= 8:
                action = "strong_buy"
                confidence = points / 12
            elif points >= 6:
                action = "buy"
                confidence = points / 12
            elif points <= 2:
                action = "strong_sell"
                confidence = 1 - (points / 12)
            elif points <= 4:
                action = "sell"
                confidence = 1 - (points / 12)
            else:
                action = "neutral"
                confidence = 0.5
            # Par signaux purs (pour compatibilité)
            strength = abs(buy_signals - sell_signals)
            signals = {"buy": buy_signals, "sell": sell_signals}
            return {
                "action": action,
                "confidence": confidence,
                "strength": strength,
                "signals": signals,
            }
        except Exception as e:
            logger.error(f"❌ Erreur génération recommandation: {e}")
            return {
                "action": "error",
                "confidence": 0,
                "strength": 0,
                "signals": {"buy": 0, "sell": 0},
                "error": str(e),
            }
    def _generate_analysis_report(self, indicators_analysis, regime):
        """Génère un rapport d'analyse du marché"""
        try:
            report = f"""
╔═════════════════════════════════════════════════╗
║           RAPPORT D'ANALYSE DE MARCHÉ           ║
╠═════════════════════════════════════════════════╣    
║ Régime: {regime}                               ║
╚═════════════════════════════════════════════════╝
    📊 Analyse par Timeframe:
    """
            for timeframe, analysis in indicators_analysis.items():
                report += f"""
🕒 {timeframe}:
├─ 📈 Tendance: {analysis['trend']['trend_strength']}
├─ 📊 Volatilité: {analysis['volatility']['current_volatility']}
├─ 📉 Volume: {analysis['volume']['volume_profile']['strength']}
└─ 🎯 Signal dominant: {analysis['dominant_signal']}
    """
            logger.info("✅ Rapport d'analyse généré avec succès")
            return report
        except Exception as e:
            error_msg = f"""
╔═════════════════════════════════════════════════╗
║                ERREUR RAPPORT                    ║
╠═════════════════════════════════════════════════╣
║ {str(e)}
╚═════════════════════════════════════════════════╝
    """
            logger.error(f"❌ Erreur génération rapport: {e}")
            return error_msg
    async def _initialize_models(self):
        """Initialise les modèles d'IA"""
        try:
            # Calcul des dimensions pour CNNLSTM
            input_shape = (
                len(config["TRADING"]["timeframes"]),  # Nombre de timeframes
                len(config["TRADING"]["pairs"]),  # Nombre de paires
                42,  # Nombre de features par candlestick
            )
            # Calcul des dimensions pour PPO-GTrXL
            state_dim = input_shape[0] * input_shape[1] * input_shape[2]
            action_dim = len(config["TRADING"]["pairs"])
            # Initialisation des modèles
            self.models = {
                "ppo_gtrxl": PPOGTrXL(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    num_layers=config["AI"]["gtrxl_layers"],
                    d_model=config["AI"]["embedding_dim"],
                ),
                "cnn_lstm": CNNLSTM(input_shape=input_shape),
            }
            # Chargement des poids pré-entraînés
            models_path = os.path.join(current_dir, "models")
            if os.path.exists(models_path):
                for model_name, model in self.models.items():
                    model_path = os.path.join(models_path, f"{model_name}.pt")
                    if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path))
                        logger.info(f"Modèle {model_name} chargé avec succès")
            logger.info("✅ Modèles initialisés avec succès")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur initialisation modèles: {e}")
            return False
    async def _cleanup(self):
        """Nettoie les ressources avant de fermer"""
        try:
            # Fermeture propre du WebSocket
            await close_websocket(self)
            # Nettoyage du buffer
            if hasattr(self, "buffer"):
                try:
                    self.buffer = None  # Au lieu de clear()
                except Exception as buffer_error:
                    logger.error(f"❌ Buffer cleanup error: {buffer_error}")
            # Nettoyage des données
            if hasattr(self, "latest_data"):
                self.latest_data = {}
            if hasattr(self, "indicators"):
                self.indicators = {}
            # Désactivation du mode trading
            if hasattr(st.session_state, "bot_running"):
                st.session_state.bot_running = False
            logger.info(
                """
╔═════════════════════════════════════════════════╗
║              CLEANUP COMPLETED                   ║
╠═════════════════════════════════════════════════╣
║ All resources cleaned successfully              ║
╚═════════════════════════════════════════════════╝
            """
            )
            return True
        except Exception as e:
            return False
    async def start(self):
        """Démarre le bot"""
        try:
            # Initialisation des WebSockets
            if not await self.ws_manager.start():
                raise Exception("Failed to start WebSocket manager")
            # Initialisation des composants
            await self._setup_components()
            # Mise à jour du statut
            self.initialized = True
            logger.info("✅ Bot started successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Bot start error: {e}")
            await self._cleanup()
            return False
    async def check_ws_connection(self):  # Changé de statique à méthode d'instance
        """Check WebSocket connection and reconnect if needed"""
        try:
            if not self.ws_connection["enabled"]:
                if (
                    self.ws_connection["reconnect_count"]
                    < self.ws_connection["max_reconnects"]
                ):
                    logger.info("Attempting WebSocket reconnection...")
                    if await initialize_websocket(self):
                        self.ws_connection["reconnect_count"] = 0
                        return True
                    self.ws_connection["reconnect_count"] += 1
                else:
                    logger.error("Max WebSocket reconnection attempts reached")
                    return False
            return True
        except Exception as e:
            logger.error(f"WebSocket check error: {e}")
            return False
    async def initialize(self):
        """Initialisation asynchrone des connexions"""
        try:
            # Initialisation du client spot si nécessaire
            if not hasattr(self, "spot_client") or self.spot_client is None:
                self.spot_client = BinanceClient(
                    api_key=os.getenv("BINANCE_API_KEY"),
                    api_secret=os.getenv("BINANCE_API_SECRET"),
                )
            # Initialisation du WebSocket MANQUANTE :
            if not getattr(self, "initialized", False):
                success = await self.start()
                if not success:
                    logger.error(
                        "❌ Impossible d'initialiser le WebSocket dans initialize()"
                    )
                    return False
            # Récupération initiale du portfolio
            portfolio = await self.get_real_portfolio()
            if portfolio:
                st.session_state.portfolio = portfolio
                logger.info("✅ Initial portfolio data loaded")
            # Mise à jour du statut
            self.ws_connection.update(
                {"enabled": True, "status": "connected", "last_message": time.time()}
            )
            return True
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            return False
    async def _setup_components(self):
        """Configure les composants du bot"""
        try:
            # Interface et monitoring
            self.dashboard = TradingDashboard()
            # News Analyzer
            self.news_analyzer = NewsAnalyzer()
            # Composants principaux
            self.arbitrage_engine = ArbitrageEngine(
                exchanges=config["ARBITRAGE"]["exchanges"],
                pairs=config["ARBITRAGE"]["pairs"],
                min_profit=config["ARBITRAGE"]["min_profit"],
                max_trade_size=config["ARBITRAGE"]["max_trade_size"],
                timeout=config["ARBITRAGE"]["timeout"],
                volume_filter=config["ARBITRAGE"]["volume_filter"],
                price_check=config["ARBITRAGE"]["price_check"],
                max_slippage=config["ARBITRAGE"]["max_slippage"],
            )
            # Configuration des analyseurs et modèles
            await self._initialize_analyzers()
            await self._initialize_models()
            return True
        except Exception as e:
            logger.error(f"Setup components error: {e}")
            return False
    async def _initialize_analyzers(self):
        """Initialize all analysis components"""
        self.advanced_indicators = MultiTimeframeAnalyzer(config=self.timeframe_config)
        self.orderflow_analysis = OrderFlowAnalysis(
            config=OrderFlowConfig(tick_size=0.1)
        )
        self.volume_analysis = VolumeAnalysis()
        self.volatility_indicators = VolatilityIndicators()
    def add_indicators(self, df):
        """Ajoute tous les indicateurs (130+) au DataFrame"""
        try:
            # Ajout de tous les indicateurs techniques
            df_with_indicators = ta.add_all_ta_features(
                df,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=True,
            )
            # Organisez les indicateurs par catégories
            indicators = {
                "trend": {
                    "sma_fast": df_with_indicators["trend_sma_fast"],
                    "sma_slow": df_with_indicators["trend_sma_slow"],
                    "ema_fast": df_with_indicators["trend_ema_fast"],
                    "ema_slow": df_with_indicators["trend_ema_slow"],
                    "adx": df_with_indicators["trend_adx"],
                    "adx_pos": df_with_indicators["trend_adx_pos"],
                    "adx_neg": df_with_indicators["trend_adx_neg"],
                    "vortex_ind_pos": df_with_indicators["trend_vortex_ind_pos"],
                    "vortex_ind_neg": df_with_indicators["trend_vortex_ind_neg"],
                    "vortex_ind_diff": df_with_indicators["trend_vortex_ind_diff"],
                    "trix": df_with_indicators["trend_trix"],
                    "mass_index": df_with_indicators["trend_mass_index"],
                    "cci": df_with_indicators["trend_cci"],
                    "dpo": df_with_indicators["trend_dpo"],
                    "kst": df_with_indicators["trend_kst"],
                    "kst_sig": df_with_indicators["trend_kst_sig"],
                    "kst_diff": df_with_indicators["trend_kst_diff"],
                    "ichimoku_a": df_with_indicators["trend_ichimoku_a"],
                    "ichimoku_b": df_with_indicators["trend_ichimoku_b"],
                    "visual_ichimoku_a": df_with_indicators["trend_visual_ichimoku_a"],
                    "visual_ichimoku_b": df_with_indicators["trend_visual_ichimoku_b"],
                    "aroon_up": df_with_indicators["trend_aroon_up"],
                    "aroon_down": df_with_indicators["trend_aroon_down"],
                    "aroon_ind": df_with_indicators["trend_aroon_ind"],
                },
                "momentum": {
                    "rsi": df_with_indicators["momentum_rsi"],
                    "stoch": df_with_indicators["momentum_stoch"],
                    "stoch_signal": df_with_indicators["momentum_stoch_signal"],
                    "tsi": df_with_indicators["momentum_tsi"],
                    "uo": df_with_indicators["momentum_uo"],
                    "stoch_rsi": df_with_indicators["momentum_stoch_rsi"],
                    "stoch_rsi_k": df_with_indicators["momentum_stoch_rsi_k"],
                    "stoch_rsi_d": df_with_indicators["momentum_stoch_rsi_d"],
                    "williams_r": df_with_indicators["momentum_wr"],
                    "ao": df_with_indicators["momentum_ao"],
                },
                "volatility": {
                    "bbm": df_with_indicators["volatility_bbm"],
                    "bbh": df_with_indicators["volatility_bbh"],
                    "bbl": df_with_indicators["volatility_bbl"],
                    "bbw": df_with_indicators["volatility_bbw"],
                    "bbp": df_with_indicators["volatility_bbp"],
                    "kcc": df_with_indicators["volatility_kcc"],
                    "kch": df_with_indicators["volatility_kch"],
                    "kcl": df_with_indicators["volatility_kcl"],
                    "kcw": df_with_indicators["volatility_kcw"],
                    "kcp": df_with_indicators["volatility_kcp"],
                    "atr": df_with_indicators["volatility_atr"],
                    "ui": df_with_indicators["volatility_ui"],
                },
                "volume": {
                    "mfi": df_with_indicators["volume_mfi"],
                    "adi": df_with_indicators["volume_adi"],
                    "obv": df_with_indicators["volume_obv"],
                    "cmf": df_with_indicators["volume_cmf"],
                    "fi": df_with_indicators["volume_fi"],
                    "em": df_with_indicators["volume_em"],
                    "sma_em": df_with_indicators["volume_sma_em"],
                    "vpt": df_with_indicators["volume_vpt"],
                    "nvi": df_with_indicators["volume_nvi"],
                    "vwap": df_with_indicators["volume_vwap"],
                },
                "others": {
                    "dr": df_with_indicators["others_dr"],
                    "dlr": df_with_indicators["others_dlr"],
                    "cr": df_with_indicators["others_cr"],
                },
            }
            logger.info(
                f"✅ Indicateurs calculés avec succès pour {len(indicators)} catégories"
            )
            return indicators
        except Exception as e:
            logger.error(f"❌ Erreur calcul indicateurs: {e}")
            return None
    async def _handle_stream(self, stream):
        """Gère un stream de données"""
        try:
            async with stream as tscm:
                while True:
                    msg = await tscm.recv()
                    await self._process_stream_message(msg)
        except Exception as e:
            logger.error(f"Erreur stream: {e}")
            return None
    async def _process_stream_message(self, msg):
        """Traite les messages des streams"""
        try:
            if not msg:
                logger.warning("Message vide reçu")
                return
            if msg.get("e") == "trade":
                await self._handle_trade(msg)
            elif msg.get("e") == "depthUpdate":
                await self._handle_orderbook(msg)
            elif msg.get("e") == "kline":
                await self._handle_kline(msg)
        except Exception as e:
            logger.error(f"Erreur traitement message: {e}")
            return None
    async def _handle_trade(self, msg):
        """Traite un trade"""
        try:
            trade_data = {
                "symbol": msg["s"],
                "price": float(msg["p"]),
                "quantity": float(msg["q"]),
                "time": msg["T"],
                "buyer": msg["b"],
                "seller": msg["a"],
            }
            # Mise à jour du buffer
            self.buffer.update_trades(trade_data)
            # Analyse du volume
            self.volume_analysis.update(trade_data)
            return trade_data
        except Exception as e:
            logger.error(f"Erreur traitement trade: {e}")
            return None
    async def _handle_orderbook(self, msg):
        """Traite une mise à jour d'orderbook"""
        try:
            orderbook_data = {
                "symbol": msg["s"],
                "bids": [[float(p), float(q)] for p, q in msg["b"]],
                "asks": [[float(p), float(q)] for p, q in msg["a"]],
                "time": msg["T"],
            }
            # Mise à jour du buffer
            self.buffer.update_orderbook(orderbook_data)
            # Analyse de la liquidité
            await self._analyze_market_liquidity()
            return orderbook_data
        except Exception as e:
            logger.error(f"Erreur traitement orderbook: {e}")
            return None
    async def _handle_kline(self, msg):
        """Traite une bougie"""
        try:
            kline = msg["k"]
            kline_data = {
                "symbol": msg["s"],
                "interval": kline["i"],
                "time": kline["t"],
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "closed": kline["x"],
            }
            # Mise à jour du buffer
            self.buffer.update_klines(kline_data)
            # Analyse technique si la bougie est fermée
            if kline_data["closed"]:
                await self.analyze_signals(
                    market_data=self.buffer.get_latest_ohlcv(kline_data["symbol"]),
                    indicators=self.advanced_indicators.analyze_timeframe(kline_data),
                )
            return kline_data
        except Exception as e:
            logger.error(f"Erreur traitement kline: {e}")
            return None
    def decision_model(self, features, timestamp=None):
        try:
            policy = self.models["ppo_gtrxl"].get_policy(features)
            value = self.models["ppo_gtrxl"].get_value(features)
            return policy, value
        except Exception as e:
            logger.error(f"[{timestamp}] Erreur decision_model: {e}")
            return None, None
    def _add_risk_management(self, decision, timestamp=None):
        try:
            # Calcul du stop loss
            stop_loss = self._calculate_stop_loss(decision)
            # Calcul du take profit
            take_profit = self._calculate_take_profit(decision)
            # Ajout trailing stop
            trailing_stop = {
                "activation_price": stop_loss * 1.02,
                "callback_rate": 0.01,
            }
            decision.update(
                {
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trailing_stop": trailing_stop,
                }
            )
            return decision
        except Exception as e:
            logger.error(f"[{timestamp}] Erreur risk management: {e}")
            return decision
    async def get_latest_data(self):
        try:
            data = {}
            # Vérification de la connexion WebSocket
            if not hasattr(self, "binance_ws") or self.binance_ws is None:
                logger.warning(
                    "🔄 WebSocket non initialisé, tentative d'initialisation..."
                )
                if not self.initialized:
                    await self.initialize()
                if not hasattr(self, "binance_ws") or self.binance_ws is None:
                    logger.error(
                        "Impossible d'initialiser le WebSocket après tentative."
                    )
                    return None
            for pair in self.config["TRADING"]["pairs"]:
                logger.info(f"📊 Récupération données pour {pair}")
                data[pair] = {}
                try:
                    async def fetch_async():
                        result = {
                            "orderbook": None,
                            "balance": None,
                            "ticker_24h": None,
                            "ticker": None,
                            "ohlcv": None,
                        }
                        # 1. Prix en temps réel via WebSocket (toujours async)
                        if hasattr(self.binance_ws, "get_symbol_ticker"):
                            result["ticker"] = await self.binance_ws.get_symbol_ticker(
                                symbol=pair.replace("/", "")
                            )
                        # 2. & 3. Orderbook et Balance
                        if hasattr(self, "spot_client"):
                            ob_func = self.spot_client.get_order_book
                            bal_func = self.spot_client.get_balance
                            # Orderbook
                            if asyncio.iscoroutinefunction(ob_func):
                                result["orderbook"] = await ob_func(pair)
                            else:
                                result["orderbook"] = ob_func(pair)
                            # Balance
                            if asyncio.iscoroutinefunction(bal_func):
                                result["balance"] = await bal_func()
                            else:
                                result["balance"] = bal_func()
                        # 4. Volume 24h (toujours async)
                        if hasattr(self.binance_ws, "get_24h_ticker"):
                            result["ticker_24h"] = await self.binance_ws.get_24h_ticker(
                                pair.replace("/", "")
                            )
                        # 5. HISTORIQUE OHLCV (pour le backtest !)
                        # PATCH : Ajout d'une vraie liste de dicts pour ohlcv
                        result["ohlcv"] = []
                        try:
                            # Essaye d'utiliser spot_client en priorité pour des données plus fiables
                            klines = None
                            if hasattr(self, "spot_client") and hasattr(
                                self.spot_client, "get_klines"
                            ):
                                k_func = self.spot_client.get_klines
                                if asyncio.iscoroutinefunction(k_func):
                                    klines = await k_func(
                                        pair, interval="1m", limit=200
                                    )
                                else:
                                    klines = k_func(pair, interval="1m", limit=200)
                            elif hasattr(self.binance_ws, "get_klines"):
                                klines = await self.binance_ws.get_klines(
                                    symbol=pair.replace("/", ""),
                                    interval="1m",
                                    limit=200,
                                )
                            # Structure standard OHLCV
                            if klines:
                                result["ohlcv"] = [
                                    {
                                        "timestamp": k[0],
                                        "open": float(k[1]),
                                        "high": float(k[2]),
                                        "low": float(k[3]),
                                        "close": float(k[4]),
                                        "volume": float(k[5]),
                                    }
                                    for k in klines
                                    if len(k) >= 6
                                ]
                        except Exception as hist_e:
                            logger.warning(f"Erreur chargement OHLCV {pair}: {hist_e}")
                        return result
                    async with asyncio.timeout(10.0):
                        result = await fetch_async()
                    # Traitement des résultats
                    if result["ticker"]:
                        data[pair]["price"] = float(result["ticker"]["price"])
                        logger.info(f"💰 Prix {pair}: {data[pair]['price']}")
                    if result["orderbook"]:
                        data[pair]["orderbook"] = {
                            "bids": result["orderbook"]["bids"][:5],
                            "asks": result["orderbook"]["asks"][:5],
                        }
                        logger.info(f"📚 Orderbook mis à jour pour {pair}")
                    if result["balance"]:
                        data[pair]["account"] = result["balance"]
                        logger.info(
                            f"💼 Balance mise à jour: {result['balance'].get('total', 0)} USDC"
                        )
                    if result["ticker_24h"]:
                        data[pair].update(
                            {
                                "volume": float(result["ticker_24h"]["volume"]),
                                "price_change": float(
                                    result["ticker_24h"]["priceChangePercent"]
                                ),
                            }
                        )
                        logger.info(f"📈 Volume 24h {pair}: {data[pair]['volume']}")
                    # AJOUT FORTEMENT RECOMMANDÉ : Toujours une liste de dicts, même vide
                    data[pair]["ohlcv"] = result["ohlcv"] if result["ohlcv"] else []
                    logger.info(
                        f"📊 OHLCV récupéré ({len(data[pair]['ohlcv'])} bougies) pour {pair}"
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ Timeout pour {pair}")
                    continue
                except Exception as inner_e:
                    logger.error(f"❌ Erreur récupération données {pair}: {inner_e}")
                    continue
            # Mise en cache des données si disponibles
            if data and any(data.values()):
                logger.info("✅ Données reçues, mise à jour du buffer")
                for symbol, symbol_data in data.items():
                    if symbol_data:
                        self.buffer.update_data(symbol, symbol_data)
                        self.latest_data[symbol] = symbol_data
                return data
            else:
                logger.warning("⚠️ Aucune donnée reçue")
                return None
        except Exception as e:
            logger.error(f"❌ Erreur critique get_latest_data: {e}")
            return None
    async def calculate_indicators(self, symbol: str) -> dict:
        """Calcule les indicateurs techniques"""
        try:
            data = self.latest_data.get(symbol)
            if not data:
                logger.error(f"❌ Pas de données pour {symbol}")
                return {}
            # Calcul des indicateurs de base
            indicators = {
                "price": data["price"],
                "volume": data["volume"],
                "bid_ask_spread": data["ask"] - data["bid"],
                "high_low_range": data["high"] - data["low"],
                "timestamp": data["timestamp"],
            }
            # Log des données reçues
            logger.info(
                f"Calcul indicateurs pour {symbol}: {data}"
            )  # Log des données reçues
            logger.info(f"Calcul indicateurs pour {symbol}: {data}")
            # Stockage des indicateurs
            self.indicators[symbol] = indicators
            return indicators
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs pour {symbol}: {str(e)}")
            return {}
    async def study_market(self, period="7d"):
        """Analyse initiale du marché"""
        logger.info("🔊 Étude du marché en cours...")
        try:
            # Récupération des données historiques
            if not getattr(self.exchange, "_initialized", False):
                await self.exchange.initialize()
            historical_data = await self.exchange.get_historical_data(
                config["TRADING"]["pairs"], config["TRADING"]["timeframes"], period
            )
            if not historical_data:
                raise ValueError("Données historiques non disponibles")
            # Analyse des indicateurs par timeframe
            indicators_analysis = {}
            for timeframe in config["TRADING"]["timeframes"]:
                try:
                    tf_data = historical_data[timeframe]
                    result = self.advanced_indicators.analyze_timeframe(
                        tf_data, timeframe
                    )
                    indicators_analysis[timeframe] = (
                        {
                            "trend": {"trend_strength": 0},
                            "volatility": {"current_volatility": 0},
                            "volume": {"volume_profile": {"strength": "N/A"}},
                            "dominant_signal": "Neutre",
                        }
                        if result is None
                        else result
                    )
                except Exception as tf_error:
                    logger.error(f"Erreur analyse timeframe {timeframe}: {tf_error}")
                    indicators_analysis[timeframe] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Erreur",
                    }
            # Détection du régime de marché
            regime = self.regime_detector.predict(indicators_analysis)
            logger.info(f"🔈 Régime de marché détecté: {regime}")
            # Génération et envoi du rapport
            try:
                analysis_report = self._generate_analysis_report(
                    indicators_analysis,
                    regime,
                )
                await self.telegram.send_message(analysis_report)
            except Exception as report_error:
                logger.error(f"Erreur génération rapport: {report_error}")
            # Mise à jour du dashboard
            try:
                self.dashboard.update_market_analysis(
                    historical_data=historical_data,
                    indicators=indicators_analysis,
                    regime=regime,
                )
            except Exception as dash_error:
                logger.error(f"Erreur mise à jour dashboard: {dash_error}")
            return regime, historical_data, indicators_analysis
        except Exception as e:
            logger.error(f"Erreur study_market: {e}")
            raise
    async def analyze_signals(self, market_data, indicators=None):
        """Analyse des signaux de trading basée sur tous les indicateurs"""
        try:
            # Si les indicateurs ne sont pas fournis, on les calcule
            if indicators is None:
                indicators = self.add_indicators(market_data)
            if not indicators:
                return None
            # Analyse des tendances
            trend_analysis = {
                "primary_trend": (
                    "bullish"
                    if indicators["trend"]["ema_fast"].iloc[-1]
                    > indicators["trend"]["sma_slow"].iloc[-1]
                    else "bearish"
                ),
                "trend_strength": indicators["trend"]["adx"].iloc[-1],
                "trend_direction": (
                    1 if indicators["trend"]["vortex_ind_diff"].iloc[-1] > 0 else -1
                ),
                "ichimoku_signal": (
                    "buy"
                    if indicators["trend"]["ichimoku_a"].iloc[-1]
                    > indicators["trend"]["ichimoku_b"].iloc[-1]
                    else "sell"
                ),
            }
            # Analyse du momentum
            momentum_analysis = {
                "rsi_signal": (
                    "oversold"
                    if indicators["momentum"]["rsi"].iloc[-1] < 30
                    else (
                        "overbought"
                        if indicators["momentum"]["rsi"].iloc[-1] > 70
                        else "neutral"
                    )
                ),
                "stoch_signal": (
                    "buy"
                    if indicators["momentum"]["stoch_rsi_k"].iloc[-1]
                    > indicators["momentum"]["stoch_rsi_d"].iloc[-1]
                    else "sell"
                ),
                "ultimate_signal": (
                    "buy"
                    if indicators["momentum"]["uo"].iloc[-1] > 70
                    else (
                        "sell"
                        if indicators["momentum"]["uo"].iloc[-1] < 30
                        else "neutral"
                    )
                ),
            }
            # Analyse de la volatilité
            volatility_analysis = {
                "bb_signal": (
                    "oversold"
                    if market_data["close"].iloc[-1]
                    < indicators["volatility"]["bbl"].iloc[-1]
                    else "overbought"
                ),
                "kc_signal": (
                    "breakout"
                    if market_data["close"].iloc[-1]
                    > indicators["volatility"]["kch"].iloc[-1]
                    else "breakdown"
                ),
                "atr_volatility": indicators["volatility"]["atr"].iloc[-1],
            }
            # Analyse du volume
            volume_analysis = {
                "mfi_signal": (
                    "buy"
                    if indicators["volume"]["mfi"].iloc[-1] < 20
                    else (
                        "sell"
                        if indicators["volume"]["mfi"].iloc[-1] > 80
                        else "neutral"
                    )
                ),
                "cmf_trend": (
                    "positive"
                    if indicators["volume"]["cmf"].iloc[-1] > 0
                    else "negative"
                ),
                "obv_trend": (
                    "up" if indicators["volume"]["obv"].diff().iloc[-1] > 0 else "down"
                ),
            }
            # Décision finale
            signal = {
                "trend": trend_analysis,
                "momentum": momentum_analysis,
                "volatility": volatility_analysis,
                "volume": volume_analysis,
                "recommendation": self._generate_recommendation(
                    trend_analysis,
                    momentum_analysis,
                    volatility_analysis,
                    volume_analysis,
                ),
            }
            logger.info(f"✅ Analyse des signaux complétée: {signal['recommendation']}")
            return signal
        except Exception as e:
            logger.error(f"❌ Erreur analyse signaux: {e}")
            return None
    async def setup_real_exchange(self):
        """Configuration sécurisée de l'exchange"""
        try:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError(
                    "Clés API Binance manquantes dans les variables d'environnement"
                )
            # Configuration de l'exchange avec ccxt
            self.exchange = ccxt.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "future",
                        "adjustForTimeDifference": True,
                        "createMarketBuyOrderRequiresPrice": False,
                    },
                }
            )
            # Chargement des marchés de manière synchrone
            self.exchange.load_markets()
            self.spot_client = self.exchange
            self.spot_client = BinanceClient(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
            )
            # Test de la connexion
            balance = self.exchange.fetch_balance()
            if not balance:
                raise ValueError(
                    "Impossible de récupérer le solde - Vérifiez vos clés API"
                )
            logger.info("Exchange configuré avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur configuration exchange: {e}")
            return False
    # 3. Correction de l'envoi des messages Telegram
    async def send_telegram_message(self, message: str):
        """Envoie un message via Telegram"""
        try:
            if hasattr(self, "telegram") and self.telegram.enabled:
                success = await self.telegram.send_message(
                    message=message, parse_mode="HTML"
                )
                if success:
                    logger.info(f"Message Telegram envoyé: {message[:50]}...")
                else:
                    logger.error("Échec envoi message Telegram")
        except Exception as e:
            logger.error(f"Erreur envoi Telegram: {e}")
    async def setup_real_telegram(self):
        """Configuration sécurisée de Telegram"""
        try:
            # Création de l'instance TelegramBot (l'initialisation se fait dans __init__)
            self.telegram = TelegramBot()
            if not self.telegram.enabled:
                logger.warning("Telegram notifications désactivées")
                return False
            # Démarrage du processeur de queue
            await self.telegram.start()
            # Test d'envoi d'un message
            success = await self.telegram.send_message(
                "🤖 Bot de trading démarré", parse_mode="HTML"
            )
            if success:
                logger.info("Telegram configuré avec succès")
                return True
            else:
                logger.error("Échec du test d'envoi Telegram")
                return False
        except Exception as e:
            logger.error(f"Erreur configuration Telegram: {e}")
            return False
    def _get_portfolio_value(self):
        """Récupère la valeur actuelle du portfolio"""
        try:
            if hasattr(self, "position_manager") and hasattr(
                self.position_manager, "positions"
            ):
                return sum(self.position_manager.positions.values())
            return 0.0
        except Exception as e:
            logger.error(f"Erreur calcul portfolio: {e}")
            return None
    def _calculate_total_pnl(self):
        try:
            if hasattr(self, "position_history"):
                return sum(trade.get("pnl", 0) for trade in self.position_history)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating PnL: {e}")
            return 0.0
    async def update_dashboard(self):
        """Met à jour le dashboard en temps réel"""
        try:
            # Mise à jour des données
            portfolio_value = self._get_portfolio_value()
            total_pnl = self._calculate_total_pnl()
            # Mise à jour de l'état de session
            st.session_state.portfolio = {
                "total_value": portfolio_value,
                "daily_pnl": total_pnl,
                "positions": (
                    self.position_manager.get_positions()
                    if hasattr(self, "position_manager")
                    else []
                ),
            }
            st.session_state.latest_data = {
                "price": self.current_price if hasattr(self, "current_price") else 0,
                "volume": self.current_volume if hasattr(self, "current_volume") else 0,
            }
            st.session_state.indicators = (
                self.get_indicators() if hasattr(self, "get_indicators") else None
            )
            return True
        except Exception as e:
            logger.error(f"Dashboard update error: {e}")
            return False
    async def get_real_portfolio(self):
        """
        Récupère le portfolio en temps réel avec les balances et positions.
        """
        try:
            # Vérification et initialisation du spot client
            if not hasattr(self, "spot_client") or self.spot_client is None:
                logger.info(
                    f"""
╔═════════════════════════════════════════════════╗
║         INITIALIZING SPOT CLIENT                 ║
╠═════════════════════════════════════════════════╣
╚═════════════════════════════════════════════════╝
            """
                )
            self.spot_client = BinanceClient(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
            )
            if not self.spot_client:
                raise Exception("Failed to initialize spot client")
            # Récupération des balances de manière asynchrone
            balance = self.spot_client.get_balance()
            if not balance or "balances" not in balance:
                raise Exception("No balance data available")
            logger.info("💰 Balance data received")
            # Traitement des balances
            portfolio = {
                "total_value": 0.0,
                "free": 0.0,
                "used": 0.0,
                "positions": [],
                "daily_pnl": 0.0,
                "volume_24h": 0.0,
                "volume_change": 0.0,
                "timestamp": int(time.time() * 1000),
            }
            # Traitement de chaque asset
            for asset_balance in balance["balances"]:
                try:
                    asset = asset_balance["asset"]
                    free = float(asset_balance["free"])
                    locked = float(asset_balance["locked"])
                    if free > 0 or locked > 0:
                        # Traitement spécial pour USDC
                        if asset == "USDC":
                            portfolio["free"] += free
                            portfolio["used"] += locked
                            portfolio["total_value"] += free + locked
                        else:
                            # Conversion en USDC pour les autres assets
                            try:
                                price = self.get_latest_price(f"{asset}USDC")
                                value = (free + locked) * price
                                if value > 0:
                                    portfolio["total_value"] += value
                                    portfolio["positions"].append(
                                        {
                                            "symbol": f"{asset}/USDC",
                                            "size": free + locked,
                                            "value": value,
                                            "price": price,
                                            "free": free,
                                            "locked": locked,
                                            "timestamp": portfolio["timestamp"],
                                        }
                                    )
                            except Exception as price_error:
                                logger.warning(
                                    f"⚠️ Cannot get price for {asset}: {price_error}"
                                )
                                continue
                except Exception as asset_error:
                    logger.warning(f"⚠️ Error processing {asset}: {asset_error}")
                    continue
            # Récupération des ordres ouverts
            try:
                for pair in self.config["TRADING"]["pairs"]:
                    open_orders = self.spot_client.get_open_orders(pair)
                    if open_orders:
                        for order in open_orders:
                            try:
                                amount = float(order["amount"])
                                price = float(order["price"])
                                if amount > 0:
                                    portfolio["positions"].append(
                                        {
                                            "symbol": order["symbol"],
                                            "size": amount,
                                            "value": price * amount,
                                            "price": price,
                                            "side": order["side"].upper(),
                                            "type": order["type"],
                                            "timestamp": portfolio["timestamp"],
                                            "order_id": order["id"],
                                        }
                                    )
                            except Exception as order_error:
                                logger.warning(
                                    f"⚠️ Error processing order: {order_error}"
                                )
                                continue
            except Exception as orders_error:
                logger.warning(f"⚠️ Cannot fetch open orders: {orders_error}")
            # Calcul des métriques finales
            portfolio.update(
                {
                    "position_count": len(portfolio["positions"]),
                    "total_position_value": sum(
                        pos["value"] for pos in portfolio["positions"]
                    ),
                    "available_margin": portfolio["free"]
                    - sum(pos.get("value", 0) for pos in portfolio["positions"]),
                }
            )
            # Récupération des données de volume sur 24h
            try:
                for pair in self.config["TRADING"]["pairs"]:
                    ticker_24h = self.spot_client.get_24h_ticker(pair)
                    if ticker_24h:
                        portfolio["volume_24h"] += float(ticker_24h["volume"])
                        portfolio["volume_change"] += float(
                            ticker_24h["priceChangePercent"]
                        )
                # Moyenne du changement de volume
                if len(self.config["TRADING"]["pairs"]) > 0:
                    portfolio["volume_change"] /= len(self.config["TRADING"]["pairs"])
            except Exception as volume_error:
                logger.warning(f"⚠️ Cannot fetch 24h volume data: {volume_error}")
            # Log de succès
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║         PORTFOLIO UPDATE SUCCESS                 ║
╠═════════════════════════════════════════════════╣
║ Total Value: {portfolio['total_value']:.2f} USDC
║ Positions: {portfolio['position_count']}
╚═════════════════════════════════════════════════╝
            """
            )
            return portfolio
        except Exception as e:
            logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║         PORTFOLIO UPDATE ERROR                   ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
╚═════════════════════════════════════════════════╝
            """
            )
            # Retourner un portfolio par défaut en cas d'erreur
            return {
                "total_value": 100.59,
                "free": 100.59,
                "used": 0.0,
                "positions": [],
                "position_count": 0,
                "daily_pnl": 0.0,
                "volume_24h": 0.0,
                "volume_change": 0.0,
                "timestamp": int(time.time() * 1000),
                "available_margin": 100.59,
                "total_position_value": 0.0,
            }
    async def execute_real_trade(self, signal):
        """Exécution sécurisée des trades"""
        try:
            # Vérification du solde
            balance = await self.get_real_portfolio()
            if not balance or balance["free"] < signal["amount"] * signal["price"]:
                logger.warning("Solde insuffisant pour le trade")
                return None
            # Calcul stop loss et take profit
            stop_loss = signal["price"] * (1 - signal["risk_ratio"])
            take_profit = signal["price"] * (1 + signal["risk_ratio"] * 2)
            # Placement de l'ordre
            order = await self.exchange.create_order(
                symbol=signal["symbol"],
                type="limit",
                side=signal["side"],
                amount=signal["amount"],
                price=signal["price"],
                params={
                    "stopLoss": {
                        "type": "trailing",
                        "stopPrice": stop_loss,
                        "callbackRate": 1.0,
                    },
                    "takeProfit": {"price": take_profit},
                },
            )
            try:
                await self.telegram.send_message(
                    chat_id=self.chat_id,
                    text=f"""🔵 Nouvel ordre:
Symbol: {order['symbol']}
Type: {order['type']}
Side: {order['side']}
Amount: {order['amount']}
Prix: {order['price']}
Stop Loss: {stop_loss}
Take Profit: {take_profit}""",
                )
            except Exception as msg_error:
                logger.error(f"Erreur envoi notification trade: {msg_error}")
            return order
        except Exception as e:
            logger.error(f"Erreur trade: {e}")
            return None
    async def run_real_trading(self):
        """Boucle de trading réel sécurisée"""
        try:
            # Configuration initiale
            if not await self.setup_real_exchange():
                raise Exception("Échec configuration exchange")
            if not await self.setup_real_telegram():
                raise Exception("Échec configuration Telegram")
            logger.info(
                f"""
╔═════════════════════════════════════════════════════════════╗
║                Trading Bot Ultimate v4 - REAL               ║
╠═════════════════════════════════════════════════════════════╣                                
║ Mode: REAL TRADING                                         ║
║ Status: RUNNING                                            ║
╚═════════════════════════════════════════════════════════════╝
                """
            )
            # Mise à jour de l'état du bot
            st.session_state.bot_running = True
        except Exception as telegram_error:
            logger.error(f"Erreur envoi Telegram: {telegram_error}")
        raise
    async def create_dashboard(self):
        """Crée le dashboard Streamlit"""
        try:
            # Récupération du portfolio
            portfolio = await self.get_real_portfolio()
            if not portfolio:
                st.error("Unable to fetch portfolio data")
                return
            # En-tête
            st.title("Trading Bot Ultimate v4 🤖")
            # Tabs pour organiser l'information
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Portfolio", "Trading", "Analysis", "Settings"]
            )
            # TAB 1: PORTFOLIO
            with tab1:
                # Métriques principales sur 4 colonnes
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total Value",
                        f"${portfolio['total_value']:,.2f}",
                        delta=f"{portfolio.get('daily_pnl', 0):+.2f}%",
                    )
                with col2:
                    st.metric("Available USDC", f"${portfolio['free']:,.2f}")
                with col3:
                    st.metric("Locked USDC", f"${portfolio['used']:,.2f}")
                with col4:
                    st.metric(
                        "Available Margin", f"${portfolio['available_margin']:,.2f}"
                    )
                # Positions actuelles
                st.subheader("📊 Active Positions")
                positions_df = pd.DataFrame(portfolio["positions"])
                if not positions_df.empty:
                    st.dataframe(positions_df, use_container_width=True)
            # TAB 2: TRADING (Signaux, Arbitrage, Ordres)
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    # Signaux de trading actifs
                    st.subheader("🎯 Trading Signals")
                    if self.indicators:
                        st.dataframe(
                            pd.DataFrame(self.indicators), use_container_width=True
                        )
                    # Opportunités d'Arbitrage
                    if (
                        "arbitrage_opportunities" in st.session_state
                        and st.session_state["arbitrage_opportunities"]
                    ):
                        st.subheader("⚡ Opportunités d'Arbitrage")
                        st.write(st.session_state["arbitrage_opportunities"])
                    # Bouton arbitrage manuel
                    if st.button("Scan Arbitrage"):
                        opps = await self.arbitrage_engine.find_opportunities()
                        if opps:
                            st.session_state["arbitrage_opportunities"] = opps
                            st.success("Arbitrage détecté !")
                with col2:
                    # Ordres en cours
                    st.subheader("📋 Open Orders")
                    if hasattr(self, "spot_client"):
                        orders = self.spot_client.get_open_orders("BTCUSDT")
                        if orders:
                            st.dataframe(pd.DataFrame(orders), use_container_width=True)
            # TAB 3: ANALYSIS (Indicateurs, Heatmap, News, Quantum, Regime)
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    # Indicateurs techniques
                    st.subheader("📉 Technical Analysis")
                    if hasattr(self, "advanced_indicators"):
                        st.dataframe(
                            pd.DataFrame(self.advanced_indicators.get_all_signals()),
                            use_container_width=True,
                        )
                    # Heatmap de liquidité
                    if (
                        "heatmap" in st.session_state
                        and st.session_state["heatmap"] is not None
                    ):
                        st.subheader("Liquidity Heatmap")
                        st.image(
                            st.session_state["heatmap"],
                            caption="Heatmap Carnet d'ordres BTC/USDT",
                        )
                with col2:
                    # News/Sentiment
                    if (
                        "news_score" in st.session_state
                        and st.session_state["news_score"]
                    ):
                        st.subheader("📰 Impact News")
                        st.write(st.session_state["news_score"])
                    if (
                        "important_news" in st.session_state
                        and st.session_state["important_news"]
                    ):
                        st.subheader("📰 News Importantes")
                        for news in st.session_state["important_news"]:
                            st.markdown(
                                f"- [{news['title']}]({news['url']}) ({news['sentiment']}, {news['confidence']:.2f})"
                            )
                    # Signal Quantum
                    if "quantum_signal" in st.session_state:
                        st.subheader("Quantum SVM Signal")
                        st.metric(
                            "Quantum SVM Signal", st.session_state["quantum_signal"]
                        )
                    # Regime de marché
                    if "regime" in st.session_state:
                        st.subheader("Régime de marché")
                        st.info(f"{st.session_state['regime']}")
            # TAB 4: SETTINGS
            with tab4:
                st.subheader("⚙️ Bot Configuration")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Trading Parameters")
                    risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 2.0)
                    max_positions = st.number_input("Max Open Positions", 1, 10, 3)
        except Exception as e:
            self.logger.error(f"Erreur création dashboard: {e}")
            st.error(f"Error creating dashboard: {str(e)}")
    def _build_decision(
        self, policy, value, technical_score, news_sentiment, regime, timestamp
    ):
        """Construit la décision finale basée sur tous les inputs"""
        try:
            # Conversion policy en numpy pour le traitement
            policy_np = policy.detach().numpy()
            # Ne garder que les actions d'achat (long only)
            buy_actions = np.maximum(policy_np, 0)
            # Calculer la confiance basée sur value et les scores
            confidence = float(
                np.mean(
                    [
                        float(value.detach().numpy()),
                        technical_score,
                        news_sentiment["score"],
                    ]
                )
            )
            # Trouver le meilleur actif à acheter
            best_pair_idx = np.argmax(buy_actions)
            # Construire la décision
            decision = {
                "action": (
                    "buy"
                    if confidence > config["AI"]["confidence_threshold"]
                    else "wait"
                ),
                "symbol": config["TRADING"]["pairs"][best_pair_idx],
                "confidence": confidence,
                "timestamp": timestamp,
                "regime": regime,
                "technical_score": technical_score,
                "news_impact": news_sentiment["sentiment"],
                "value_estimate": float(value.detach().numpy()),
                "position_size": buy_actions[best_pair_idx],
            }
            return decision
        except Exception as e:
            logger.error(f"[{timestamp}] Erreur construction décision: {e}")
            return None
    def _combine_features(self, technical_features, news_impact, regime):
        """Combine toutes les features pour le GTrXL"""
        try:
            # Conversion en tensors
            technical_tensor = technical_features["tensor"]
            news_tensor = torch.tensor(news_impact["embeddings"], dtype=torch.float32)
            regime_tensor = torch.tensor(
                self._encode_regime(regime), dtype=torch.float32
            )
            # Ajout de dimensions si nécessaire
            if news_tensor.dim() == 1:
                news_tensor = news_tensor.unsqueeze(0)
            if regime_tensor.dim() == 1:
                regime_tensor = regime_tensor.unsqueeze(0)
            # Combinaison
            features = torch.cat([technical_tensor, news_tensor, regime_tensor], dim=-1)
            return features
        except Exception as e:
            logger.error(f"Erreur: {e}")
            raise
    def _encode_regime(self, regime):
        """Encode le régime de marché en vecteur"""
        regime_mapping = {
            "High Volatility Bull": [1, 0, 0, 0, 0],
            "Low Volatility Bull": [0, 1, 0, 0, 0],
            "High Volatility Bear": [0, 0, 1, 0, 0],
            "Low Volatility Bear": [0, 0, 0, 1, 0],
            "Sideways": [0, 0, 0, 0, 1],
        }
        return regime_mapping.get(regime, [0, 0, 0, 0, 0])
    async def execute_trades(self, decision):
        """Exécution des trades selon la décision"""
        # Vérification du circuit breaker
        if await self.circuit_breaker.should_stop_trading():
            await self.telegram.send_message(
                "⚠️ Trading suspendu: Circuit breaker activé\n"
            )
            return
        if decision and decision["confidence"] > config["AI"]["confidence_threshold"]:
            try:
                # Vérification des opportunités d'arbitrage
                arb_ops = await self.arbitrage_engine.find_opportunities()
                if arb_ops:
                    await self.telegram.send_message(
                        f"💰 Opportunité d'arbitrage détectée:\n" f"Details: {arb_ops}"
                    )
                # Récupération du prix actuel
                current_price = await self.exchange.get_price(decision["symbol"])
                decision["entry_price"] = current_price
                # Calcul de la taille de position avec gestion du risque
                position_size = self.position_manager.calculate_position_size(
                    decision,
                    available_balance=await self.exchange.get_balance(
                        config["TRADING"]["base_currency"]
                    ),
                )
                # Vérification finale avant l'ordre
                if not self._validate_trade(decision, position_size):
                    return
                # Placement de l'ordre avec stop loss
                order = await self.exchange.create_order(
                    symbol=decision["symbol"],
                    type="limit",
                    side="buy",  # Achat uniquement comme demandé
                    amount=position_size,
                    price=decision["entry_price"],
                    params={
                        "stopLoss": {
                            "type": "trailing",
                            "activation_price": decision["trailing_stop"][
                                "activation_price"
                            ],
                            "callback_rate": decision["trailing_stop"]["callback_rate"],
                        },
                        "takeProfit": {"price": decision["take_profit"]},
                    },
                )
                # Notification Telegram détaillée
                await self.telegram.send_message(
                    f"📄 Ordre placé:\n"
                    f"Symbol: {order['symbol']}\n"
                    f"Type: {order['type']}\n"
                    f"Prix: {order['price']}\n"
                    f"Stop Loss: {decision['stop_loss']}\n"
                    f"Take Profit: {decision['take_profit']}\n"
                    f"Trailing Stop: {decision['trailing_stop']['activation_price']}\n"
                    f"Confiance: {decision['confidence']:.2%}\n"
                    f"Régime: {decision['regime']}\n"
                    f"News Impact: {decision['news_impact']}\n"
                    f"Volume: {position_size} {config['TRADING']['base_currency']}"
                )
                # Mise à jour du dashboard
                self.dashboard.update_trades(order)
            except Exception as e:
                logger.error(f"Erreur: {e}")
                await self.telegram.send_message(f"⚠️ Erreur d'exécution: {str(e)}\n")
    def _validate_trade(self, decision, position_size):
        """Validation finale avant l'exécution du trade"""
        try:
            # Vérification de la taille minimale
            if position_size < 0.001:  # Exemple de taille minimale
                return False
            # Vérification du spread
            if self._check_spread_too_high(decision["symbol"]):
                return False
            # Vérification de la liquidité
            if not self._check_sufficient_liquidity(decision["symbol"], position_size):
                return False
            # Vérification des news à haut risque
            if self._check_high_risk_news():
                return False
            # Vérification des limites de position
            if not self.position_manager.check_position_limits(position_size):
                return False
            # Vérification du timing d'entrée
            if not self._check_entry_timing(decision):
                return False
            return True
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return False
    def _check_spread_too_high(self, symbol):
        """Vérifie si le spread est trop important"""
        try:
            orderbook = self.buffer.get_orderbook(symbol)
            best_bid = orderbook["bids"][0][0]
            best_ask = orderbook["asks"][0][0]
            spread = (best_ask - best_bid) / best_bid
            return spread > 0.001  # 0.1% spread maximum
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return True  # Par sécurité
    def _check_sufficient_liquidity(self, symbol, position_size):
        """Vérifie s'il y a assez de liquidité pour le trade"""
        try:
            orderbook = self.buffer.get_orderbook(symbol)
            # Calcul de la profondeur de marché nécessaire
            required_liquidity = position_size * 3  # 3x la taille pour la sécurité
            # Somme de la liquidité disponible
            available_liquidity = sum(vol for _, vol in orderbook["bids"][:10])
            return available_liquidity >= required_liquidity
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return False
    def _check_entry_timing(self, decision):
        """Vérifie si le timing d'entrée est optimal"""
        try:
            # Vérification des signaux de momentum
            momentum_signals = self._analyze_momentum_signals()
            if momentum_signals["strength"] < 0.5:
                return False
            # Vérification de la volatilité
            volatility = self._analyze_volatility()
            if volatility["current"] > volatility["threshold"]:
                return False
            # Vérification du volume
            volume_analysis = self._analyze_volume_profile()
            if not volume_analysis["supports_entry"]:
                return False
            return True
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return False
    def _analyze_momentum_signals(self):
        """Analyse des signaux de momentum"""
        try:
            signals = {
                "rsi": self._calculate_rsi(self.buffer.get_latest()),
                "macd": self._calculate_macd(self.buffer.get_latest()),
                "stoch": self._calculate_stoch_rsi(self.buffer.get_latest()),
            }
            # Calcul de la force globale
            strengths = []
            if signals["rsi"]:
                strengths.append(abs(signals["rsi"]["strength"]))
            if signals["macd"]:
                strengths.append(abs(signals["macd"]["strength"]))
            if signals["stoch"]:
                strengths.append(abs(signals["stoch"]["strength"]))
            return {
                "signals": signals,
                "strength": np.mean(strengths) if strengths else 0,
            }
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return {"strength": 0, "signals": {}}
    def _analyze_volatility(self):
        """Analyse de la volatilité actuelle"""
        try:
            # Calcul des indicateurs de volatilité
            bbands = self._calculate_bbands(self.buffer.get_latest())
            atr = self._calculate_atr(self.buffer.get_latest())
            # Calcul de la volatilité normalisée
            current_volatility = 0
            if bbands and atr:
                bb_width = bbands["bandwidth"]
                atr_norm = atr["normalized"]
                current_volatility = (bb_width + atr_norm) / 2
            return {
                "current": current_volatility,
                "threshold": 0.8,  # Seuil dynamique basé sur le régime
                "indicators": {"bbands": bbands, "atr": atr},
            }
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return {"current": float("inf"), "threshold": 0.8, "indicators": {}}
    def _analyze_volume_profile(self):
        """Analyse du profil de volume"""
        try:
            volume_data = self.buffer.get_volume_profile()
            if not volume_data:
                return {"supports_entry": False}
            # Calcul des niveaux de support/résistance basés sur le volume
            poc_level = self._calculate_poc(volume_data)
            value_area = self._calculate_value_area(volume_data)
            # Analyse de la distribution du volume
            volume_distribution = {
                "above_poc": sum(v for p, v in volume_data.items() if p > poc_level),
                "below_poc": sum(v for p, v in volume_data.items() if p < poc_level),
            }
            # Calcul du ratio de support du volume
            current_price = self.buffer.get_latest_price()
            volume_support = (
                volume_distribution["above_poc"]
                / (volume_distribution["above_poc"] + volume_distribution["below_poc"])
                if current_price > poc_level
                else volume_distribution["below_poc"]
                / (volume_distribution["above_poc"] + volume_distribution["below_poc"])
            )
            return {
                "supports_entry": volume_support > 0.6,
                "poc": poc_level,
                "value_area": value_area,
                "distribution": volume_distribution,
            }
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return {"supports_entry": False}
    def _calculate_poc(self, volume_profile):
        """Calcul du Point of Control"""
        try:
            if not volume_profile:
                return None
            return max(volume_profile.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logger.error(f"Erreur calcul POC: {e}")
            return None
    def _calculate_value_area(self, volume_profile, value_area_pct=0.68):
        """Calcul de la Value Area"""
        try:
            if not volume_profile:
                return None
            # Trier les prix par volume décroissant
            sorted_prices = sorted(
                volume_profile.items(), key=lambda x: x[1], reverse=True
            )
            # Calculer le volume total
            total_volume = sum(volume_profile.values())
            target_volume = total_volume * value_area_pct
            cumulative_volume = 0
            value_area_prices = []
            # Construire la value area
            for price, volume in sorted_prices:
                cumulative_volume += volume
                value_area_prices.append(price)
                if cumulative_volume >= target_volume:
                    break
            return {"high": max(value_area_prices), "low": min(value_area_prices)}
        except Exception as e:
            logger.error(f"Erreur calcul Value Area: {e}")
            return None
    async def run(self):
        """Point d'entrée principal du bot"""
        try:
            # Configuration initiale
            await self.setup_streams()
            # Étude initiale du marché
            market_regime, historical_data, initial_analysis = await self.study_market()
            while True:
                try:
                    # Mise à jour des données
                    market_data = await self.get_latest_data()
                    if not market_data:
                        continue
                    # Analyse technique
                    signals = await self.analyze_signals(market_data)
                    # Analyse des news
                    news_impact = await self.news_analyzer.analyze()
                    # Construction des features
                    features = self._combine_features(
                        technical_features=signals,
                        news_impact=news_impact,
                        regime=market_regime,
                    )
                    # Obtention de la politique et valeur
                    policy, value = self.decision_model(features)
                    if policy is not None and value is not None:
                        # Construction de la décision
                        decision = self._build_decision(
                            policy=policy,
                            value=value,
                            technical_score=signals["recommendation"]["confidence"],
                            news_sentiment=news_impact,
                            regime=market_regime,
                            timestamp=pd.Timestamp.utcnow(),
                        )
                        # Ajout gestion des risques
                        decision = self._add_risk_management(decision)
                        # Exécution des trades
                        await self.execute_trades(decision)
                    # Attente avant la prochaine itération
                    await asyncio.sleep(config["TRADING"]["update_interval"])
                except Exception as loop_error:
                    logger.error(f"Erreur dans la boucle principale: {loop_error}")
                    continue
        except Exception as e:
            logger.error(f"Erreur fatale: {e}")
            await self.telegram.send_message(f"🚨 Erreur critique du bot:\n{str(e)}\n")
            raise
    def _should_train(self, historical_data):
        """Détermine si les modèles doivent être réentraînés"""
        try:
            # Vérification de la taille minimale des données
            if len(historical_data.get("1h", [])) < config["AI"]["min_training_size"]:
                return False
            # Vérification de la dernière session d'entraînement
            return True
            return time_since_training.days >= 1  # Réentraînement quotidien
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return False
    async def _train_models(self, historical_data, initial_analysis):
        """Entraîne ou met à jour les modèles"""
        try:
            # Préparation des données d'entraînement
            X_train, y_train = self._prepare_training_data(
                historical_data, initial_analysis
            )
            # Entraînement du modèle hybride
            self.hybrid_model.train(
                market_data=historical_data,
                indicators=initial_analysis,
                epochs=config["AI"]["n_epochs"],
                batch_size=config["AI"]["batch_size"],
                learning_rate=config["AI"]["learning_rate"],
            )
            # Entraînement du PPO-GTrXL
            self.models["ppo_gtrxl"].train(
                env=self.env,
                total_timesteps=100000,
                batch_size=config["AI"]["batch_size"],
                learning_rate=config["AI"]["learning_rate"],
                gradient_clip=config["AI"]["gradient_clip"],
            )
            # Entraînement du CNN-LSTM
            self.models["cnn_lstm"].train(
                X_train,
                y_train,
                epochs=config["AI"]["n_epochs"],
                batch_size=config["AI"]["batch_size"],
                validation_split=0.2,
            )
            # Mise à jour du timestamp d'entraînement
            # Sauvegarde des modèles
            self._save_models()
        except Exception as e:
            logger.error(f"Erreur: {e}")
            raise
    def _prepare_training_data(self, historical_data, initial_analysis):
        """Prépare les données pour l'entraînement"""
        try:
            features = []
            labels = []
            # Pour chaque timeframe
            for timeframe in config["TRADING"]["timeframes"]:
                tf_data = historical_data[timeframe]
                tf_analysis = initial_analysis[timeframe]
                # Extraction des features
                technical_features = self._extract_technical_features(tf_data)
                market_features = self._extract_market_features(tf_data)
                indicator_features = self._extract_indicator_features(tf_analysis)
                # Combinaison des features
                combined_features = np.concatenate(
                    [technical_features, market_features, indicator_features], axis=1
                )
                features.append(combined_features)
                # Création des labels (returns futurs)
                future_returns = self._calculate_future_returns(tf_data)
                labels.append(future_returns)
            # Fusion des données de différents timeframes
            X = np.concatenate(features, axis=1)
            y = np.mean(labels, axis=0)
            return X, y
        except Exception as e:
            logger.error(f"Erreur: {e}")
            raise
    def _extract_technical_features(self, data):
        """Extrait les features techniques des données"""
        try:
            features = []
            # Features de tendance
            trend_data = self._calculate_trend_features(data)
            if trend_data:
                features.append(trend_data)
            # Features de momentum
            if momentum_data := self._calculate_momentum_features(data):
                features.append(momentum_data)
            # Features de volatilité
            if volatility_data := self._calculate_volatility_features(data):
                features.append(volatility_data)
            # Features de volume
            if volume_data := self._calculate_volume_features(data):
                features.append(volume_data)
            # Features d'orderflow
            if orderflow_data := self._calculate_orderflow_features(data):
                features.append(orderflow_data)
            return np.concatenate(features, axis=1)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _extract_market_features(self, data):
        """Extrait les features de marché"""
        try:
            features = []
            # Prix relatifs
            close = data["close"].values
            features.append(close[1:] / close[:-1] - 1)  # Returns
            # Volumes relatifs
            volume = data["volume"].values
            features.append(volume[1:] / volume[:-1] - 1)  # Volume change
            # Spread
            features.append((data["high"] - data["low"]) / data["close"])
            # Gap analysis
            features.append(self._calculate_gap_features(data))
            # Liquidité
            features.append(self._calculate_liquidity_features(data))
            return np.column_stack(features)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _extract_indicator_features(self, analysis):
        """Extrait les features des indicateurs"""
        try:
            features = []
            # Features de tendance
            if "trend" in analysis:
                trend_strength = analysis["trend"].get("trend_strength", 0)
                features.append(trend_strength)
            # Features de volatilité
            if "volatility" in analysis:
                volatility = analysis["volatility"].get("current_volatility", 0)
                features.append(volatility)
            # Features de volume
            if "volume" in analysis:
                volume_profile = analysis["volume"].get("volume_profile", {})
                strength = float(volume_profile.get("strength", 0))
                features.append(strength)
            # Signal dominant
            if "dominant_signal" in analysis:
                signal_mapping = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
                signal = signal_mapping.get(analysis["dominant_signal"], 0)
                features.append(signal)
            return np.array(features)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _calculate_trend_features(self, data):
        """Calcule les features de tendance"""
        try:
            features = []
            # Supertrend
            if st_data := self._calculate_supertrend(data):
                features.append(st_data["value"])
                features.append(st_data["direction"])
                features.append(st_data["strength"])
            # Ichimoku
            if ichi_data := self._calculate_ichimoku(data):
                features.append(ichi_data["tenkan"] / data["close"])
                features.append(ichi_data["kijun"] / data["close"])
                features.append(ichi_data["senkou_a"] / data["close"])
                features.append(ichi_data["senkou_b"] / data["close"])
                features.append(ichi_data["cloud_strength"])
            # EMA Ribbon
            if ema_data := self._calculate_ema_ribbon(data):
                features.append(ema_data["trend"])
                features.append(ema_data["strength"])
                for ema in ema_data["emas"].values():
                    features.append(ema / data["close"])
            # Parabolic SAR
            if psar_data := self._calculate_psar(data):
                features.append(psar_data["value"] / data["close"])
                features.append(psar_data["trend"])
                features.append(psar_data["strength"])
            return np.column_stack(features)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _calculate_momentum_features(self, data):
        """Calcule les features de momentum"""
        try:
            features = []
            # RSI
            if rsi_data := self._calculate_rsi(data):
                features.append(rsi_data["value"])
                features.append(float(rsi_data["overbought"]))
                features.append(float(rsi_data["oversold"]))
                features.append(rsi_data["divergence"])
            # Stochastic RSI
            if stoch_data := self._calculate_stoch_rsi(data):
                features.append(stoch_data["k_line"])
                features.append(stoch_data["d_line"])
                features.append(float(stoch_data["overbought"]))
                features.append(float(stoch_data["oversold"]))
                features.append(stoch_data["crossover"])
            # MACD
            if macd_data := self._calculate_macd(data):
                features.append(macd_data["macd"])
                features.append(macd_data["signal"])
                features.append(macd_data["histogram"])
                features.append(macd_data["crossover"])
                features.append(macd_data["strength"])
            # Awesome Oscillator
            if ao_data := self._calculate_ao(data):
                features.append(ao_data["value"])
                features.append(ao_data["momentum_shift"])
                features.append(ao_data["strength"])
                features.append(float(ao_data["zero_cross"]))
            # TSI
            if tsi_data := self._calculate_tsi(data):
                features.append(tsi_data["tsi"])
                features.append(tsi_data["signal"])
                features.append(tsi_data["histogram"])
                features.append(tsi_data["divergence"])
            return np.column_stack(features)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _calculate_volatility_features(self, data):
        """Calcule les features de volatilité"""
        try:
            features = []
            # Bollinger Bands
            if bb_data := self._calculate_bbands(data):
                features.append((bb_data["upper"] - data["close"]) / data["close"])
                features.append((bb_data["middle"] - data["close"]) / data["close"])
                features.append((bb_data["lower"] - data["close"]) / data["close"])
                features.append(bb_data["bandwidth"])
                features.append(bb_data["percent_b"])
                features.append(float(bb_data["squeeze"]))
            # Keltner Channels
            if kc_data := self._calculate_keltner(data):
                features.append((kc_data["upper"] - data["close"]) / data["close"])
                features.append((kc_data["middle"] - data["close"]) / data["close"])
                features.append((kc_data["lower"] - data["close"]) / data["close"])
                features.append(kc_data["width"])
                features.append(kc_data["position"])
            # ATR
            if atr_data := self._calculate_atr(data):
                features.append(atr_data["value"])
                features.append(atr_data["normalized"])
                features.append(atr_data["trend"])
                features.append(atr_data["volatility_regime"])
            # VIX Fix
            if vix_data := self._calculate_vix_fix(data):
                features.append(vix_data["value"])
                features.append(vix_data["regime"])
                features.append(vix_data["trend"])
                features.append(vix_data["percentile"])
            return np.column_stack(features)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _calculate_gap_features(self, data):
        """Calcule les features de gaps"""
        try:
            features = []
            # Prix d'ouverture vs clôture précédente
            open_close_gap = (data["open"] - data["close"].shift(1)) / data[
                "close"
            ].shift(1)
            features.append(open_close_gap)
            # Gap haussier/baissier
            features.append(np.where(open_close_gap > 0, 1, -1))
            # Force du gap
            features.append(abs(open_close_gap))
            # Gap comblé
            gap_filled = (data["low"] <= data["close"].shift(1)) & (
                data["high"] >= data["open"]
            )
            features.append(gap_filled.astype(float))
            return np.column_stack(features)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _calculate_liquidity_features(self, data):
        """Calcule les features de liquidité"""
        try:
            features = []
            # Analyse du carnet d'ordres
            if orderbook := self.buffer.get_orderbook(data.name):
                # Déséquilibre bid/ask
                bid_volume = sum(vol for _, vol in orderbook["bids"][:10])
                ask_volume = sum(vol for _, vol in orderbook["asks"][:10])
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                features.append(imbalance)
                # Profondeur de marché
                depth = (bid_volume + ask_volume) / data["volume"].mean()
                features.append(depth)
                # Spread relatif
                spread = (
                    orderbook["asks"][0][0] - orderbook["bids"][0][0]
                ) / orderbook["bids"][0][0]
                features.append(spread)
                # Clusters de liquidité
                clusters = self._detect_liquidity_clusters(orderbook)
                features.append(len(clusters["bid_clusters"]))
                features.append(len(clusters["ask_clusters"]))
                # Score de résistance à l'impact
                impact_resistance = self._calculate_impact_resistance(orderbook)
                features.append(impact_resistance)
            # Métriques historiques
            # Volume moyen sur 24h
            vol_24h = data["volume"].rolling(window=1440).mean()  # 1440 minutes = 24h
            features.append(data["volume"] / vol_24h)
            # Ratio de liquidité de Amihud
            daily_returns = data["close"].pct_change()
            amihud = abs(daily_returns) / (data["volume"] * data["close"])
            features.append(amihud)
            # Ratio de turnover
            turnover = (
                data["volume"]
                * data["close"]
                / data["volume"].rolling(window=20).mean()
            )
            features.append(turnover)
            return np.column_stack(features)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _detect_liquidity_clusters(self, orderbook):
        """Détecte les clusters de liquidité dans le carnet d'ordres"""
        try:
            bid_clusters = []
            ask_clusters = []
            # Paramètres de clustering
            min_volume = 1.0  # Volume minimum pour un cluster
            price_threshold = 0.001  # Distance maximale entre prix pour un même cluster
            # Détection des clusters côté bid
            current_cluster = {"start_price": None, "total_volume": 0}
            for price, volume in orderbook["bids"]:
                if volume >= min_volume:
                    if current_cluster["start_price"] is None:
                        current_cluster = {"start_price": price, "total_volume": volume}
                    elif abs(price - current_cluster["start_price"]) <= price_threshold:
                        current_cluster["total_volume"] += volume
                    else:
                        if current_cluster["total_volume"] >= min_volume:
                            bid_clusters.append(current_cluster)
                        current_cluster = {"start_price": price, "total_volume": volume}
            # Détection des clusters côté ask
            current_cluster = {"start_price": None, "total_volume": 0}
            for price, volume in orderbook["asks"]:
                if volume >= min_volume:
                    if current_cluster["start_price"] is None:
                        current_cluster = {"start_price": price, "total_volume": volume}
                    elif abs(price - current_cluster["start_price"]) <= price_threshold:
                        current_cluster["total_volume"] += volume
                    else:
                        if current_cluster["total_volume"] >= min_volume:
                            ask_clusters.append(current_cluster)
                        current_cluster = {"start_price": price, "total_volume": volume}
            return {
                "bid_clusters": bid_clusters,
                "ask_clusters": ask_clusters,
            }
        except Exception as e:
            logger.error(f"Erreur: {e}")
    def _calculate_impact_resistance(self, orderbook, impact_size=1.0):
        """Calcule la résistance à l'impact de marché"""
        try:
            # Calcul de l'impact sur les bids
            cumulative_bid_volume = 0
            bid_impact = 0
            for price, volume in orderbook["bids"]:
                cumulative_bid_volume += volume
                if cumulative_bid_volume >= impact_size:
                    bid_impact = (orderbook["bids"][0][0] - price) / orderbook["bids"][
                        0
                    ][0]
                    break
            # Calcul de l'impact sur les asks
            cumulative_ask_volume = 0
            ask_impact = 0
            for price, volume in orderbook["asks"]:
                cumulative_ask_volume += volume
                if cumulative_ask_volume >= impact_size:
                    ask_impact = (price - orderbook["asks"][0][0]) / orderbook["asks"][
                        0
                    ][0]
                    break
            # Score de résistance
            resistance_score = (
                1 / (bid_impact + ask_impact)
                if (bid_impact + ask_impact) > 0
                else float("inf")
            )
            return resistance_score
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return
    def _calculate_future_returns(self, data, horizons=[1, 5, 10, 20]):
        """Calcule les returns futurs pour différents horizons"""
        try:
            returns = []
            for horizon in horizons:
                # Calcul du return futur
                future_return = data["close"].shift(-horizon) / data["close"] - 1
                returns.append(future_return)
                # Calcul de la volatilité future
                future_volatility = (
                    data["close"].rolling(window=horizon).std().shift(-horizon)
                )
                returns.append(future_volatility)
                # Calcul du volume futur normalisé
                future_volume = (
                    (data["volume"].shift(-horizon) / data["volume"])
                    .rolling(window=horizon)
                    .mean()
                )
                returns.append(future_volume)
            return np.column_stack(returns)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _save_models(self):
        """Sauvegarde les modèles entraînés"""
        try:
            # Création du dossier de sauvegarde
            save_dir = os.path.join(current_dir, "models")
            os.makedirs(save_dir, exist_ok=True)
            # Sauvegarde du modèle hybride
            hybrid_path = os.path.join(save_dir, "hybrid_model.pt")
            torch.save(self.hybrid_model.state_dict(), hybrid_path)
            # Sauvegarde du PPO-GTrXL
            ppo_path = os.path.join(save_dir, "ppo_gtrxl.pt")
            torch.save(self.models["ppo_gtrxl"].state_dict(), ppo_path)
            # Sauvegarde du CNN-LSTM
            cnn_lstm_path = os.path.join(save_dir, "cnn_lstm.pt")
            torch.save(self.models["cnn_lstm"].state_dict(), cnn_lstm_path)
            # Sauvegarde des métadonnées
            metadata = {
                "model_versions": {
                    "hybrid": self.hybrid_model.version,
                    "ppo_gtrxl": self.models["ppo_gtrxl"].version,
                    "cnn_lstm": self.models["cnn_lstm"].version,
                },
                "training_metrics": self._get_training_metrics(),
            }
            metadata_path = os.path.join(save_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Erreur: {e}")
            raise
    def _get_training_metrics(self):
        """Récupère les métriques d'entraînement"""
        try:
            metrics = {
                "hybrid_model": {
                    "loss": self.hybrid_model.training_history["loss"],
                    "val_loss": self.hybrid_model.training_history["val_loss"],
                    "accuracy": self.hybrid_model.training_history["accuracy"],
                },
                "ppo_gtrxl": {
                    "policy_loss": self.models["ppo_gtrxl"].training_info[
                        "policy_loss"
                    ],
                    "value_loss": self.models["ppo_gtrxl"].training_info["value_loss"],
                    "entropy": self.models["ppo_gtrxl"].training_info["entropy"],
                },
                "cnn_lstm": {
                    "loss": self.models["cnn_lstm"].history["loss"],
                    "val_loss": self.models["cnn_lstm"].history["val_loss"],
                    "mae": self.models["cnn_lstm"].history["mae"],
                },
            }
            return metrics
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return {}
    async def _should_stop_trading(self):
        """Vérifie les conditions d'arrêt du trading"""
        try:
            # Vérification du circuit breaker
            if await self.circuit_breaker.should_stop_trading():
                return True
            # Vérification du drawdown maximum
            current_drawdown = self.position_manager.calculate_drawdown()
            if current_drawdown > config["RISK"]["max_drawdown"]:
                return True
            # Vérification de la perte journalière
            daily_loss = self.position_manager.calculate_daily_loss()
            if daily_loss > config["RISK"]["daily_stop_loss"]:
                return True
            # Vérification des conditions de marché
            market_conditions = await self._check_market_conditions()
            if not market_conditions["safe_to_trade"]:
                return True
            return False
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return True  # Par sécurité
    async def _check_market_conditions(self):
        """Vérifie les conditions de marché"""
        try:
            conditions = {"safe_to_trade": True, "reason": None}
            # Vérification de la volatilité
            volatility = self._analyze_volatility()
            if volatility["current"] > volatility["threshold"] * 2:
                conditions["safe_to_trade"] = False
                conditions["reason"] = "Volatilité excessive"
                return conditions
            # Vérification de la liquidité
            liquidity = await self._analyze_market_liquidity()
            if liquidity["status"] == "insufficient":
                conditions["safe_to_trade"] = False
                conditions["reason"] = "Liquidité insuffisante"
                return conditions
            # Vérification des news à haut risque
            if await self._check_high_risk_news():
                conditions["safe_to_trade"] = False
                conditions["reason"] = "News à haut risque"
                return conditions
            # Vérification des conditions techniques
            technical_check = self._check_technical_conditions()
            if not technical_check["safe"]:
                conditions["safe_to_trade"] = False
                conditions["reason"] = technical_check["reason"]
                return conditions
            return conditions
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return {"safe_to_trade": False, "reason": "Erreur système"}
    async def _analyze_market_liquidity(self):
        """Analyse détaillée de la liquidité du marché"""
        try:
            liquidity_status = {
                "status": "sufficient",
                "metrics": {},
            }
            # Analyse du carnet d'ordres
            for pair in config["TRADING"]["pairs"]:
                orderbook = self.buffer.get_orderbook(pair)
                if orderbook:
                    # Profondeur de marché
                    depth = self._calculate_market_depth(orderbook)
                    # Ratio bid/ask
                    bid_ask_ratio = self._calculate_bid_ask_ratio(orderbook)
                    # Spread moyen
                    avg_spread = self._calculate_average_spread(orderbook)
                    # Résistance à l'impact
                    impact_resistance = self._calculate_impact_resistance(orderbook)
                    liquidity_status["metrics"][pair] = {
                        "depth": depth,
                        "bid_ask_ratio": bid_ask_ratio,
                        "avg_spread": avg_spread,
                        "impact_resistance": impact_resistance,
                    }
                    # Vérification des seuils
                    if (
                        depth < 100000  # Exemple de seuil
                        or abs(1 - bid_ask_ratio) > 0.2
                        or avg_spread > 0.001
                        or impact_resistance < 0.5
                    ):
                        liquidity_status["status"] = "insufficient"
            return liquidity_status
        except Exception as e:
            logger.error(f"Erreur analyse liquidité: {e}")
            return {"status": "insufficient", "metrics": {}}
    def _check_technical_conditions(self):
        """Vérifie les conditions techniques du marché"""
        try:
            conditions = {"safe": True, "reason": None, "details": {}}
            for pair in config["TRADING"]["pairs"]:
                pair_data = self.buffer.get_latest_ohlcv(pair)
                # Vérification des divergences
                divergences = self._check_divergences(pair_data)
                if divergences["critical"]:
                    conditions["safe"] = False
                    conditions["reason"] = f"Divergence critique sur {pair}"
                    conditions["details"][pair] = divergences
                    return conditions
                # Vérification des patterns critiques
                patterns = self._check_critical_patterns(pair_data)
                if patterns["detected"]:
                    conditions["safe"] = False
                    conditions["reason"] = (
                        f"Pattern critique sur {pair}: {patterns['pattern']}"
                    )
                    conditions["details"][pair] = patterns
                    return conditions
                # Vérification des niveaux clés
                levels = self._check_key_levels(pair_data)
                if levels["breach"]:
                    conditions["safe"] = False
                    conditions["reason"] = f"Rupture niveau clé sur {pair}"
                    conditions["details"][pair] = levels
                    return conditions
                conditions["details"][pair] = {
                    "divergences": divergences,
                    "patterns": patterns,
                    "levels": levels,
                }
            return conditions
        except Exception as e:
            logger.error(f"Erreur: {e}")
            return {"safe": False, "reason": "Erreur système", "details": {}}
    def _check_divergences(self, data):
        """Détecte les divergences entre prix et indicateurs"""
        try:
            divergences = {
                "critical": False,
                "types": [],
            }
            # RSI Divergence
            rsi = self._calculate_rsi(data)
            if rsi:
                price_peaks = self._find_peaks(data["close"])
                rsi_peaks = self._find_peaks(rsi["value"])
                if self._is_bearish_divergence(price_peaks, rsi_peaks):
                    divergences["critical"] = True
                    divergences["types"].append("RSI_BEARISH")
                if self._is_bullish_divergence(price_peaks, rsi_peaks):
                    divergences["types"].append("RSI_BULLISH")
            # MACD Divergence
            macd = self._calculate_macd(data)
            if macd:
                price_peaks = self._find_peaks(data["close"])
                macd_peaks = self._find_peaks(macd["histogram"])
                if self._is_bearish_divergence(price_peaks, macd_peaks):
                    divergences["critical"] = True
                    divergences["types"].append("MACD_BEARISH")
                if self._is_bullish_divergence(price_peaks, macd_peaks):
                    divergences["types"].append("MACD_BULLISH")
            return divergences
        except Exception as e:
            logger.error(f"Erreur: {e}")
    def _check_critical_patterns(self, data):
        """Détecte les patterns techniques critiques"""
        try:
            patterns = {
                "detected": False,
                "pattern": None,
                "confidence": 0,
            }
            # Head and Shoulders
            if self._detect_head_shoulders(data):
                patterns["detected"] = True
                patterns["pattern"] = "HEAD_AND_SHOULDERS"
                patterns["confidence"] = 0.85
                return patterns
            # Double Top/Bottom
            if self._detect_double_pattern(data):
                patterns["detected"] = True
                patterns["pattern"] = (
                    "DOUBLE_TOP"
                    if data["close"].iloc[-1] < data["close"].mean()
                    else "DOUBLE_BOTTOM"
                )
                patterns["confidence"] = 0.80
                return patterns
            # Rising/Falling Wedge
            if self._detect_wedge(data):
                patterns["detected"] = True
                patterns["pattern"] = (
                    "RISING_WEDGE"
                    if data["close"].iloc[-1] > data["close"].mean()
                    else "FALLING_WEDGE"
                )
                patterns["confidence"] = 0.75
                return patterns
            return patterns
        except Exception as e:
            logger.error(f"Erreur: {e}")
    async def run_adaptive_trading(self, period="7d"):
        """
        Boucle principale adaptative : étude du marché, stratégie, trading auto.
        """
        # 1. Étudier le marché sur la période définie (ex: 7j)
        regime, historical_data, indicators_analysis = await self.study_market(
            period=period
        )
        # 2. Établir un plan/stratégie selon le régime détecté
        strategy = self.choose_strategy(regime, indicators_analysis)
        await self.telegram.send_message(
            f"📊 Plan établi : {strategy} | Régime détecté : {regime}"
        )
        self.current_regime = regime
        self.current_strategy = strategy
        while st.session_state.get("bot_running", True):
            # 3. Mise à jour continue du marché
            market_data = await self.get_latest_data()
            signals = await self.analyze_signals(market_data)
            news = (
                await self.news_analyzer.analyze()
                if hasattr(self, "news_analyzer")
                else None
            )
            arbitrage_opps = (
                await self.arbitrage_engine.find_opportunities()
                if hasattr(self, "arbitrage_engine")
                else None
            )
            new_regime = (
                self.regime_detector.predict(signals)
                if hasattr(self, "regime_detector")
                else self.current_regime
            )
            # 4. Adaptation : news, arbitrage, changement de régime
            if news and news.get("impact", 0) > 0.7:
                await self.telegram.send_message(f"📰 News critique détectée : {news}")
                self.current_strategy = "Defensive/No Trade"
            elif arbitrage_opps:
                await self.telegram.send_message(
                    f"⚡ Arbitrage détecté : {arbitrage_opps}"
                )
                self.current_strategy = "Arbitrage"
            elif new_regime != self.current_regime:
                self.current_regime = new_regime
                self.current_strategy = self.choose_strategy(new_regime, signals)
                await self.telegram.send_message(
                    f"🔄 Changement de régime : {new_regime} ⇒ Nouvelle stratégie : {self.current_strategy}"
                )
            # 5. Prendre position selon la stratégie courante
            decision = self.make_trade_decision(
                signals, self.current_strategy, news, arbitrage_opps
            )
            if decision and decision.get("action") in ["buy", "sell"]:
                order = await self.execute_real_trade(decision)
                await self.telegram.send_message(f"✅ Trade exécuté : {decision}")
            await asyncio.sleep(2)  # ajustable selon besoins
    def choose_strategy(self, regime, indicators):
        # Logique simple d'exemple : personnalise selon tes besoins
        if "Bull" in regime:
            return "Trend Following"
        elif "Bear" in regime:
            return "Short/Defensive"
        elif "Arbitrage" in regime:
            return "Arbitrage"
        else:
            return "Range/Scalping"
    def make_trade_decision(self, signals, strategy, news, arbitrage_opps):
        # Logique simple d'exemple : personnalise selon tes besoins
        if strategy == "Arbitrage" and arbitrage_opps:
            # Place un trade d'arbitrage (implémente selon ta structure)
            return {"action": "arbitrage", "details": arbitrage_opps}
        if signals and signals.get("recommendation", {}).get("action") in [
            "buy",
            "sell",
        ]:
            return signals["recommendation"]
        return None
    def _calculate_supertrend(self, data):
        try:
            # Log de début de calcul
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           CALCULATING SUPERTREND                 ║
╠═════════════════════════════════════════════════╣
╚═════════════════════════════════════════════════╝
            """
            )
            # Vérification de la configuration
            if not (
                self.config.get("INDICATORS", {}).get("trend", {}).get("supertrend", {})
            ):
                logger.warning("Missing Supertrend configuration")
                self.dashboard.update_indicator_status(
                    "Supertrend", "DISABLED - Missing config"
                )
                return None
            # Récupération des paramètres
            try:
                period = self.config["INDICATORS"]["trend"]["supertrend"]["period"]
                multiplier = self.config["INDICATORS"]["trend"]["supertrend"][
                    "multiplier"
                ]
                logger.info(
                    f"Using parameters: period={period}, multiplier={multiplier}"
                )
            except KeyError as ke:
                logger.error(f"Missing parameter: {ke}")
                self.dashboard.update_indicator_status(
                    "Supertrend", "DISABLED - Missing parameters"
                )
                return None
            # Vérification des données d'entrée
            if data is None or data.empty:
                logger.error("No input data provided")
                self.dashboard.update_indicator_status("Supertrend", "ERROR - No data")
                return None
            required_columns = ["high", "low", "close"]
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns: {required_columns}")
                self.dashboard.update_indicator_status(
                    "Supertrend", "ERROR - Missing columns"
                )
                return None
            # Extraction des séries de prix
            high = data["high"]
            low = data["low"]
            close = data["close"]
            # Calcul du True Range (TR)
            tr = pd.DataFrame()
            tr["h-l"] = high - low
            tr["h-pc"] = abs(high - close.shift(1))
            tr["l-pc"] = abs(low - close.shift(1))
            tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
            # Calcul de l'ATR
            atr = tr["tr"].rolling(window=period, min_periods=1).mean()
            # Calcul des bandes
            hl2 = (high + low) / 2
            final_upperband = hl2 + (multiplier * atr)
            final_lowerband = hl2 - (multiplier * atr)
            # Initialisation des séries Supertrend
            supertrend = pd.Series(index=data.index, dtype=float)
            direction = pd.Series(index=data.index, dtype=float)
            # Calcul du Supertrend
            for i in range(period, len(data)):
                try:
                    if close[i] > final_upperband[i - 1]:
                        supertrend[i] = final_lowerband[i]
                        direction[i] = 1
                    elif close[i] < final_lowerband[i - 1]:
                        supertrend[i] = final_upperband[i]
                        direction[i] = -1
                    else:
                        supertrend[i] = supertrend[i - 1]
                        direction[i] = direction[i - 1]
                except IndexError as idx_error:
                    logger.error(f"Index error at position {i}: {idx_error}")
                    continue
            # Calcul de la force du signal
            strength = abs(close - supertrend) / close
            # Mise à jour du statut
            self.dashboard.update_indicator_status("Supertrend", "ACTIVE")
            # Log de succès
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           SUPERTREND CALCULATED                  ║
╠═════════════════════════════════════════════════╣
║ Status: Success
║ Direction: {'Bullish' if direction.iloc[-1] == 1 else 'Bearish'}
║ Strength: {strength.iloc[-1]:.4f}
╚═════════════════════════════════════════════════╝
            """
            )
            return {
                "value": supertrend,
                "direction": direction,
                "strength": strength,
                "parameters": {"period": period, "multiplier": multiplier},
                "metadata": {
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "status": "SUCCESS",
                },
            }
        except Exception as e:
            # Log d'erreur détaillé
            logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║           SUPERTREND ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
            """
            )
            # Mise à jour du statut dans le dashboard
            self.dashboard.update_indicator_status(
                "Supertrend", f"ERROR - {type(e).__name__}"
            )
            return None
        finally:
            # Nettoyage et libération des ressources si nécessaire
            try:
                del tr
            except:
                pass
async def run_trading_bot():
    """Point d'entrée synchrone pour le bot de trading"""
    try:
        # Stats en temps réel
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Portfolio Value", f"{portfolio_value:.2f} USDC", f"{pnl:+.2f} USDC"
            )
        with col2:
            st.metric("Active Positions", "2", "Open")
        with col3:
            st.metric("24h P&L", "+123 USDC", "+1.23%")
        # Bouton de démarrage
        if st.button("Start Trading Bot", type="primary"):
            try:
                # Récupère (ou crée) une seule instance du bot si nécessaire
                bot = get_bot()
                # On lance la tâche de trading adaptatif si elle n'existe pas déjà
                if (
                    "trading_task" not in st.session_state
                    or st.session_state.trading_task is None
                    or st.session_state.trading_task.done()
                ):
                    loop = st.session_state.loop or asyncio.get_event_loop()
                    st.session_state.trading_task = loop.create_task(
                        bot.run_adaptive_trading(period="7d")
                    )
                    st.session_state.bot_running = True
                    st.success("🚀 Trading adaptatif lancé.")
                else:
                    st.info("Le bot est déjà en cours d’exécution.")
            except Exception as e:
                logger.error(f"Trading bot runtime error: {e}")
                st.error(f"❌ Runtime error: {str(e)}")
            finally:
                # Nettoyage des ressources si le bot a crashé
                if "bot" in locals() and hasattr(bot, "_cleanup"):
                    try:
                        cleanup_coro = bot._cleanup()
                        if asyncio.iscoroutine(cleanup_coro):
                            loop = st.session_state.loop or asyncio.get_event_loop()
                            loop.run_until_complete(cleanup_coro)
                    except Exception as cleanup_error:
                        logger.error(f"Cleanup error: {cleanup_error}")
    except Exception as e:
        logger.error(f"Trading bot error: {e}")
        st.error(f"❌ Trading bot error: {str(e)}")
async def main_async():
    """Point d'entrée principal de l'application"""
    try:
        # 1. Configuration de l'event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        nest_asyncio.apply()
        st.session_state.loop = loop
        # 2. Vérification de session unique - AJOUTEZ CETTE PARTIE
        if not st.session_state.get("session_initialized"):
            st.session_state.initialization_time = current_time
            st.session_state.session_initialized = True
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           NEW SESSION INITIALIZED                ║
╠═════════════════════════════════════════════════╣
║ Session ID: {st.session_state.session_id}
╚═════════════════════════════════════════════════╝
            """
            )
        else:
            # Utiliser la session existante
            current_time = st.session_state.get("last_update_time", current_time)
            st.session_state.last_update_time = current_time
        # 3. Protection de session
        session_manager.protect_session()
        # 4. Interface principale
        st.title("Trading Bot Ultimate v4 🤖")
        # 3. Session state avec vérification des initialisations multiples
        if not st.session_state.get("initialized", False):
            st.session_state.initialization_time = current_time
            st.session_state.initialized = True
            default_session_state = {
                "portfolio": None,
                "latest_data": None,
                "indicators": None,
                "bot_running": False,
                "refresh_count": 0,
                "ws_status": "disconnected",
                "ws_initialized": False,
                "ws_connection_status": "disconnected",
                "last_update_time": current_time,
                "needs_update": False,
                "update_interval": 2.0,
                "last_refresh": time.time(),
            }
            for key, value in default_session_state.items():
                if key not in st.session_state:
                    st.session_state[key] = value
        # 4. Initialisation du bot (une seule fois)
        if "bot_instance" not in st.session_state:
            bot = get_bot()
            if bot is None:
                st.error("❌ Failed to initialize bot")
                return
            st.session_state.bot_instance = bot
        else:
            bot = st.session_state.bot_instance
        # --- DEBUG données disponibles ---
        st.sidebar.markdown("#### Données présentes dans bot.latest_data :")
        # --- CORRIGE ici pour toujours refléter l'état session_state ---
        latest_data = st.session_state.get("latest_data", {})
        if not isinstance(latest_data, dict):
            latest_data = {}
        st.sidebar.write(
            {k: getattr(v, "shape", str(type(v))) for k, v in latest_data.items()}
            if latest_data
            else "Aucune donnée"
        )
        # 5. Interface principale - État et contrôles
        status_col1, status_col2 = st.columns([2, 1])
        with status_col1:
            ws_status = st.session_state.get("ws_connection_status", "disconnected")
            ws_icon = {
                "connected": "🟢",
                "disconnected": "🔴",
                "initializing": "🔄",
                "error": "⚠️",
            }.get(ws_status, "🔴")
            status_info = f"""
            ### Bot Status
            - 🚦 Trading: {'🟢 Active' if st.session_state.bot_running else '🔴 Stopped'}
            - 📡 WebSocket: {ws_icon} {ws_status.title()}
            - 💼 Portfolio: {'✅ Available' if st.session_state.portfolio else '⚠️ Not Available'}
            - ⏰ Last Update: {st.session_state.last_update_time}
            """
            st.info(status_info)
        # 6. Contrôles de la barre latérale avec gestion améliorée
        with st.sidebar:
            st.header("🛠️ Bot Controls")
            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value="Low",
                key=f"risk_level_slider_{st.session_state.session_id}",
            )
            st.divider()
            if not st.session_state.get("bot_running", False):
                if st.button(
                    "🟢 Start Trading", key="start_button", use_container_width=True
                ):
                    st.session_state.bot_running = True
                    # Protection pour éviter plusieurs tâches concurrentes
                    if not st.session_state.get("trading_task"):
                        loop = st.session_state.loop or asyncio.get_event_loop()
                        st.session_state.trading_task = loop.create_task(
                            bot.run_adaptive_trading(period="7d")
                        )
                    st.success(
                        "Trading adaptatif lancé (étude marché + stratégie auto)."
                    )
            else:
                if st.button(
                    "🔴 Stop Trading", key="stop_button", use_container_width=True
                ):
                    st.session_state.bot_running = False
                    # Arrêt propre de la tâche si elle existe
                    if st.session_state.get("trading_task"):
                        st.session_state.trading_task.cancel()
                        st.session_state.trading_task = None
                    st.warning("Trading stoppé.")
            # --- GESTION DES DONNEES ET BACKTEST ---
            # TOUJOURS lire depuis session_state !
            latest_data = st.session_state.get("latest_data")
            if not isinstance(latest_data, dict):
                latest_data = {}
            st.write("DEBUG - latest_data:", latest_data)  # <-- À enlever ensuite
            def _has_valid_ohlcv(item):
                return (
                    isinstance(item, dict)
                    and "ohlcv" in item
                    and isinstance(item["ohlcv"], list)
                    and len(item["ohlcv"]) > 0
                    and isinstance(item["ohlcv"][0], dict)
                    and all(
                        k in item["ohlcv"][0]
                        for k in ["timestamp", "open", "high", "low", "close", "volume"]
                    )
                )
            data_ready = any(_has_valid_ohlcv(item) for item in latest_data.values())
            if not data_ready:
                st.warning(
                    "Aucune donnée OHLCV disponible. Clique sur le bouton ci-dessous pour charger les données de marché."
                )
                if st.button("Charger les données", key="load_data_btn"):
                    with st.spinner("Chargement des données..."):
                        loaded = False
                        try:
                            if not hasattr(bot, "binance_ws") or bot.binance_ws is None:
                                st.info("Initialisation de la WebSocket…")
                                await bot.initialize()
                            if hasattr(bot, "get_latest_data"):
                                data = await bot.get_latest_data()
                                st.write("DEBUG - Résultat get_latest_data:", data)
                                if data and isinstance(data, dict) and len(data) > 0:
                                    st.session_state["latest_data"] = (
                                        data  # <-- SYNC dans la session
                                    )
                                    loaded = True
                                else:
                                    st.error(
                                        "La récupération a retourné None ou un dict vide : pas de données."
                                    )
                            elif hasattr(bot, "load_all_data"):
                                await bot.load_all_data()
                                latest_data = getattr(bot, "latest_data", {}) or {}
                                if not isinstance(latest_data, dict):
                                    latest_data = {}
                                st.write(
                                    "DEBUG - latest_data après load_all_data:",
                                    latest_data,
                                )
                                loaded = (
                                    isinstance(latest_data, dict)
                                    and len(latest_data) > 0
                                )
                                if loaded:
                                    st.session_state["latest_data"] = latest_data
                                else:
                                    st.error(
                                        "La récupération a retourné None ou un dict vide : pas de données."
                                    )
                            else:
                                st.error(
                                    "Aucune méthode de chargement trouvée sur le bot."
                                )
                        except Exception as exc:
                            st.error(f"Erreur lors du chargement des données : {exc}")
                        if loaded:
                            st.success("Données chargées ! Tu peux lancer un backtest.")
                            st.rerun()
            else:
                # --- BACKTEST CLASSIQUE ---
                if st.button("Lancer Backtest", key="backtest_all_btn"):
                    results = {}
                    st.info("Backtest en cours sur toutes les paires...")
                    try:
                        for symbol, data in latest_data.items():
                            try:
                                if _has_valid_ohlcv(data):
                                    import pandas as pd
                                    df = pd.DataFrame(data["ohlcv"])
                                    def strategy_func(df, **params):
                                        return (
                                            df["close"] > df["close"].rolling(5).mean()
                                        ).astype(int)
                                    engine = BacktestEngine(initial_capital=10000)
                                    results[symbol] = engine.run_backtest(
                                        df, strategy_func
                                    )
                                else:
                                    st.warning(
                                        f"Aucune donnée OHLCV exploitable pour {symbol}"
                                    )
                            except Exception as pair_exc:
                                st.warning(f"Erreur sur {symbol}: {pair_exc}")
                        st.session_state["all_backtest_results"] = results
                        st.success("Backtest terminé ✅")
                    except Exception as batch_exc:
                        st.error(f"Erreur lors du backtest: {batch_exc}")
                # Résultats
                if st.session_state.get("all_backtest_results"):
                    st.markdown("**Résultats Backtest Classique :**")
                    for symbol, res in st.session_state["all_backtest_results"].items():
                        st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")
                    if st.button(
                        "Lancer Backtest Quantique", key="quantum_backtest_all_btn"
                    ):
                        st.info("Backtest quantique en cours sur toutes les paires...")
                        results = {}
                        if not isinstance(latest_data, dict):
                            latest_data = {}
                        st.write(
                            "DEBUG - Paire/Data dispo :",
                            {
                                k: getattr(v, "shape", str(type(v)))
                                for k, v in latest_data.items()
                            },
                        )
                        try:
                            for symbol, data in latest_data.items():
                                st.write(f"Test {symbol} ...")
                                try:
                                    if _has_valid_ohlcv(data):
                                        import pandas as pd
                                        df = pd.DataFrame(data["ohlcv"])
                                        def strategy_func(df, **params):
                                            return (
                                                df["close"]
                                                > df["close"].rolling(5).mean()
                                            ).astype(int)
                                        engine = BacktestEngine(initial_capital=10000)
                                        results[symbol] = engine.run_backtest(
                                            df, strategy_func
                                        )
                                    else:
                                        st.warning(
                                            f"Aucune donnée OHLCV exploitable pour {symbol}"
                                        )
                                except Exception as pair_exc:
                                    st.warning(
                                        f"Erreur quantique sur {symbol}: {pair_exc}"
                                    )
                            st.session_state["all_quantum_results"] = results
                            st.success("Backtest quantique terminé ✅")
                            st.write("DEBUG - Résultats quantum :", results)
                        except Exception as batch_exc:
                            st.error(f"Erreur lors du backtest quantique: {batch_exc}")
                # Affichage des résultats
                if st.session_state.get("all_backtest_results"):
                    st.markdown("**Résultats Backtest Classique :**")
                    for symbol, res in st.session_state["all_backtest_results"].items():
                        st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")
                if st.session_state.get("all_quantum_results"):
                    st.markdown("**Résultats Backtest Quantique :**")
                    for symbol, res in st.session_state["all_quantum_results"].items():
                        st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")
        # 8. Onglets principaux avec gestion d'erreur
        try:
            portfolio_tab, trading_tab, analysis_tab = st.tabs(
                ["📈 Portfolio", "🎯 Trading", "📊 Analysis"]
            )
            # Onglet Portfolio
            with portfolio_tab:
                await _render_portfolio_tab(bot)
            # Onglet Trading
            with trading_tab:
                await _render_trading_tab(bot)
            # Onglet Analysis
            with analysis_tab:
                await _render_analysis_tab(bot)
        except Exception as tab_error:
            logger.error(f"Tab rendering error: {tab_error}")
            st.error("Error rendering tabs")
    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}")
        st.error(f"❌ Application error: {str(e)}")
    finally:
        # Protection finale sans nettoyage si le bot est en cours
        if not st.session_state.get("bot_running", False):
            try:
                session_manager.protect_session()
                    "%Y-%m-%d %H:%M:%S"
                )
            except Exception as protect_error:
                logger.error(f"Session protection error: {protect_error}")
# Fonctions auxiliaires pour le rendu des onglets
async def _render_portfolio_tab(bot):
    """Rendu de l'onglet Portfolio"""
    if st.session_state.bot_running:
        try:
            portfolio = st.session_state.get("portfolio")
            if portfolio:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "💰 Total Value",
                        f"{portfolio.get('total_value', 0):.2f} USDC",
                        f"{portfolio.get('daily_pnl', 0):+.2f} USDC",
                    )
                with col2:
                    st.metric(
                        "📈 24h Volume",
                        f"{portfolio.get('volume_24h', 0):.2f} USDC",
                        f"{portfolio.get('volume_change', 0):+.2f}%",
                    )
                with col3:
                    positions = portfolio.get("positions", [])
                    st.metric(
                        "🔄 Active Positions",
                        str(len(positions)),
                        f"{len(positions)} active",
                    )
                if positions:
                    st.subheader("Active Positions")
                    st.dataframe(pd.DataFrame(positions), use_container_width=True)
                else:
                    st.info("💡 No active positions")
            else:
                st.warning("⚠️ Waiting for portfolio data...")
        except Exception as e:
            st.error(f"❌ Portfolio error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view portfolio")
async def _render_trading_tab(bot):
    """Rendu de l'onglet Trading"""
    if st.session_state.bot_running:
        try:
            latest_data = bot.latest_data.get("BTCUSDT", {})
            if latest_data:
                col1, col2 = st.columns(2)
                with col1:
                    current_price = latest_data[-1]["close"]
                    prev_price = (
                        latest_data[-2]["close"]
                        if len(latest_data) > 1
                        else current_price
                    )
                    price_change = (
                        ((current_price - prev_price) / prev_price * 100)
                        if prev_price
                        else 0
                    )
                    st.metric(
                        "BTC/USDC Price",
                        f"{current_price:.2f}",
                        f"{price_change:+.2f}%",
                    )
                with col2:
                    current_vol = latest_data[-1]["volume"]
                    prev_vol = (
                        latest_data[-2]["volume"]
                        if len(latest_data) > 1
                        else current_vol
                    )
                    vol_change = (
                        ((current_vol - prev_vol) / prev_vol * 100) if prev_vol else 0
                    )
                    st.metric(
                        "Trading Volume", f"{current_vol:.2f}", f"{vol_change:+.2f}%"
                    )
            if bot.indicators:
                st.subheader("Trading Signals")
                st.dataframe(pd.DataFrame(bot.indicators), use_container_width=True)
            else:
                st.info("💡 Waiting for signals...")
        except Exception as e:
            st.error(f"❌ Trading data error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view signals")
async def _render_analysis_tab(bot):
    """Rendu de l'onglet Analysis"""
    if st.session_state.bot_running:
        try:
            if bot.latest_data and bot.indicators:
                st.subheader("Technical Analysis")
                for symbol in bot.latest_data:
                    await process_market_data(bot, symbol)
                if hasattr(bot, "advanced_indicators"):
                    analysis = bot.advanced_indicators.get_all_signals()
                    st.dataframe(pd.DataFrame(analysis), use_container_width=True)
                else:
                    st.info("💡 Processing analysis...")
            else:
                st.info("💡 Waiting for market data...")
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view analysis")
    # --- Signal Quantum SVM ---
    if hasattr(bot, "qsvm") and bot.qsvm is not None:
        try:
            # Prépare les features à passer à predict (adapte cette ligne selon ta logique)
            features = (
                bot.latest_data
            )  # ou bot.indicators ou ton dataframe, adapte selon besoin
            quantum_signal = bot.qsvm.predict(features)
            st.subheader("Quantum SVM Signal")
            st.metric("Quantum SVM Signal", quantum_signal)
        except Exception as e:
            st.warning(f"Erreur Quantum SVM : {e}")
async def shutdown():
    """Arrêt propre de l'application"""
    try:
        # Récupération des tâches en cours
        tasks = [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if tasks:
            # Annulation des tâches
            for task in tasks:
                task.cancel()
            # Attente de la fin des tâches avec timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout during tasks cancellation")
        # Nettoyage via le gestionnaire de sessions
        await session_manager.cleanup()
        # Nettoyage des ressources du bot
        if "bot_instance" in st.session_state:
            bot = st.session_state.bot_instance
            await cleanup_resources(bot)
        logger.info(
            """
╔═════════════════════════════════════════════════╗
║              SHUTDOWN COMPLETED                  ║
╠═════════════════════════════════════════════════╣
║ All resources cleaned and sessions closed       ║
╚═════════════════════════════════════════════════╝
        """
        )
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
def main():
    """Point d'entrée principal avec protection renforcée et gestion des événements améliorée"""
    try:
        # 1. Initialisation et protection de la session
        global session_manager
        session_manager = StreamlitSessionManager()
        session_manager.protect_session()
        # 2. Log de démarrage
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              STARTING APPLICATION                ║
╠═════════════════════════════════════════════════╣
║ Session: {session_manager.session_id}
╚═════════════════════════════════════════════════╝
        """
        )
        # 3. Initialisation de l'état de session
        _initialize_session_state()
        # 4. Configuration et vérification de la boucle d'événements
        event_loop = _setup_and_verify_event_loop()
        if not event_loop:
            raise RuntimeError("Failed to initialize event loop")
        # 5. Exécution de la coroutine principale
        event_loop.run_until_complete(main_async())
    except asyncio.CancelledError:
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              GRACEFUL SHUTDOWN                   ║
╠═════════════════════════════════════════════════╣
╚═════════════════════════════════════════════════╝
        """
        )
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              RUNTIME ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        st.error(f"❌ Application error: {str(e)}")
    finally:
        _perform_cleanup()
def _initialize_session_state():
    """Initialise l'état de la session avec des valeurs sûres et logging détaillé"""
    try:
        # États par défaut avec horodatage
        default_state = {
            # États de base
            "session_id": session_id,
            "initialized": True,
            # États du bot
            "bot_running": False,
            "portfolio": None,
            "latest_data": {},
            "indicators": None,
            "refresh_count": 0,
            # États de la boucle événementielle
            "loop": None,
            "error_count": 0,
            # États WebSocket
            "ws_status": "disconnected",
            "ws_initialized": False,
            "ws_connection_status": "disconnected",
            # Protections
            "keep_alive": True,
            "prevent_cleanup": True,
            "force_cleanup": False,
            "cleanup_allowed": False,
        }
        # Initialisation des états manquants uniquement
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
        # Log de succès
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION STATE INITIALIZED              ║
╠═════════════════════════════════════════════════╣
║ Session ID: {session_id}
║ Status: Active
╚═════════════════════════════════════════════════╝
        """
        )
        return True
    except Exception as e:
        # Log d'erreur
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION STATE ERROR                    ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        return False
def _setup_and_verify_event_loop():
    """Configure et vérifie la boucle d'événements avec gestion d'erreur améliorée"""
    try:
        # Vérification de l'existence d'une boucle
        if not st.session_state.get("loop"):
            # Création et configuration de la nouvelle boucle
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply()
            # Sauvegarde dans la session
            st.session_state.loop = loop
            # Log de succès d'initialisation
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP INITIALIZED              ║
╠═════════════════════════════════════════════════╣
║ Status: Successfully configured
║ Loop ID: {id(loop)}
╚═════════════════════════════════════════════════╝
            """
            )
            return loop
        # Vérification de la boucle existante
        existing_loop = st.session_state.loop
        if existing_loop.is_closed():
            logger.warning(
                f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP CLOSED                   ║
╠═════════════════════════════════════════════════╣
║ Status: Creating new loop
║ Previous Loop ID: {id(existing_loop)}
╚═════════════════════════════════════════════════╝
            """
            )
            # Création d'une nouvelle boucle
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            nest_asyncio.apply()
            st.session_state.loop = new_loop
            return new_loop
        # Retour de la boucle existante
        logger.debug(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP VERIFIED                 ║
╠═════════════════════════════════════════════════╣
║ Status: Using existing loop
║ Loop ID: {id(existing_loop)}
╚═════════════════════════════════════════════════╝
        """
        )
        return existing_loop
    except Exception as e:
        # Log d'erreur détaillé
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP ERROR                    ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ Details: {traceback.format_exc()}
╚═════════════════════════════════════════════════╝
        """
        )
        # Incrément du compteur d'erreurs
        st.session_state.error_count = st.session_state.get("error_count", 0) + 1
        return None
    finally:
        # Mise à jour du timestamp
def _perform_cleanup():
    """Effectue le nettoyage final de l'application"""
    try:
        # 1. Protection de la session
        session_manager.protect_session()
        # 2. Nettoyage de la boucle d'événements
        if st.session_state.get("loop"):
            loop = st.session_state.loop
            if not loop.is_closed():
                try:
                    # Nettoyage conditionnel des ressources
                    if st.session_state.get(
                        "force_cleanup", False
                    ) and st.session_state.get("cleanup_allowed", False):
                        if "bot_instance" in st.session_state:
                            loop.run_until_complete(
                                cleanup_resources(st.session_state.bot_instance)
                            )
                    # NE PAS FERMER LA BOUCLE ! On ne fait PAS loop.close()
                except Exception as e:
                    logger.error(f"Loop cleanup error: {e}")
                finally:
                    # On ne détruit pas la boucle ici non plus
                    pass
        logger.info(
            """
╔═════════════════════════════════════════════════╗
║              CLEANUP COMPLETED                   ║
╠═════════════════════════════════════════════════╣
║ Status: All resources cleaned
╚═════════════════════════════════════════════════╝
        """
        )
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              CLEANUP ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
    finally:
        # Protection finale absolue
        session_manager.protect_session()
def ensure_event_loop():
    """Vérifie et assure l'existence d'une boucle d'événements valide"""
    try:
        if not st.session_state.get("loop"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply()
            st.session_state.loop = loop
            logger.info("✅ New event loop created and configured")
            return loop
        return st.session_state.loop
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP ERROR                    ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        return None
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              KEYBOARD INTERRUPT                  ║
╠═════════════════════════════════════════════════╣
║ Status: Graceful shutdown initiated
╚═════════════════════════════════════════════════╝
        """
        )
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              CRITICAL ERROR                      ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        sys.exit(1)
    finally:
        try:
            # Nettoyage final avec nouvelle boucle si nécessaire
            if "bot_instance" in st.session_state:
                try:
                    # Création d'une nouvelle boucle pour le nettoyage final
                    cleanup_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(cleanup_loop)
                    cleanup_loop.run_until_complete(
                        cleanup_resources(st.session_state.bot_instance)
                    )
                    cleanup_loop.close()
                except Exception as e:
                    logger.error(f"Final cleanup error: {e}")
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║              FINAL CLEANUP                       ║
╠═════════════════════════════════════════════════╣
║ Status: All resources cleaned
╚═════════════════════════════════════════════════╝
            """
            )
        except Exception as cleanup_error:
            logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║              CLEANUP ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Error: {str(cleanup_error)}
╚═════════════════════════════════════════════════╝
            """
            )
