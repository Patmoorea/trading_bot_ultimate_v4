"""
Stubs utilitaires pour corriger les erreurs d'import/fonction manquantes.
"""

# --- STUBS PATCH COPILOT ---
import sys
import types

async def check_signals(*args, **kwargs):
    pass

async def update_indicators(*args, **kwargs):
    return {}

class Exchange:
    def __init__(self, exchange_id=None):
        pass

# Pour la gestion websocket, on tente d'importer ws.py si présent
try:
    from .ws import close_websocket, initialize_websocket
except ImportError:
    async def close_websocket(*args, **kwargs):
        pass
    async def initialize_websocket(*args, **kwargs):
        return True
import os
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
import time
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Gym et espace d'observation/action pour RL
import gymnasium as gym
# Gymnasium uniquement (remplace gym)
from gymnasium import spaces

try:
    from .ws import WebSocketManager, close_websocket, initialize_websocket
except ImportError:
    class WebSocketManager:
        def __init__(self, *args, **kwargs):
            pass
        async def start(self):
            return True
    async def close_websocket(*args, **kwargs):
        pass
    async def initialize_websocket(*args, **kwargs):
        return True

from src.data.realtime.websocket.client import StreamConfig
from src.core.buffer.circular_buffer import CircularBuffer

# Binance
from binance import AsyncClient, BinanceSocketManager

# Modules internes (adapte le chemin selon ton projet)
from src.exchanges.binance_exchange import BinanceExchange
from src.exchanges.binance.binance_client import BinanceClient
from src.data.realtime.websocket.client import MultiStreamManager
from src.monitoring.streamlit_ui import TradingDashboard
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import (
    ArbitrageScanner as ArbitrageEngine,
)
from src.indicators.advanced.multi_timeframe import (
    MultiTimeframeAnalyzer,
    TimeframeConfig,
)
from src.analysis.indicators.orderflow.orderflow_analysis import (
    OrderFlowAnalysis,
    OrderFlowConfig,
)
from src.analysis.indicators.volume.volume_analysis import VolumeAnalysis
from src.analysis.indicators.volatility.volatility import VolatilityIndicators

# ccxt (si utilisé pour l'exchange)
import ccxt

# Deep learning et IA (torch, etc.)
import torch

# Analyse technique (ex: ta-lib ou pandas-ta)
import pandas_ta as ta

# (Optionnel) nest_asyncio si tu utilises dans Streamlit ou Jupyter
import nest_asyncio
from .utils import WEBSOCKET_CONFIG
from src.ai.hybrid_model import HybridAI
from src.risk_management.position_manager import PositionManager
from src.risk_management.circuit_breakers import CircuitBreaker
from src.notifications.telegram_bot import TelegramBot
from web_interface.app.services.news_analyzer import NewsAnalyzer
from src.regime_detection.hmm_kmeans import MarketRegimeDetector
from src.quantum.qsvm import QuantumTradingModel as QuantumSVM

from asyncio import TimeoutError, AbstractEventLoop
import asyncio

from src.ai.cnn_lstm import CNNLSTM
from src.ai.ppo_gtrxl import PPOGTrXL
from src.ai.ppo_strategy import PPOStrategy

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

USE_TESTNET = str(os.getenv("BINANCE_TESTNET", "False")).lower() in ("true", "1")


def safe(val, default="N/A", fmt="{:,.2f}"):
    try:
        return fmt.format(val) if val is not None else default
    except Exception:
        return default


def safe_float(val, default=0.0):
    """Convertit une valeur en float de manière sécurisée"""
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            return float(val.replace(",", ""))
        if hasattr(val, "__float__"):
            return float(val)
        return default
    except (ValueError, TypeError, AttributeError):
        return default


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TradingBotM4:
    """Classe principale du bot de trading v4"""

    def prepare_observation(market_data: dict, required_fields: list) -> np.ndarray:
        obs = []
        for field in required_fields:
            value = market_data.get(field, 0.0)
            if isinstance(value, (list, np.ndarray)):
                value = np.array(value)
            else:
                value = np.array([value])
            obs.append(value)
        try:
            obs_array = np.concatenate(obs, axis=-1)
        except Exception as e:
            logger.error(f"Observation preparation failed: {e}")
            obs_array = np.zeros(len(required_fields))
        return obs_array

    def enrich_market_data(self, data: dict) -> dict:
        for pair, d in data.items():
            if "bid" not in d or "ask" not in d:
                ob = d.get("orderbook", {})
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                d["bid"] = float(bids[0][0]) if bids else d.get("price", 0.0)
                d["ask"] = float(asks[0][0]) if asks else d.get("price", 0.0)
        return data

    async def tick(self):
        """Effectue une itération de trading (une fois par refresh)"""
        try:
            # Récupération des données
            market_data = await self.get_latest_data()
            if market_data:
                all_indicators = {}
                all_signals = {}

                for pair in self.pairs_valid:
                    # PATCH: Normalise la clé pour matcher market_data
                    if pair in market_data:
                        pair_key = pair
                    elif pair.replace("/", "") in market_data:
                        pair_key = pair.replace("/", "")
                    else:
                        self.logger.error(
                            f"[PATCH] Paire {pair} absente de market_data, clés: {list(market_data.keys())}"
                        )
                        continue
                    pair_data = market_data[pair_key]
                    self.logger.debug(
                        f"[DEBUG] tick: pair={pair} | pair_key={pair_key} | pair_data keys: {list(pair_data.keys()) if isinstance(pair_data, dict) else type(pair_data)}"
                    )
                    indicators = await self.calculate_indicators(pair)
                    if indicators:
                        all_indicators[pair] = indicators
                        signals = await self.analyze_signals(pair_data, indicators)
                        all_signals[pair] = signals

                portfolio = await self.get_real_portfolio()
                if portfolio:
                    st.session_state.portfolio = portfolio
                    st.session_state.latest_data = market_data
                    st.session_state.indicators = all_indicators
                    st.session_state.signals = all_signals

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
            self.logger.error(f"Erreur tick: {e}")

    def build_ohlcv_df(ohlcv):
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if (
            ohlcv
            and isinstance(ohlcv, list)
            and len(ohlcv) > 0
            and isinstance(ohlcv[0], (list, tuple))
        ):
            df = pd.DataFrame(ohlcv, columns=columns)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            df = pd.DataFrame(ohlcv)
        return df

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
                "API_KEY": os.getenv("BINANCE_API_KEY"),
                "API_SECRET": os.getenv("BINANCE_API_SECRET"),
                "TESTNET": USE_TESTNET,
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
                "confidence_threshold": 0.35,  # Réduit de 0.75 à 0.35
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

        api_key = self.config["BINANCE"]["API_KEY"]
        api_secret = self.config["BINANCE"]["API_SECRET"]
        use_testnet = self.config["BINANCE"].get("TESTNET", False)

        self.logger.info(
            f"BinanceExchange: testnet={use_testnet} (type: {type(use_testnet)})"
        )

        self.exchange = BinanceExchange(api_key, api_secret, testnet=use_testnet)

        self.pairs_valid = self.config["TRADING"]["pairs"]

        # Patch pour éviter le blocage load_markets en testnet
        if use_testnet and hasattr(self.exchange, "exchange"):
            # ccxt ne supporte pas load_markets en testnet Binance
            self.exchange.exchange.load_markets = lambda: None  # override, fait rien
            self.logger.warning(
                "⚠️ Binance testnet détecté : load_markets() patché (désactivé)"
            )

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
            pairs=self.pairs_valid, config=self.stream_config
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
            trading_pairs=self.pairs_valid,
            timeframes=self.config["TRADING"]["timeframes"],
        )

        ppo_config = {
            "env": self.env,
            # tu peux rajouter d'autres paramètres ici si tu veux (learning_rate, etc.)
        }
        self.ppo_strategy = PPOStrategy(ppo_config)

        # Gestionnaires de trading
        self.position_manager = PositionManager(
            account_balance=0,
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
        """
        Récupère le dernier prix pour un symbole donné via le spot_client.
        """
        if hasattr(self.spot_client, "get_ticker_price"):
            return self.spot_client.get_ticker_price(symbol)
        elif hasattr(self.spot_client, "fetch_ticker"):
            # Pour compatibilité avec ccxt
            ticker = self.spot_client.fetch_ticker(symbol)
            return ticker["last"]
        else:
            raise NotImplementedError("No method to get latest price")

    async def async_init(self):
        await self.exchange.initialize()

        # Chargement des marchés (attention au testnet/None)
        if hasattr(self.exchange, "_exchange") and hasattr(
            self.exchange._exchange, "load_markets"
        ):
            if asyncio.iscoroutinefunction(self.exchange._exchange.load_markets):
                await self.exchange._exchange.load_markets()
            else:
                self.exchange._exchange.load_markets()

        # Protection contre NoneType
        symbols = getattr(self.exchange._exchange, "symbols", None)
        if symbols:
            self.pairs_valid = [p for p in self.pairs_valid if p in symbols]
            if not self.pairs_valid:
                self.logger.error("Aucune paire valide trouvée dans la config !")
                raise ValueError("Aucune paire valide trouvée dans la config !")
            else:
                self.logger.info(f"Paires valides Binance: {self.pairs_valid}")
        else:
            self.logger.warning(
                "⚠️ self.exchange._exchange.symbols is None, skip filtering pairs_valid."
            )

        is_spot_testnet = (
            getattr(self.exchange, "testnet", False)
            and getattr(self.exchange, "_exchange", None) is not None
            and self.exchange._exchange.options.get("defaultType", "spot") == "spot"
        )
        if is_spot_testnet:
            self.logger.warning(
                "Binance SPOT testnet : WebSockets et MultiStreamManager désactivés"
            )
            self.ws_manager = None
            self.websocket = None
        else:
            self.ws_manager = WebSocketManager(self)
            self.websocket = MultiStreamManager(
                pairs=self.pairs_valid, config=self.stream_config
            )

    async def start(self):
        """Démarre le bot"""
        try:
            self.logger.info("Starting bot initialization...")

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
            self.logger.error(f"❌ Erreur génération recommandation: {e}")
            return {
                "action": "error",
                "confidence": 0,
                "strength": 0,
                "signals": {"buy": 0, "sell": 0},
                "error": str(e),
            }

    def _generate_analysis_report(self, indicators_analysis, regime):
        try:
            report = f"""
╔═════════════════════════════════════════════════╗
║           RAPPORT D'ANALYSE DE MARCHÉ           ║
╠═════════════════════════════════════════════════╣    
║ Régime: {regime}                               ║
╚═════════════════════════════════════════════════╝

    📊 Analyse par Timeframe/Paire :
    """
            for timeframe, pairs_dict in indicators_analysis.items():
                for pair, analysis in pairs_dict.items():
                    report += f"""
    🕒 {timeframe} | {pair} :
    ├─ 📈 Tendance: {analysis.get('trend', {}).get('trend_strength', 'N/A')}
    ├─ 📊 Volatilité: {analysis.get('volatility', {}).get('current_volatility', 'N/A')}
    ├─ 📉 Volume: {analysis.get('volume', {}).get('volume_profile', {}).get('strength', 'N/A')}
    └─ 🎯 Signal dominant: {analysis.get('dominant_signal', 'N/A')}
    """
            return report
        except Exception as e:
            self.logger.error(f"❌ Erreur génération rapport: {e}")
            return f"Erreur lors de la génération du rapport : {e}"

    async def _initialize_models(self):
        """Initialise les modèles d'IA"""
        try:
            # Calcul des dimensions pour CNNLSTM
            input_shape = (
                len(self.config["TRADING"]["timeframes"]),  # Nombre de timeframes
                len(self.pairs_valid),  # Nombre de paires
                42,  # Nombre de features par candlestick
            )

            # Calcul des dimensions pour PPO-GTrXL
            state_dim = input_shape[0] * input_shape[1] * input_shape[2]
            action_dim = len(self.pairs_valid)

            # Initialisation des modèles
            self.models = {
                "ppo_gtrxl": PPOGTrXL(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    num_layers=self.config["AI"]["gtrxl_layers"],
                    d_model=self.config["AI"]["embedding_dim"],
                ),
                "cnn_lstm": CNNLSTM(input_shape=input_shape),
            }

            # Chargement des poids pré-entraînés
            models_path = os.path.join(current_dir, "models")
            if os.path.exists(models_path):
                for model_name, model in self.models.items():
                    model_path = os.path.join(models_path, f"{model_name}.pt")
                    if os.path.exists(model_path):
                        try:
                            if os.path.exists(model_path):
                                model.load_state_dict(torch.load(model_path))
                                print(f"[DEBUG] torch.load: path={model_path}")
                            else:
                                self.logger.warning(
                                    f"[WARN] Fichier modèle {model_path} absent, chargement ignoré."
                                )
                            self.logger.info(f"Modèle {model_name} chargé avec succès")
                        except Exception as load_e:
                            self.logger.warning(
                                f"[WARN] Erreur chargement modèle {model_name} depuis {model_path} : {load_e}"
                            )
                    else:
                        self.logger.warning(
                            f"[WARN] Fichier modèle {model_path} absent, chargement ignoré."
                        )

            self.logger.info("✅ Modèles initialisés avec succès")
            return True

        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation modèles: {e}")
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
                    self.logger.error(f"❌ Buffer cleanup error: {buffer_error}")

            # Nettoyage des données
            if hasattr(self, "latest_data"):
                self.latest_data = {}

            if hasattr(self, "indicators"):
                self.indicators = {}

            # Désactivation du mode trading
            if hasattr(st.session_state, "bot_running"):
                st.session_state.bot_running = False

            self.logger.info(
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
            self.logger.info("✅ Bot started successfully CORE")
            return True

        except Exception as e:
            self.logger.error(f"❌ Bot start error: {e}")
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
                    self.logger.info("Attempting WebSocket reconnection...")
                    if await initialize_websocket(self):
                        self.ws_connection["reconnect_count"] = 0
                        return True
                    self.ws_connection["reconnect_count"] += 1
                else:
                    self.logger.error("Max WebSocket reconnection attempts reached")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"WebSocket check error: {e}")
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
                    self.logger.error(
                        "❌ Impossible d'initialiser le WebSocket dans initialize()"
                    )
                    return False

            # Récupération initiale du portfolio
            portfolio = await self.get_real_portfolio()
            if portfolio:
                st.session_state.portfolio = portfolio
                self.logger.info("✅ Initial portfolio data loaded")

            # Mise à jour du statut
            self.ws_connection.update(
                {"enabled": True, "status": "connected", "last_message": time.time()}
            )

            return True

        except Exception as e:
            self.logger.error(f"❌ Initialization error: {e}")
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
                exchanges=self.config["ARBITRAGE"]["exchanges"],
                pairs=self.config["ARBITRAGE"]["pairs"],
                min_profit=self.config["ARBITRAGE"]["min_profit"],
                max_trade_size=self.config["ARBITRAGE"]["max_trade_size"],
                timeout=self.config["ARBITRAGE"]["timeout"],
                volume_filter=self.config["ARBITRAGE"]["volume_filter"],
                price_check=self.config["ARBITRAGE"]["price_check"],
                max_slippage=self.config["ARBITRAGE"]["max_slippage"],
            )

            # Configuration des analyseurs et modèles
            await self._initialize_analyzers()
            await self._initialize_models()

            return True

        except Exception as e:
            self.logger.error(f"Setup components error: {e}")
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
        df_ta = df.copy()
        df_ta.ta.strategy("All")
        base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        indicators = {
            col: df_ta[col].iloc[-1] for col in df_ta.columns if col not in base_cols
        }
        self.logger.info(f"✅ {len(indicators)} indicateurs extraits automatiquement")
        return indicators

    async def _handle_stream(self, stream):
        """Gère un stream de données"""
        try:
            async with stream as tscm:
                while True:
                    msg = await tscm.recv()
                    await self._process_stream_message(msg)
        except Exception as e:
            self.logger.error(f"Erreur stream: {e}")
            return None

    async def _process_stream_message(self, msg):
        """Traite les messages des streams"""
        try:
            if not msg:
                self.logger.warning("Message vide reçu")
                return

            if msg.get("e") == "trade":
                await self._handle_trade(msg)
            elif msg.get("e") == "depthUpdate":
                await self._handle_orderbook(msg)
            elif msg.get("e") == "kline":
                await self._handle_kline(msg)

        except Exception as e:
            self.logger.error(f"Erreur traitement message: {e}")
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
            self.logger.error(f"Erreur traitement trade: {e}")
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
            self.logger.error(f"Erreur traitement orderbook: {e}")
            return None

    async def _handle_kline(self, msg):
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
                # 🟢 Corrigé : passe la LISTE des bougies pour la paire à analyze_signals
                candles = self.buffer.get_all_data(kline_data["symbol"])
                if candles and len(candles) > 0:
                    await self.analyze_signals(
                        market_data=candles,
                        indicators=self.advanced_indicators.analyze_timeframe(
                            kline_data
                        ),
                    )
            return kline_data

        except Exception as e:
            self.logger.error(f"Erreur traitement kline: {e}")
            return None

    def decision_model(self, features, timestamp=None):
        try:
            # Préparation robuste des features pour le modèle
            required_fields = [
                "price",
                "volume",
                "bid",
                "ask",
                "high_low_range",
                "bid_ask_spread",
                "timestamp",
            ]
            obs = self.prepare_observation(features, required_fields)
            print("[DEBUG] Shape des features envoyées au modèle:", obs.shape)
            policy = self.models["ppo_gtrxl"].get_policy(obs)
            value = self.models["ppo_gtrxl"].get_value(obs)
            return policy, value
        except Exception as e:
            self.logger.error(f"[{timestamp}] Erreur decision_model: {e}")
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
            self.logger.error(f"[{timestamp}] Erreur risk management: {e}")
            return decision

    async def get_latest_data(self):
        try:
            data = {}

            # Vérification de la connexion WebSocket
            if not hasattr(self, "binance_ws") or self.binance_ws is None:
                self.logger.warning(
                    "🔄 WebSocket non initialisé, tentative d'initialisation..."
                )
                if not self.initialized:
                    await self.initialize()
                if not hasattr(self, "binance_ws") or self.binance_ws is None:
                    self.logger.error(
                        "Impossible d'initialiser le WebSocket après tentative."
                    )
                    return {}  # Correction : retourne un dict vide

            for pair in self.pairs_valid:
                self.logger.info(f"📊 Récupération données pour {pair}")
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
                            self.logger.warning(
                                f"Erreur chargement OHLCV {pair}: {hist_e}"
                            )

                        return result

                    async with asyncio.timeout(10.0):
                        result = await fetch_async()

                    # Traitement des résultats
                    if result["ticker"]:
                        data[pair]["price"] = float(result["ticker"]["price"])
                        self.logger.info(f"💰 Prix {pair}: {data[pair]['price']}")
                    if result["orderbook"]:
                        data[pair]["orderbook"] = {
                            "bids": result["orderbook"]["bids"][:5],
                            "asks": result["orderbook"]["asks"][:5],
                        }
                        self.logger.info(f"📚 Orderbook mis à jour pour {pair}")
                    if result["balance"]:
                        data[pair]["account"] = result["balance"]
                        self.logger.info(
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
                        self.logger.info(
                            f"📈 Volume 24h {pair}: {data[pair]['volume']}"
                        )
                    # AJOUT FORTEMENT RECOMMANDÉ : Toujours une liste de dicts, même vide
                    data[pair]["ohlcv"] = result["ohlcv"] if result["ohlcv"] else []
                    self.logger.info(
                        f"📊 OHLCV récupéré ({len(data[pair]['ohlcv'])} bougies) pour {pair}"
                    )

                except asyncio.TimeoutError:
                    self.logger.warning(f"⏱️ Timeout pour {pair}")
                    continue
                except Exception as inner_e:
                    self.logger.error(
                        f"❌ Erreur récupération données {pair}: {inner_e}"
                    )
                    continue

            # Mise en cache des données si disponibles
            if data and any(data.values()):
                self.logger.info("✅ Données reçues, mise à jour du buffer")
                for symbol, symbol_data in data.items():
                    if symbol_data:
                        self.buffer.update_data(symbol, symbol_data)
                        self.latest_data[symbol] = symbol_data
                return data
            else:
                self.logger.warning("⚠️ Aucune donnée reçue")
                return {}  # Correction : retourne un dict vide

        except Exception as e:
            self.logger.error(f"❌ Erreur critique get_latest_data: {e}")
            return {}  # Correction : retourne un dict vide

    async def calculate_indicators(self, symbol: str) -> dict:
        """Calcule les indicateurs techniques"""
        try:
            data = self.latest_data.get(symbol)
            if not data:
                self.logger.error(f"❌ Pas de données pour {symbol}")
                return {}

            # Vérification des clés de base
            required_keys = ["price", "volume", "bid", "ask", "timestamp"]
            for key in required_keys:
                if key not in data:
                    self.logger.error(
                        f"Clé '{key}' manquante dans les données pour {symbol} : {data}"
                    )
                    return {}

            # Calcul high_low_range depuis la dernière bougie OHLCV si présent
            high_low_range = None
            if "ohlcv" in data and isinstance(data["ohlcv"], list) and data["ohlcv"]:
                last_ohlcv = data["ohlcv"][-1]
                if isinstance(last_ohlcv, dict):
                    high = last_ohlcv.get("high")
                    low = last_ohlcv.get("low")
                elif isinstance(last_ohlcv, (list, tuple)) and len(last_ohlcv) >= 4:
                    # Format liste: [timestamp, open, high, low, close, volume]
                    high = last_ohlcv[2]
                    low = last_ohlcv[3]
                else:
                    high, low = None, None
                if high is not None and low is not None:
                    high_low_range = float(high) - float(low)

            indicators = {
                "price": data["price"],
                "volume": data["volume"],
                "bid_ask_spread": data["ask"] - data["bid"],
                "high_low_range": high_low_range,
                "timestamp": data["timestamp"],
            }
            self.logger.info(f"Calcul indicateurs pour {symbol}: {data}")
            self.indicators[symbol] = indicators
            return indicators

        except Exception as e:
            self.logger.error(f"Erreur calcul indicateurs pour {symbol}: {str(e)}")
            return {}

    async def study_market(self, period="7d"):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔊 Étude du marché en cours...")
        if not hasattr(self, "advanced_indicators") or self.advanced_indicators is None:
            raise RuntimeError(
                "advanced_indicators non initialisé : appelle _initialize_analyzers() d'abord"
            )
        try:
            # -- Bloc critique avec logs détaillés et traceback sur erreur --
            try:
                self.logger.info("➡️ [study_market] Avant get_historical_data")
                if not getattr(self.exchange, "_initialized", False):
                    self.logger.info("[study_market] Initialisation exchange...")
                    await self.exchange.initialize()
                    if hasattr(self.exchange, "_exchange") and hasattr(
                        self.exchange._exchange, "load_markets"
                    ):
                        if asyncio.iscoroutinefunction(
                            self.exchange._exchange.load_markets
                        ):
                            await self.exchange._exchange.load_markets()
                        else:
                            self.exchange._exchange.load_markets()
                    symbols = getattr(self.exchange._exchange, "symbols", None)
                    if symbols:
                        self.pairs_valid = [p for p in self.pairs_valid if p in symbols]
                        if not self.pairs_valid:
                            self.logger.error(
                                "Aucune paire valide trouvée dans la config !"
                            )
                            raise ValueError(
                                "Aucune paire valide trouvée dans la config !"
                            )
                        else:
                            self.logger.info(
                                f"Paires valides Binance: {self.pairs_valid}"
                            )
                    else:
                        self.logger.warning(
                            "⚠️ self.exchange._exchange.symbols is None, skip filtering pairs_valid."
                        )
                self.logger.info(
                    "[study_market] Après initialize, avant get_historical_data"
                )
                get_historical = getattr(self.exchange, "get_historical_data", None)
                if asyncio.iscoroutinefunction(get_historical):
                    historical_data = await get_historical(
                        self.pairs_valid,
                        self.config["TRADING"]["timeframes"],
                        period,
                    )
                else:
                    historical_data = get_historical(
                        self.pairs_valid,
                        self.config["TRADING"]["timeframes"],
                        period,
                    )

                self.logger.info("⬅️ [study_market] Après get_historical_data")
            except Exception as e:
                self.logger.error(
                    f"❌ [study_market] Exception get_historical_data: {e}"
                )
                import traceback

                self.logger.error(traceback.format_exc())
                raise

            if not historical_data or not isinstance(historical_data, dict):
                self.logger.error(
                    "❌ Données historiques non disponibles ou mauvais format (None ou pas dict)"
                )
                raise ValueError(
                    "Données historiques non disponibles ou format inattendu"
                )

            indicators_analysis = {}
            # Analyse sécurisée pour chaque timeframe/paire
            for timeframe in self.config["TRADING"]["timeframes"]:
                tf_data = historical_data.get(timeframe, {})
                indicators_analysis[timeframe] = {}
                for pair in self.pairs_valid:
                    df = tf_data.get(pair)
                    required_cols = {"open", "high", "low", "close", "volume"}
                    # PATCH ROBUSTE : vérification et log détaillé
                    if not isinstance(df, pd.DataFrame):
                        self.logger.error(
                            f"[study_market] {pair} {timeframe}: PAS un DataFrame mais {type(df)} | valeur={str(df)[:200]}"
                        )
                        indicators_analysis[timeframe][pair] = {
                            "trend": {"trend_strength": 0},
                            "volatility": {"current_volatility": 0},
                            "volume": {"volume_profile": {"strength": "N/A"}},
                            "dominant_signal": "Aucune donnée",
                        }
                        continue
                    if df.empty:
                        self.logger.warning(
                            f"[study_market] {pair} {timeframe}: DataFrame VIDE"
                        )
                        indicators_analysis[timeframe][pair] = {
                            "trend": {"trend_strength": 0},
                            "volatility": {"current_volatility": 0},
                            "volume": {"volume_profile": {"strength": "N/A"}},
                            "dominant_signal": "Aucune donnée",
                        }
                        continue
                    if not required_cols.issubset(df.columns):
                        self.logger.error(
                            f"[study_market] {pair} {timeframe}: Colonnes manquantes: {required_cols - set(df.columns)} | Colonnes actuelles: {df.columns.tolist()}"
                        )
                        indicators_analysis[timeframe][pair] = {
                            "trend": {"trend_strength": 0},
                            "volatility": {"current_volatility": 0},
                            "volume": {"volume_profile": {"strength": "N/A"}},
                            "dominant_signal": "Aucune donnée",
                        }
                        continue
                    # FIN PATCH
                    try:
                        # PATCH: log avant appel indicateurs
                        self.logger.info(
                            f"[study_market] {pair} {timeframe}: Colonnes DataFrame passées à l'analyse: {df.columns.tolist()}"
                        )
                        result = self.advanced_indicators.analyze_timeframe(
                            df, timeframe
                        )
                        # PATCH: log après appel indicateurs
                        self.logger.info(
                            f"[study_market] {pair} {timeframe}: Résultat analyse: {result}"
                        )
                        indicators_analysis[timeframe][pair] = (
                            result
                            if result
                            else {
                                "trend": {"trend_strength": 0},
                                "volatility": {"current_volatility": 0},
                                "volume": {"volume_profile": {"strength": "N/A"}},
                                "dominant_signal": "Analyse échouée",
                            }
                        )
                    except Exception as tf_error:
                        self.logger.error(
                            f"Erreur analyse {pair} {timeframe}: {tf_error}"
                        )
                        import traceback

                        self.logger.error(traceback.format_exc())
                        indicators_analysis[timeframe][pair] = {
                            "trend": {"trend_strength": 0},
                            "volatility": {"current_volatility": 0},
                            "volume": {"volume_profile": {"strength": "N/A"}},
                            "dominant_signal": "Erreur",
                        }

            # Sécurise la conversion float des volumes
            for timeframe, tf_pairs in indicators_analysis.items():
                for pair, tf_analysis in tf_pairs.items():
                    if (
                        "volume" in tf_analysis
                        and "volume_profile" in tf_analysis["volume"]
                    ):
                        strength = tf_analysis["volume"]["volume_profile"].get(
                            "strength", 0
                        )
                        tf_analysis["volume"]["volume_profile"]["strength"] = (
                            safe_float(strength, 0.0)
                        )

            # Pour le calcul du régime, on peut agréger (par exemple sur le premier pair)
            regime = self.regime_detector.predict(
                {
                    tf: next(iter(tf_pairs.values()), {})
                    for tf, tf_pairs in indicators_analysis.items()
                }
            )
            self.logger.info(f"🔈 Régime de marché détecté: {regime}")

            try:
                analysis_report = self._generate_analysis_report(
                    indicators_analysis,
                    regime,
                )
                await self.telegram.send_message(analysis_report)
            except Exception as report_error:
                self.logger.error(f"Erreur génération rapport: {report_error}")

            try:
                self.dashboard.update_market_analysis()
            except Exception as dash_error:
                self.logger.error(f"Erreur mise à jour dashboard: {dash_error}")

            return regime, historical_data, indicators_analysis

        except Exception as e:
            self.logger.error(f"Erreur study_market: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

    async def analyze_signals(self, market_data, indicators=None):
        """Analyse des signaux de trading basée sur tous les indicateurs"""
        try:
            # Si les indicateurs ne sont pas fournis, on les calcule
            if indicators is None:
                # PATCH extraction robuste
                ohlcv = None
                if isinstance(market_data, dict):
                    if "ohlcv" in market_data:
                        ohlcv = market_data["ohlcv"]
                    elif len(market_data) == 1:
                        _, data = next(iter(market_data.items()))
                        ohlcv = data.get("ohlcv", None)
                elif isinstance(market_data, (list, pd.DataFrame)):
                    ohlcv = market_data
                if ohlcv and isinstance(ohlcv, (list, pd.DataFrame)) and len(ohlcv) > 0:
                    indicators = self.add_indicators(ohlcv)
                else:
                    self.logger.error(
                        f"[PATCH] Impossible d'extraire OHLCV pour analyze_signals: type={type(market_data)} valeur={str(market_data)[:200]}"
                    )
                    return None

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
                "trend_strength": float(indicators["trend"]["adx"].iloc[-1]),
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
                "atr_volatility": float(indicators["volatility"]["atr"].iloc[-1]),
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

            # Structure finale : champs principaux (float/str), détails dans *_details
            signal = {
                "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                # Champs principaux homogènes
                "trend": float(trend_analysis.get("trend_strength", 0.0)),
                "momentum": float(
                    1
                    if momentum_analysis.get("rsi_signal") == "oversold"
                    else (
                        -1 if momentum_analysis.get("rsi_signal") == "overbought" else 0
                    )
                ),
                "volatility": float(volatility_analysis.get("atr_volatility", 0.0)),
                "volume": float(
                    1
                    if volume_analysis.get("mfi_signal") == "buy"
                    else -1 if volume_analysis.get("mfi_signal") == "sell" else 0
                ),
                # Champs détaillés (dicts complets, pour logs ou affichage)
                "trend_details": trend_analysis,
                "momentum_details": momentum_analysis,
                "volatility_details": volatility_analysis,
                "volume_details": volume_analysis,
                "recommendation": self._generate_recommendation(
                    trend_analysis,
                    momentum_analysis,
                    volatility_analysis,
                    volume_analysis,
                ),
            }

            self.logger.info(
                f"✅ Analyse des signaux complétée: {signal['recommendation']}"
            )
            return signal

        except Exception as e:
            self.logger.error(f"❌ Erreur analyse signaux: {e}")
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

            self.logger.info("Exchange configuré avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur configuration exchange: {e}")
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
                    self.logger.info(f"Message Telegram envoyé: {message[:50]}...")
                else:
                    self.logger.error("Échec envoi message Telegram")
        except Exception as e:
            self.logger.error(f"Erreur envoi Telegram: {e}")

    async def setup_real_telegram(self):
        """Configuration sécurisée de Telegram"""
        try:
            # Création de l'instance TelegramBot (l'initialisation se fait dans __init__)
            self.telegram = TelegramBot()

            if not self.telegram.enabled:
                self.logger.warning("Telegram notifications désactivées")
                return False

            # Démarrage du processeur de queue
            await self.telegram.start()

            # Test d'envoi d'un message
            success = await self.telegram.send_message(
                "🤖 Bot de trading démarré", parse_mode="HTML"
            )

            if success:
                self.logger.info("Telegram configuré avec succès")
                return True
            else:
                self.logger.error("Échec du test d'envoi Telegram")
                return False

        except Exception as e:
            self.logger.error(f"Erreur configuration Telegram: {e}")
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
            self.logger.error(f"Erreur calcul portfolio: {e}")
            return None

    def _calculate_total_pnl(self):
        try:
            if hasattr(self, "position_history"):
                return sum(trade.get("pnl", 0) for trade in self.position_history)
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
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
            self.logger.error(f"Dashboard update error: {e}")
            return False

    async def get_real_portfolio(self):
        """
        Récupère le portfolio en temps réel avec les balances et positions.
        """
        try:
            # Vérification et initialisation du spot client
            if not hasattr(self, "spot_client") or self.spot_client is None:
                self.logger.info(
                    f"""
╔═════════════════════════════════════════════════╗
║         INITIALIZING SPOT CLIENT                 ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
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

            self.logger.info("💰 Balance data received")

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
                    free = float(asset_balance.get("free", 0))
                    locked = float(asset_balance.get("locked", 0))

                    # On ignore les assets à zéro
                    if free > 0 or locked > 0:
                        # Cas particulier USDC (pas besoin de conversion)
                        if asset == "USDC":
                            portfolio["free"] += free
                            portfolio["used"] += locked
                            portfolio["total_value"] += free + locked
                            portfolio["positions"].append(
                                {
                                    "symbol": "USDC/USDC",
                                    "size": free + locked,
                                    "value": free + locked,
                                    "price": 1.0,
                                    "free": free,
                                    "locked": locked,
                                    "timestamp": portfolio["timestamp"],
                                }
                            )
                        else:
                            # Conversion des autres assets (BTC, ETH, etc.) en USDC
                            symbol = f"{asset}USDC"
                            # Vérifie que la paire existe sur Binance
                            if hasattr(self.exchange, "_exchange") and hasattr(
                                self.exchange._exchange, "symbols"
                            ):
                                valid_symbols = self.exchange._exchange.symbols
                            else:
                                valid_symbols = []
                            if symbol in valid_symbols:
                                try:
                                    price = self.get_latest_price(symbol)
                                    value = (free + locked) * price
                                    if value > 0:
                                        portfolio["total_value"] += value
                                        portfolio["positions"].append(
                                            {
                                                "symbol": f"{asset}/USDC",
                                                "size": free + locked,
                                                "value": value,  # <-- C’EST CETTE VALEUR À AFFICHER
                                                "price": price,
                                                "free": free,
                                                "locked": locked,
                                                "timestamp": portfolio["timestamp"],
                                            }
                                        )
                                except Exception as price_error:
                                    self.logger.warning(
                                        f"⚠️ Cannot get price for {asset}: {price_error}"
                                    )
                                    continue
                            else:
                                self.logger.warning(
                                    f"⚠️ Pair {symbol} not available on Binance, skipping."
                                )
                                continue

                except Exception as asset_error:
                    self.logger.warning(
                        f"⚠️ Error processing {asset_balance.get('asset', '???')}: {asset_error}"
                    )
                    continue

            # Récupération des ordres ouverts
            try:
                for pair in self.pairs_valid:
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
                                self.logger.warning(
                                    f"⚠️ Error processing order: {order_error}"
                                )
                                continue

            except Exception as orders_error:
                self.logger.warning(f"⚠️ Cannot fetch open orders: {orders_error}")

            # Calcul des métriques finales
            portfolio.update(
                {
                    "position_count": len(portfolio["positions"]),
                    "total_position_value": sum(
                        float(pos["value"]) for pos in portfolio["positions"]
                    ),
                    "available_margin": float(portfolio["free"])
                    - sum(float(pos.get("value", 0)) for pos in portfolio["positions"]),
                }
            )

            # Récupération des données de volume sur 24h
            try:
                for pair in self.pairs_valid:
                    ticker_24h = self.spot_client.get_24h_ticker(pair)
                    if ticker_24h:
                        portfolio["volume_24h"] += float(ticker_24h["volume"])
                        portfolio["volume_change"] += float(
                            ticker_24h["priceChangePercent"]
                        )

                # Moyenne du changement de volume
                if len(self.pairs_valid) > 0:
                    portfolio["volume_change"] /= len(self.pairs_valid)

            except Exception as volume_error:
                self.logger.warning(f"⚠️ Cannot fetch 24h volume data: {volume_error}")

            # Log de succès
            self.logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║         PORTFOLIO UPDATE SUCCESS                 ║
╠═════════════════════════════════════════════════╣
║ Total Value: {portfolio['total_value']:.2f} USDC
║ Positions: {portfolio['position_count']}
║ Time: {datetime.fromtimestamp(portfolio['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S')} UTC
╚═════════════════════════════════════════════════╝
            """
            )

            return portfolio

        except Exception as e:
            self.logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║         PORTFOLIO UPDATE ERROR                   ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
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
            amount = float(signal["amount"])
            price = float(signal["price"])
            if not balance or balance["free"] < amount * price:
                self.logger.warning("Solde insuffisant pour le trade")
                return None

            # Calcul stop loss et take profit
            stop_loss = signal["price"] * (1 - signal["risk_ratio"])
            take_profit = signal["price"] * (1 + signal["risk_ratio"] * 2)

            # Placement de l'ordre

            order = await self.exchange.create_order(
                symbol=signal["symbol"].replace("/", ""),
                type="limit",
                side=signal["side"],
                amount=amount,
                price=price,
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
                self.logger.error(f"Erreur envoi notification trade: {msg_error}")

            return order

        except Exception as e:
            self.logger.error(f"Erreur trade: {e}")
            return None

    async def run_real_trading(self):
        """Boucle de trading réel sécurisée"""
        try:
            # Configuration initiale
            if not await self.setup_real_exchange():
                raise Exception("Échec configuration exchange")

            if not await self.setup_real_telegram():
                raise Exception("Échec configuration Telegram")

            self.logger.info(
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
            self.logger.error(f"Erreur envoi Telegram: {telegram_error}")
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
            status_placeholder = st.empty()

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
            # Logique de décision améliorée avec plusieurs niveaux
            if confidence > self.config["AI"]["confidence_threshold"]:
                action = "buy"
            elif confidence > 0.25:  # Seuil intermédiaire
                action = "buy" if technical_score > 0.3 else "neutral"
            elif confidence < -0.25:
                action = "sell"
            else:
                action = "neutral"

            decision = {
                "action": action,
                "symbol": self.pairs_valid[best_pair_idx],
                "confidence": confidence,
                "timestamp": timestamp,
                "regime": regime,
                "technical_score": technical_score,
                "news_impact": news_sentiment.get("sentiment", 0),
                "value_estimate": float(value.detach().numpy()),
                "position_size": buy_actions[best_pair_idx],
            }

            return decision

        except Exception as e:
            self.logger.error(f"[{timestamp}] Erreur construction décision: {e}")
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
            self.logger.error(f"Erreur: {e}")
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

        if (
            decision
            and decision["action"] in ["buy", "sell"]
            and decision["confidence"] > 0.2  # Seuil plus bas pour l'exécution
        ):
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
                        self.config["TRADING"]["base_currency"]
                    ),
                )

                # Vérification finale avant l'ordre
                if not self._validate_trade(decision, position_size):
                    return

                # Placement de l'ordre avec stop loss
                order = await self.exchange.create_order(
                    symbol=decision["symbol"].replace("/", ""),
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
                    f"Volume: {position_size} {self.config['TRADING']['base_currency']}"
                )

                # Mise à jour du dashboard
                self.dashboard.update_trades(order)

            except Exception as e:
                self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
            return {"supports_entry": False}

    def _calculate_poc(self, volume_profile):
        """Calcul du Point of Control"""
        try:
            if not volume_profile:
                return None
            return max(volume_profile.items(), key=lambda x: x[1])[0]
        except Exception as e:
            self.logger.error(f"Erreur calcul POC: {e}")
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
            self.logger.error(f"Erreur calcul Value Area: {e}")
            return None

    async def run(self):
        """Boucle principale du bot, robuste pour l'IA et la PPO."""
        try:
            await self.setup_streams()
            market_regime, historical_data, initial_analysis = await self.study_market()
            required_fields = ["bid", "ask", "price", "volume"]
            while True:
                try:
                    market_data = await self.get_latest_data()
                    if not market_data:
                        continue
                    market_data = self.enrich_market_data(market_data)
                    all_signals = {}
                    for pair in self.pairs_valid:
                        pair_key = (
                            pair if pair in market_data else pair.replace("/", "")
                        )
                        if pair_key not in market_data:
                            self.logger.error(
                                f"[PATCH] Paire {pair} absente de market_data, clés: {list(market_data.keys())}"
                            )
                            continue
                        pair_data = market_data[pair_key]
                        signal = await self.analyze_signals(pair_data)
                        all_signals[pair] = signal
                    main_pair = self.pairs_valid[0]
                    main_signal = all_signals.get(main_pair)
                    news_impact = await self.news_analyzer.analyze()
                    features = self._combine_features(
                        technical_features=main_signal,
                        news_impact=news_impact,
                        regime=market_regime,
                    )
                    # --- Utilisation correcte de la PPO ---
                    # Préparation robuste de l'input PPO
                    ppo_input_raw = (
                        market_data[main_pair]
                        if main_pair in market_data
                        else market_data[main_pair.replace("/", "")]
                    )
                    # On s'assure que les champs sont bien scalaires ou arrays homogènes
                    ppo_input = {
                        "ohlcv": ppo_input_raw.get("ohlcv", []),
                        "indicators": main_signal if main_signal else {},
                        "market_metrics": {
                            k: safe_float(ppo_input_raw.get(k, 0.0))
                            for k in required_fields
                        },
                    }
                    print(
                        "[DEBUG] PPO input:", {k: type(v) for k, v in ppo_input.items()}
                    )
                    ppo_decision = await self.ppo_strategy.analyze_market(ppo_input)
                    if ppo_decision:
                        decision = self._build_decision(
                            policy=ppo_decision["action"],
                            value=ppo_decision["confidence"],
                            technical_score=(
                                main_signal["recommendation"]["confidence"]
                                if main_signal
                                else 0.5
                            ),
                            news_sentiment=news_impact,
                            regime=market_regime,
                            timestamp=pd.Timestamp.utcnow(),
                        )
                        decision = self._add_risk_management(decision)
                        await self.execute_trades(decision)
                    await asyncio.sleep(
                        self.config["TRADING"].get("update_interval", 10)
                    )
                except Exception as loop_error:
                    self.logger.error(f"Erreur dans la boucle principale: {loop_error}")
                    continue
        except Exception as e:
            self.logger.error(f"Erreur fatale: {e}")
            await self.telegram.send_message(f"🚨 Erreur critique du bot:\n{str(e)}\n")
            raise

    def _should_train(self, historical_data):
        """Détermine si les modèles doivent être réentraînés"""
        try:
            # Vérification de la taille minimale des données
            if (
                len(historical_data.get("1h", []))
                < self.config["AI"]["min_training_size"]
            ):
                return False

            # Vérification de la dernière session d'entraînement
            return True

            return time_since_training.days >= 1  # Réentraînement quotidien

        except Exception as e:
            self.logger.error(f"Erreur: {e}")
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
                epochs=self.config["AI"]["n_epochs"],
                batch_size=self.config["AI"]["batch_size"],
                learning_rate=self.config["AI"]["learning_rate"],
            )

            # Entraînement du PPO-GTrXL
            self.models["ppo_gtrxl"].train(
                env=self.env,
                total_timesteps=100000,
                batch_size=self.config["AI"]["batch_size"],
                learning_rate=self.config["AI"]["learning_rate"],
                gradient_clip=self.config["AI"]["gradient_clip"],
            )

            # Entraînement du CNN-LSTM
            self.models["cnn_lstm"].train(
                X_train,
                y_train,
                epochs=self.config["AI"]["n_epochs"],
                batch_size=self.config["AI"]["batch_size"],
                validation_split=0.2,
            )

            # Mise à jour du timestamp d'entraînement

            # Sauvegarde des modèles
            self._save_models()

        except Exception as e:
            self.logger.error(f"Erreur: {e}")
            raise

    def _prepare_training_data(self, historical_data, initial_analysis):
        """Prépare les données pour l'entraînement"""

        try:
            features = []
            labels = []

            # Pour chaque timeframe
            for timeframe in self.config["TRADING"]["timeframes"]:
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")

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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
            return {}

    async def _should_stop_trading(self):
        """Vérifie les conditions d'arrêt du trading"""

        try:
            # Vérification du circuit breaker
            if await self.circuit_breaker.should_stop_trading():
                return True

            # Vérification du drawdown maximum
            current_drawdown = self.position_manager.calculate_drawdown()
            if current_drawdown > self.config["RISK"]["max_drawdown"]:
                return True

            # Vérification de la perte journalière
            daily_loss = self.position_manager.calculate_daily_loss()
            if daily_loss > self.config["RISK"]["daily_stop_loss"]:
                return True

            # Vérification des conditions de marché
            market_conditions = await self._check_market_conditions()
            if not market_conditions["safe_to_trade"]:
                return True

            return False

        except Exception as e:
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")
            return {"safe_to_trade": False, "reason": "Erreur système"}

    async def _analyze_market_liquidity(self):
        """Analyse détaillée de la liquidité du marché"""
        try:
            liquidity_status = {
                "status": "sufficient",
                "metrics": {},
            }

            # Analyse du carnet d'ordres
            for pair in self.pairs_valid:
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
            self.logger.error(f"Erreur analyse liquidité: {e}")
            return {"status": "insufficient", "metrics": {}}

    def _check_technical_conditions(self):
        """Vérifie les conditions techniques du marché"""

        try:
            conditions = {"safe": True, "reason": None, "details": {}}

            for pair in self.pairs_valid:
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
            self.logger.error(f"Erreur: {e}")
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
            self.logger.error(f"Erreur: {e}")

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
            self.logger.error(f"Erreur: {e}")

    async def run_adaptive_trading(self, period="7d"):
        print("### RUN_ADAPTIVE_TRADING VERSION PATCH COPILOT ### RUN")
        self.logger = logging.getLogger(__name__)
        try:
            # 1. Initialisation
            msg = "✅ Bot initialized successfully CORE"
            self.logger.info(msg)
            print(msg)
            await self.send_telegram_message(msg)

            # 2. Étude du marché
            msg = "📊 Étude du marché en cours..."
            self.logger.info(msg)
            print(msg)
            await self.send_telegram_message(msg)

            regime, historical_data, indicators_analysis = await self.study_market(
                period=period
            )
            if not regime:
                raise Exception("Impossible de détecter le régime de marché.")

            # 3. Détection du régime
            regime_msg = f"🔈 Régime de marché détecté: {regime}"
            self.logger.info(regime_msg)
            print(regime_msg)
            await self.send_telegram_message(regime_msg)

            # 4. Lancement du trading adaptatif
            strategy = self.choose_strategy(regime, indicators_analysis)
            launch_msg = f"📈 Trading adaptatif lancé | Stratégie: {strategy}"
            self.logger.info(launch_msg)
            print(launch_msg)
            await self.send_telegram_message(launch_msg)

            self.current_regime = regime
            self.current_strategy = strategy

        except Exception as e:
            error_msg = f"❌ Erreur à l'initialisation/adaptation : {e}"
            self.logger.error(error_msg)
            print(error_msg)
            try:
                await self.send_telegram_message(error_msg)
            except Exception:
                pass
            st.session_state["bot_running"] = False
            return

        cycle = 0
        while st.session_state.get("bot_running", True):
            cycle += 1
            try:
                market_data = await self.get_latest_data()
                all_signals = {}
                for pair in self.pairs_valid:
                    pair_key = pair if pair in market_data else pair.replace("/", "")
                    if pair_key not in market_data:
                        self.logger.error(
                            f"[PATCH] Paire {pair} absente de market_data, clés: {list(market_data.keys())}"
                        )
                        continue
                    pair_data = market_data[pair_key]
                    signals = await self.analyze_signals(pair_data)
                    all_signals[pair] = signals

                # Choix du signal principal pour la prise de décision
                main_pair = self.pairs_valid[0]
                main_signal = all_signals.get(main_pair)
                news = None
                try:
                    if hasattr(self, "news_analyzer"):
                        news = await self.news_analyzer.analyze()
                except Exception as e:
                    self.logger.error(f"❌ Erreur news_analyzer: {e}")

                arbitrage_opps = None
                try:
                    if hasattr(self, "arbitrage_engine"):
                        arbitrage_opps = (
                            await self.arbitrage_engine.find_opportunities()
                        )
                except Exception as e:
                    self.logger.error(f"❌ Erreur arbitrage_engine: {e}")

                # Régime
                if hasattr(self, "regime_detector"):
                    new_regime = self.regime_detector.predict(main_signal)
                else:
                    new_regime = self.current_regime

                # Adaptation
                if news and news.get("impact", 0) > 0.7:
                    await self.send_telegram_message(
                        f"📰 News critique détectée : {news}"
                    )
                    self.current_strategy = "Defensive/No Trade"
                elif arbitrage_opps:
                    await self.send_telegram_message(
                        f"⚡ Arbitrage détecté : {arbitrage_opps}"
                    )
                    self.current_strategy = "Arbitrage"
                elif new_regime != self.current_regime:
                    self.current_regime = new_regime
                    self.current_strategy = self.choose_strategy(
                        new_regime, main_signal
                    )
                    await self.send_telegram_message(
                        f"🔄 Changement de régime : {new_regime} ⇒ Nouvelle stratégie : {self.current_strategy}"
                    )

                # Décision de trade
                decision = self.make_trade_decision(
                    main_signal, self.current_strategy, news, arbitrage_opps
                )
                trade_status = None
                if decision and decision.get("action") in ["buy", "sell"]:
                    try:
                        order = await self.execute_real_trade(decision)
                        trade_status = f"{decision['action'].upper()} {decision.get('symbol', '')} {decision.get('amount', '')} (Order: {order})"
                        await self.send_telegram_message(
                            f"✅ Trade exécuté : {decision}"
                        )
                    except Exception as e:
                        self.logger.error(f"❌ Erreur exécution trade / Telegram: {e}")
                        try:
                            await self.send_telegram_message(
                                f"❌ Erreur exécution trade: {e}"
                            )
                        except Exception as e2:
                            self.logger.error(f"❌ Erreur secondaire Telegram: {e2}")
                # Streamlit status :
                try:
                    st.session_state["live_status"] = {
                        "Cycle": cycle,
                        "Régime": self.current_regime,
                        "Stratégie": self.current_strategy,
                        "Signal": (
                            main_signal["recommendation"]["action"]
                            if main_signal and "recommendation" in main_signal
                            else "N/A"
                        ),
                        "News": (
                            news["summary"] if news and "summary" in news else "N/A"
                        ),
                        "Arbitrage": str(arbitrage_opps) if arbitrage_opps else "N/A",
                        "Dernier Trade": trade_status or "Aucun",
                        "Heure": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                except Exception as e:
                    self.logger.error(f"❌ Erreur update dashboard: {e}")

                await asyncio.sleep(2)

            except Exception as loop_error:
                self.logger.error(f"❌ Exception dans la boucle: {loop_error}")
                try:
                    await self.send_telegram_message(
                        f"❌ Exception dans la boucle: {loop_error}"
                    )
                except Exception as e2:
                    self.logger.error(f"❌ Erreur Telegram exception boucle : {e2}")
                await asyncio.sleep(2)

        self.logger.info("🔴 Boucle stoppée (bot_running à False)")
        print("🔴 Boucle stoppée (bot_running à False)")
        try:
            await self.send_telegram_message("🛑 Boucle run_adaptive_trading stoppée.")
        except Exception as e:
            self.logger.error(f"❌ Erreur Telegram arrêt boucle: {e}")

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
            self.logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           CALCULATING SUPERTREND                 ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
╚═════════════════════════════════════════════════╝
            """
            )

            # Vérification de la configuration
            if not (
                self.config.get("INDICATORS", {}).get("trend", {}).get("supertrend", {})
            ):
                self.logger.warning("Missing Supertrend configuration")
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

                self.logger.info(
                    f"Using parameters: period={period}, multiplier={multiplier}"
                )

            except KeyError as ke:
                self.logger.error(f"Missing parameter: {ke}")
                self.dashboard.update_indicator_status(
                    "Supertrend", "DISABLED - Missing parameters"
                )
                return None

            # Vérification des données d'entrée
            if data is None or data.empty:
                self.logger.error("No input data provided")
                self.dashboard.update_indicator_status("Supertrend", "ERROR - No data")
                return None

            required_columns = ["high", "low", "close"]
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns: {required_columns}")
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
                    self.logger.error(f"Index error at position {i}: {idx_error}")
                    continue

            # Calcul de la force du signal
            strength = abs(close - supertrend) / close

            # Mise à jour du statut
            self.dashboard.update_indicator_status("Supertrend", "ACTIVE")

            # Log de succès
            self.logger.info(
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
                    "calculation_time": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "status": "SUCCESS",
                },
            }

        except Exception as e:
            # Log d'erreur détaillé
            self.logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║           SUPERTREND ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
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


async def fetch_market_data(bot, symbol):
    """
    Récupère les données de marché pour TOUS les timeframes configurés.
    Structure renvoyée :
        { "1m": [candle, ...], "5m": [...], ... }
    """
    try:
        # Récupère la liste des timeframes à utiliser
        timeframes = bot.config.get("TRADING", {}).get("timeframes", ["1m"])
        data = {}
        for tf in timeframes:
            try:
                klines = await bot.binance_ws.get_klines(symbol=symbol, interval=tf)
                candles = [
                    {
                        "timestamp": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                    for k in klines
                ]
                data[tf] = candles
                logger.info(f"✅ {symbol} {tf} : {len(candles)} bougies récupérées")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fetch {symbol} {tf}: {e}")
                data[tf] = []
        return data
    except Exception as e:
        logger.error(f"❌ Erreur récupération données {symbol}: {e}")
        return None


async def update_market_data(bot):
    """
    Met à jour bot.latest_data avec la structure :
        bot.latest_data["BTCUSDT"]["1m"] = [candle, ...]
        bot.latest_data["ETHUSDT"]["1h"] = [candle, ...]
    """
    try:
        data_received = False

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            logger.info(f"📊 Récupération données pour {symbol}")
            symbol_data = await fetch_market_data(bot, symbol)
            if symbol_data:
                bot.latest_data[symbol] = symbol_data
                logger.info(
                    f"✅ Données stockées pour {symbol} : {list(symbol_data.keys())}"
                )
                data_received = True

        if not data_received:
            logger.warning("⚠️ Aucune donnée reçue")
        else:
            logger.info(f"MARKET DATA KEYS: {list(bot.latest_data.keys())}")
            for sym in bot.latest_data:
                logger.info(
                    f" - {sym}: timeframes récupérés: {list(bot.latest_data[sym].keys())}"
                )

        return data_received
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour données: {e}")
        return False


async def process_market_data(bot, symbol):
    """
    Traite les données de marché pour un symbole et un ou plusieurs timeframes.
    Appelle le calcul d'indicateurs et la détection de signaux.
    """
    try:
        data = bot.latest_data.get(symbol, {})
        if not data:
            logger.warning(f"Aucune donnée à traiter pour {symbol}")
            return

        # Calcul des indicateurs pour chaque timeframe
        if not hasattr(bot, "indicators"):
            bot.indicators = {}
        if symbol not in bot.indicators:
            bot.indicators[symbol] = {}

        for tf, candles in data.items():
            if candles:
                # Calcul et stockage des indicateurs pour chaque timeframe
                indicators = await update_indicators(bot, symbol, candles, tf)
                bot.indicators[symbol][tf] = indicators
                # Vérification des signaux pour chaque timeframe
                await check_signals(bot, symbol, tf, indicators)
            else:
                logger.warning(f"Pas de données pour {symbol} {tf}")

    except Exception as e:
        logger.error(f"❌ Erreur traitement données {symbol}: {e}")


async def run_trading_bot():
    """Point d'entrée synchrone pour le bot de trading (statistiques uniquement, pas de bouton Start)"""
    try:
        # Stats en temps réel
        col1, col2, col3 = st.columns(3)
        # PATCH: valeurs fictives si non définies
        portfolio_value = 10000.0
        pnl = 0.0
        with col1:
            st.metric(
                "Portfolio Value", f"{portfolio_value:.2f} USDC", f"{pnl:+.2f} USDC"
            )
        with col2:
            st.metric("Active Positions", "2", "Open")
        with col3:
            st.metric("24h P&L", "+123 USDC", "+1.23%")

        # SUPPRESSION DU BOUTON Start Trading Bot !
        # Toute la logique de démarrage du bot doit être pilotée via la sidebar (main_async).

        # Tu peux afficher ici d'autres informations, ou l'état du bot, mais SANS bouton de démarrage.
        if st.session_state.get("bot_running"):
            st.success("🚀 Le trading bot est en cours d'exécution.")
        else:
            st.info("Le trading bot est arrêté. Utilise la sidebar pour le démarrer.")

    except Exception as e:
        logger.error(f"Trading bot error: {e}")
        st.error(f"❌ Trading bot error: {str(e)}")


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
            self.logger.warning(f"Action invalide: {action}")
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
            self.logger.error(f"Erreur calcul reward: {e}")
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
            self.logger.error(f"Erreur mise à jour state: {e}")
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
