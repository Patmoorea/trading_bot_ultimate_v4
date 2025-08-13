import logging
from datetime import datetime, timezone
import os
import asyncio
from asyncio import AbstractEventLoop
import traceback

# Pour Streamlit
import streamlit as st

# Pour nest_asyncio (si tu utilises Streamlit avec asyncio)
import nest_asyncio

USE_TESTNET = str(os.getenv("BINANCE_TESTNET", "False")).lower() in ("true", "1")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


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


def setup_asyncio():
    """Configure l'environnement asyncio pour Streamlit."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply()
        return loop
    except Exception as e:
        logger.error(f"Error setting up asyncio: {e}")
        return None


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

config = {
    "NEWS": {"enabled": True, "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", "")},
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


def _initialize_session_state():
    """Initialise l'état de la session avec des valeurs sûres et logging détaillé"""
    current_time = datetime.now(timezone.utc)
    current_user = os.getenv("USER", "Patmoorea")
    session_id = f"{current_user}_{int(current_time.timestamp())}"

    try:
        # États par défaut avec horodatage
        default_state = {
            # États de base
            "session_id": session_id,
            "initialization_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_update_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "user": current_user,
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
            "ws_last_heartbeat": current_time.strftime("%Y-%m-%d %H:%M:%S"),
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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ User: {current_user}
╚═════════════════════════════════════════════════╝
        """
        )
        return False


def _setup_and_verify_event_loop():
    """Configure et vérifie la boucle d'événements avec gestion d'erreur améliorée"""
    current_time = datetime.now(timezone.utc)
    current_user = os.getenv("USER", "Patmoorea")

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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ User: {current_user}
║ Details: {traceback.format_exc()}
╚═════════════════════════════════════════════════╝
        """
        )

        # Incrément du compteur d'erreurs
        st.session_state.error_count = st.session_state.get("error_count", 0) + 1

        return None

    finally:
        # Mise à jour du timestamp
        st.session_state.last_update_time = current_time.strftime("%Y-%m-%d %H:%M:%S")


class StreamlitSessionManager:
    """Gestionnaire de session Streamlit avec protection et logging améliorés"""

    def __init__(self):
        """Initialisation du gestionnaire de session"""
        self.init_time = datetime.now(timezone.utc)
        self.user = os.getenv("USER", "Patmoorea")
        self.session_id = f"{self.user}_{int(self.init_time.timestamp())}"
        self.logger = logging.getLogger(__name__)

        # Initialisation immédiate de la session
        if "session_initialized" not in st.session_state:
            if self._initialize_session_state():
                self._log_initialization()

        # Initialisation de advanced_indicators
        self.advanced_indicators = {}

    # Ajoute cette fonction utilitaire
    def safe_float(self, val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _initialize_session_state(self):
        """Initialise l'état de la session avec des valeurs sûres"""
        try:
            # États par défaut avec horodatage
            default_state = {
                # États de base
                "session_id": self.session_id,
                "initialization_time": self.init_time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_update_time": self.init_time.strftime("%Y-%m-%d %H:%M:%S"),
                "user": self.user,
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
                "ws_last_heartbeat": self.init_time.strftime("%Y-%m-%d %H:%M:%S"),
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
║ Time: {self.init_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {self.user}
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
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
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
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Session ID: {self.session_id}
║ Last Action: {st.session_state.get('last_action_time')}
╚═════════════════════════════════════════════════╝
        """
        )

    def protect_session(self):
        """Protection renforcée de la session"""
        try:
            # Vérification et réinitialisation si nécessaire
            if not st.session_state.get("session_initialized"):
                self._initialize_session_state()

            # Mise à jour du timestamp
            current_time = datetime.now(timezone.utc)
            st.session_state.last_action_time = current_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Activation des protections
            st.session_state.prevent_cleanup = True
            st.session_state.keep_alive = True
            st.session_state.force_cleanup = False
            st.session_state.cleanup_allowed = False

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
                self.logger.warning("Cannot allow cleanup while bot is running")
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
                "user": self.user,
                "session_id": self.session_id,
                "init_time": self.init_time.strftime("%Y-%m-%d %H:%M:%S"),
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


def _setup_and_verify_event_loop():
    """Configure et vérifie la boucle d'événements avec gestion d'erreur améliorée"""
    current_time = datetime.now(timezone.utc)
    current_user = os.getenv("USER", "Patmoorea")

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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
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
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ User: {current_user}
║ Details: {traceback.format_exc()}
╚═════════════════════════════════════════════════╝
        """
        )

        # Incrément du compteur d'erreurs
        st.session_state.error_count = st.session_state.get("error_count", 0) + 1

        return None

    finally:
        # Mise à jour du timestamp
        st.session_state.last_update_time = current_time.strftime("%Y-%m-%d %H:%M:%S")


# Création de l'instance globale avec vérification
try:
    session_manager = StreamlitSessionManager()
    logger.info("✅ Session manager initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize session manager: {e}")
    session_manager = None
