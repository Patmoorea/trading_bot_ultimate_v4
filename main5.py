# 5. Imports des bibliothèques externes
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
import telegram
import ccxt
import ccxt.async_support as ccxt
import ta


# 6. Imports des modules internes (exchanges, core, etc.)

from src.core.exchange import ExchangeInterface as Exchange
from src.core.buffer.circular_buffer import CircularBuffer
from src.connectors.binance import BinanceConnector
from src.portfolio.real_portfolio import RealPortfolio
from src.regime_detection.hmm_kmeans import MarketRegimeDetector


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
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
╚═════════════════════════════════════════════════╝
        """
        )
        return None
