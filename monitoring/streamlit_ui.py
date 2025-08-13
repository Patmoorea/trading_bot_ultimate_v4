import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psutil
import platform
import subprocess
from pathlib import Path
import sys
import time
from typing import Dict, Any
from ta.trend import SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import ccxt
import websockets
import asyncio
import json
import logging
from dataclasses import dataclass


@dataclass
class MarketData:
    symbol: str
    timeframe: str
    open_prices: pd.Series
    high_prices: pd.Series
    low_prices: pd.Series
    close_prices: pd.Series
    volume: pd.Series
    timestamp: pd.Series


class TechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._initialize_data()

    def _initialize_data(self):
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

    def calculate_all_indicators(self):
        indicators = {}
        # Calcul des moyennes mobiles
        indicators["sma_20"] = self.calculate_sma(20)
        indicators["sma_50"] = self.calculate_sma(50)
        indicators["ema_20"] = self.calculate_ema(20)
        # Calcul des bandes de Bollinger
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands()
        indicators.update(
            {"bb_upper": bb_upper, "bb_middle": bb_middle, "bb_lower": bb_lower}
        )
        # Calcul du MACD
        macd_line, signal_line, histogram = self.calculate_macd()
        indicators.update(
            {"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram}
        )
        # Calcul du RSI
        indicators["rsi"] = self.calculate_rsi()
        # Calcul de l'ATR
        indicators["atr"] = self.calculate_atr()
        return indicators

    def calculate_sma(self, period: int = 20) -> pd.Series:
        return self.data["close"].rolling(window=period).mean()

    def calculate_ema(self, period: int = 20) -> pd.Series:
        return self.data["close"].ewm(span=period, adjust=False).mean()

    def calculate_bollinger_bands(self, period: int = 20, std_dev: int = 2):
        middle_band = self.calculate_sma(period)
        std = self.data["close"].rolling(window=period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        return upper_band, middle_band, lower_band

    def calculate_macd(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        exp1 = self.data["close"].ewm(span=fast_period, adjust=False).mean()
        exp2 = self.data["close"].ewm(span=slow_period, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        delta = self.data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, period: int = 14) -> pd.Series:
        high_low = self.data["high"] - self.data["low"]
        high_close = np.abs(self.data["high"] - self.data["close"].shift())
        low_close = np.abs(self.data["low"] - self.data["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()


class TradingDashboard:
    def __init__(self):
        # Suppression de self.setup_streamlit()
        self.initialize_components()
        self.setup_logging()
        self.initialize_exchange()
        self.load_initial_data()

    def initialize_components(self):
        """Initialisation des composants du dashboard"""
        self.market_data = {}
        self.indicators = {}
        self.positions = {}
        self.trades_history = []
        self.performance_metrics = {
            "pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "trades_count": 0,
        }
        if "start_time" not in st.session_state:
            st.session_state.start_time = "2025-06-24 04:43:37"  # Current UTC time
        if "selected_symbol" not in st.session_state:
            st.session_state.selected_symbol = "BTC/USDT"
        if "selected_timeframe" not in st.session_state:
            st.session_state.selected_timeframe = "1m"

    def setup_logging(self):
        """Configuration du logging"""
        self.logger = logging.getLogger("TradingDashboard")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def initialize_exchange(self):
        """Initialisation de la connexion à l'exchange"""
        try:
            self.exchange = ccxt.binance(
                {"enableRateLimit": True, "options": {"defaultType": "future"}}
            )
            self.logger.info("Exchange initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            self.exchange = None

    def load_initial_data(self):
        """Chargement des données initiales"""
        try:
            symbol = st.session_state.selected_symbol
            timeframe = st.session_state.selected_timeframe
            if self.exchange:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                self.market_data[symbol] = MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open_prices=df["open"],
                    high_prices=df["high"],
                    low_prices=df["low"],
                    close_prices=df["close"],
                    volume=df["volume"],
                    timestamp=df["timestamp"],
                )
                self.calculate_indicators(symbol)
                self.logger.info(f"Initial data loaded for {symbol}")
            else:
                self.logger.warning("Exchange not initialized, using sample data")
                self._load_sample_data()
        except Exception as e:
            self.logger.error(f"Failed to load initial data: {e}")
            self._load_sample_data()

    def _load_sample_data(self):
        """Charge des données d'exemple"""
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
        prices = np.random.randn(1000).cumsum() + 100
        volumes = np.random.randint(1, 100, 1000)
        symbol = st.session_state.selected_symbol
        self.market_data[symbol] = MarketData(
            symbol=symbol,
            timeframe="1m",
            open_prices=pd.Series(prices),
            high_prices=pd.Series(prices * 1.01),
            low_prices=pd.Series(prices * 0.99),
            close_prices=pd.Series(prices),
            volume=pd.Series(volumes),
            timestamp=pd.Series(dates),
        )
        self.calculate_indicators(symbol)

    def calculate_indicators(self, symbol: str):
        """Calcul des indicateurs techniques"""
        data = self.market_data[symbol]

        # Vérification de la présence de toutes les séries nécessaires
        required_attrs = [
            "open_prices",
            "high_prices",
            "low_prices",
            "close_prices",
            "volume",
        ]
        for attr in required_attrs:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                print(
                    f"DEBUG: Attribut {attr} manquant ou vide pour {symbol} dans data: {data}"
                )
                return  # Ou lève une exception ou retourne un dict vide

        df = pd.DataFrame(
            {
                "open": data.open_prices,
                "high": data.high_prices,
                "low": data.low_prices,
                "close": data.close_prices,
                "volume": data.volume,
            }
        )
        indicators = TechnicalIndicators(df)
        self.indicators[symbol] = indicators.calculate_all_indicators()

    def update_market_analysis(self, symbol: str = None, timeframe: str = None):
        if symbol is None:
            symbol = st.session_state.get("selected_symbol", "BTC/USDT")
        if timeframe is None:
            timeframe = st.session_state.get("selected_timeframe", "1m")

        try:
            if self.exchange:
                # Recharge les données OHLCV pour le symbole/timeframe
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                self.market_data[symbol] = MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open_prices=df["open"],
                    high_prices=df["high"],
                    low_prices=df["low"],
                    close_prices=df["close"],
                    volume=df["volume"],
                    timestamp=df["timestamp"],
                )
                self.calculate_indicators(symbol)
                self.logger.info(f"Market analysis updated for {symbol} ({timeframe})")
            else:
                self.logger.warning(
                    "Exchange not initialized, using sample data for update_market_analysis"
                )
                self._load_sample_data()
        except Exception as e:
            self.logger.error(f"Failed to update market analysis for {symbol}: {e}")
            self._load_sample_data()

    async def start_websocket(self, symbol: str):
        """Démarrage de la connexion WebSocket"""
        url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_1m"
        async with websockets.connect(url) as ws:
            self.logger.info(f"WebSocket connected for {symbol}")
            try:
                while True:
                    msg = await ws.recv()
                    await self.process_websocket_message(json.loads(msg))
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")

    async def process_websocket_message(self, msg):
        """Traitement des messages WebSocket"""
        if msg.get("e") == "kline":
            k = msg["k"]
            symbol = msg["s"]
            new_data = pd.Series(
                {
                    "timestamp": pd.to_datetime(k["t"], unit="ms"),
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"]),
                }
            )
            if symbol not in self.market_data:
                self.market_data[symbol] = MarketData(
                    symbol=symbol,
                    timeframe="1m",
                    open_prices=pd.Series([new_data["open"]]),
                    high_prices=pd.Series([new_data["high"]]),
                    low_prices=pd.Series([new_data["low"]]),
                    close_prices=pd.Series([new_data["close"]]),
                    volume=pd.Series([new_data["volume"]]),
                    timestamp=pd.Series([new_data["timestamp"]]),
                )
            else:
                data = self.market_data[symbol]
                data.open_prices = pd.concat(
                    [data.open_prices, pd.Series([new_data["open"]])]
                )
                data.high_prices = pd.concat(
                    [data.high_prices, pd.Series([new_data["high"]])]
                )
                data.low_prices = pd.concat(
                    [data.low_prices, pd.Series([new_data["low"]])]
                )
                data.close_prices = pd.concat(
                    [data.close_prices, pd.Series([new_data["close"]])]
                )
                data.volume = pd.concat([data.volume, pd.Series([new_data["volume"]])])
                data.timestamp = pd.concat(
                    [data.timestamp, pd.Series([new_data["timestamp"]])]
                )
            self.calculate_indicators(symbol)

    def render_dashboard(self):
        """Rendu du dashboard complet"""
        st.title("🤖 Trading Bot Ultimate Dashboard")
        # Sidebar
        self.render_sidebar()
        # Métriques principales
        self.render_metrics()
        # Layout principal
        col1, col2 = st.columns([2, 1])
        with col1:
            self.render_main_chart()
        with col2:
            self.render_positions()
            self.render_orderbook()
            self.render_trades_history()

    def render_sidebar(self):
        """Rendu de la sidebar"""
        with st.sidebar:
            st.title("Contrôles")
            # Sélection du symbol
            st.session_state.selected_symbol = st.selectbox(
                "Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], index=0
            )
            # Sélection du timeframe
            st.session_state.selected_timeframe = st.selectbox(
                "Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=0
            )
            # Sélection des indicateurs
            st.subheader("Indicateurs")
            show_sma = st.checkbox("SMA", value=True)
            show_ema = st.checkbox("EMA", value=True)
            show_bb = st.checkbox("Bollinger Bands", value=True)
            show_macd = st.checkbox("MACD", value=True)
            show_rsi = st.checkbox("RSI", value=True)
            # Stats système
            st.subheader("Stats Système")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CPU", f"{psutil.cpu_percent()}%")
            with col2:
                st.metric("RAM", f"{psutil.virtual_memory().percent}%")
            st.metric(
                "Uptime",
            )

    def render_metrics(self):
        """Rendu des métriques principales"""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "PnL Total", f"${self.performance_metrics['pnl']:,.2f}", delta="↑ 2.5%"
            )
        with col2:
            st.metric(
                "Win Rate",
                f"{self.performance_metrics['win_rate']:.1%}",
                delta="↑ 1.2%",
            )
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{self.performance_metrics['sharpe_ratio']:.2f}",
                delta="↑ 0.1",
            )
        with col4:
            st.metric(
                "Max Drawdown",
                f"{self.performance_metrics['max_drawdown']:.1%}",
                delta="↓ 0.5%",
            )

    def render_main_chart(self):
        """Rendu du graphique principal"""
        st.subheader("Graphique Principal")
        symbol = st.session_state.selected_symbol
        if symbol in self.market_data:
            data = self.market_data[symbol]
            fig = make_subplots.make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=("Prix", "Volume", "Indicateurs"),
            )
            # Chandelles
            fig.add_trace(
                go.Candlestick(
                    x=data.timestamp,
                    open=data.open_prices,
                    high=data.high_prices,
                    low=data.low_prices,
                    close=data.close_prices,
                    name=symbol,
                ),
                row=1,
                col=1,
            )
            # Volume
            fig.add_trace(
                go.Bar(x=data.timestamp, y=data.volume, name="Volume"), row=2, col=1
            )
            # RSI
            if (
                st.session_state.get("show_rsi", True)
                and "rsi" in self.indicators[symbol]
            ):
                fig.add_trace(
                    go.Scatter(
                        x=data.timestamp, y=self.indicators[symbol]["rsi"], name="RSI"
                    ),
                    row=3,
                    col=1,
                )
            # Layout
            fig.update_layout(
                height=800,
                xaxis_title="Date",
                yaxis_title="Prix",
                template="plotly_dark",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Pas de données disponibles pour {symbol}")

    def render_positions(self):
        """Rendu des positions actuelles"""
        st.subheader("Positions Actuelles")
        if self.positions:
            df_positions = pd.DataFrame(self.positions).T
            st.dataframe(df_positions)
        else:
            st.info("Pas de positions ouvertes")

    def render_orderbook(self):
        """Rendu du carnet d'ordres"""
        st.subheader("Carnet d'Ordres")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Achats (Bids)")
            df_bids = pd.DataFrame(
                [
                    {"Prix": 35000, "Quantité": 1.5},
                    {"Prix": 34990, "Quantité": 2.1},
                    {"Prix": 34980, "Quantité": 1.8},
                ]
            )
            st.dataframe(df_bids)
        with col2:
            st.write("Ventes (Asks)")
            df_asks = pd.DataFrame(
                [
                    {"Prix": 35010, "Quantité": 1.2},
                    {"Prix": 35020, "Quantité": 1.9},
                    {"Prix": 35030, "Quantité": 1.6},
                ]
            )
            st.dataframe(df_asks)

    def render_trades_history(self):
        """Rendu de l'historique des trades"""
        st.subheader("Historique des Trades")
        if self.trades_history:
            df_trades = pd.DataFrame(self.trades_history)
            st.dataframe(df_trades)
        else:
            st.info("Pas d'historique de trades")

    def run(self):
        """Point d'entrée principal du dashboard"""
        try:
            self.render_dashboard()
            # Démarrage du WebSocket en arrière-plan
            if not hasattr(self, "ws_running"):
                symbol = st.session_state.selected_symbol.lower().replace("/", "")
                asyncio.run(self.start_websocket(symbol))
                self.ws_running = True
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
            st.error(f"Une erreur s'est produite: {e}")


if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
