import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

# self.logger = logging.getLogger(__name__)  # Commenté car hors classe


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calcule le RSI d'une série de prix."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def compute_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, float]:
    """Calcule le MACD et la ligne de signal."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return {
        "macd": macd.iloc[-1],
        "signal": signal_line.iloc[-1],
        "histogram": (macd - signal_line).iloc[-1],
    }


def trend_signal(prices: pd.Series, short_win: int = 20, long_win: int = 50) -> str:
    """Détermine le signal de tendance en croisant les moyennes mobiles."""
    sma_short = prices.rolling(window=short_win).mean()
    sma_long = prices.rolling(window=long_win).mean()
    if sma_short.iloc[-1] > sma_long.iloc[-1]:
        return "bullish"
    elif sma_short.iloc[-1] < sma_long.iloc[-1]:
        return "bearish"
    else:
        return "neutral"


def volatility_signal(prices: pd.Series, window: int = 20) -> float:
    """Calcule la volatilité (écart-type) sur la période donnée."""
    return prices.pct_change().rolling(window=window).std().iloc[-1]


def volume_signal(volume: pd.Series, window: int = 20) -> str:
    """Compare le volume actuel à la moyenne mobile du volume."""
    avg_vol = volume.rolling(window=window).mean()
    if volume.iloc[-1] > avg_vol.iloc[-1]:
        return "high"
    else:
        return "normal"


def momentum_signal(prices: pd.Series, window: int = 14) -> float:
    """Calcule la variation de prix sur la fenêtre donnée."""
    return prices.iloc[-1] - prices.iloc[-window]


def analyze_signals(
    market_data: pd.DataFrame, indicators: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyse les signaux techniques sur le marché et retourne une synthèse exploitable par le bot.
    Args:
        market_data: DataFrame avec colonnes ['close', 'volume'] (et éventuellement 'open', 'high', 'low')
        indicators: dictionnaire d'indicateurs externes déjà calculés (optionnel)
    Returns:
        Dict synthétique des signaux (trend, rsi, macd, volatility, volume, momentum, recommendation)
    """
    signals = {}

    if market_data is None or len(market_data) < 50:
        self.logger.warning("Pas assez de données pour analyse des signaux.")
        return {}

    close = market_data["close"]
    volume = market_data["volume"]

    # RSI
    signals["rsi"] = compute_rsi(close)
    # MACD
    macd_res = compute_macd(close)
    signals.update(macd_res)
    # Trend
    signals["trend"] = trend_signal(close)
    # Volatility
    signals["volatility"] = volatility_signal(close)
    # Volume
    signals["volume"] = volume_signal(volume)
    # Momentum
    signals["momentum"] = momentum_signal(close)

    # Recommandation simple
    recommendation = "hold"
    if (
        signals["trend"] == "bullish"
        and signals["macd"] > signals["signal"]
        and signals["rsi"] < 70
    ):
        recommendation = "buy"
    elif (
        signals["trend"] == "bearish"
        and signals["macd"] < signals["signal"]
        and signals["rsi"] > 30
    ):
        recommendation = "sell"
    signals["recommendation"] = recommendation

    return signals


async def async_analyze_signals(
    market_data: pd.DataFrame, indicators: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Version asynchrone d'analyse des signaux (pour intégration dans async bot).
    """
    return analyze_signals(market_data, indicators)


# Optionnel : helpers pour formatage
def format_signals_report(signals: Dict[str, Any]) -> str:
    """Génère un rapport textuel lisible à partir du dict de signaux."""
    if not signals:
        return "Aucun signal disponible."
    lines = [
        f"Tendance : {signals.get('trend')}",
        f"RSI : {signals.get('rsi'):.2f}",
        f"MACD : {signals.get('macd'):.4f} | Signal : {signals.get('signal'):.4f}",
        f"Momentum : {signals.get('momentum'):.2f}",
        f"Volatilité : {signals.get('volatility'):.4%}",
        f"Volume : {signals.get('volume')}",
        f"Recommandation : **{signals.get('recommendation').upper()}**",
    ]
    return "\n".join(lines)
