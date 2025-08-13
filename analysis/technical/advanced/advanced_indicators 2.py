import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import talib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AdvancedIndicators:
    """Classe pour le calcul d'indicateurs techniques avancés"""

    def __init__(self):
        """Initialise les paramètres des indicateurs"""
        self.indicators = {
            "trend": {
                "supertrend": self._supertrend,
                "vwma": self._vwma,
                "kama": self._kama,
                "psar": self._psar,
                "trix": self._trix,
            },
            "momentum": {
                "ao": self._awesome_oscillator,
                "williams_r": self._williams_r,
                "cci": self._cci,
            },
            "volatility": {
                "parkinson": self._parkinson,
                "yang_zhang": self._yang_zhang,
            },
            "volume": {
                "accumulation": self._accumulation,
                "cmf": self._chaikin_money_flow,
                "eom": self._ease_of_movement,
            },
            "orderflow": {
                "delta_volume": self._delta_volume,
                "imbalance": self._imbalance,
                "smart_money_index": self._smart_money_index,
                "liquidity_wave": self._liquidity_wave,
                "bid_ask_ratio": self._bid_ask_ratio,
            },
        }

    def _supertrend(
        self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> Dict:
        """Calcule l'indicateur Supertrend"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]

            # Calcul de l'ATR
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join="inner").max(axis=1)
            atr = tr.ewm(span=period, min_periods=period).mean()

            # Calcul des bandes
            hl2 = (high + low) / 2
            final_upperband = hl2 + (multiplier * atr)
            final_lowerband = hl2 - (multiplier * atr)

            supertrend = pd.Series(index=data.index, dtype=float)
            direction = pd.Series(index=data.index, dtype=int)

            for i in range(period, len(data)):
                if close[i] > final_upperband[i - 1]:
                    supertrend[i] = final_lowerband[i]
                    direction[i] = 1
                elif close[i] < final_lowerband[i - 1]:
                    supertrend[i] = final_upperband[i]
                    direction[i] = -1
                else:
                    supertrend[i] = supertrend[i - 1]
                    direction[i] = direction[i - 1]

            return {
                "value": supertrend,
                "direction": direction,
                "strength": abs(close - supertrend) / close,
            }
        except Exception as e:
            logger.error(f"Erreur calcul Supertrend: {e}")
            return None

    def _ichimoku(self, data: pd.DataFrame) -> Dict:
        """Calcule l'indicateur Ichimoku"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]

            # Paramètres par défaut
            tenkan_period = 9
            kijun_period = 26
            senkou_span_b_period = 52

            # Tenkan-sen (Conversion Line)
            tenkan_sen = pd.Series(dtype=float)
            for i in range(tenkan_period, len(data)):
                high_val = high[i - tenkan_period : i].max()
                low_val = low[i - tenkan_period : i].min()
                tenkan_sen[i] = (high_val + low_val) / 2

            # Kijun-sen (Base Line)
            kijun_sen = pd.Series(dtype=float)
            for i in range(kijun_period, len(data)):
                high_val = high[i - kijun_period : i].max()
                low_val = low[i - kijun_period : i].min()
                kijun_sen[i] = (high_val + low_val) / 2

            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

            # Senkou Span B (Leading Span B)
            senkou_span_b = pd.Series(dtype=float)
            for i in range(senkou_span_b_period, len(data)):
                high_val = high[i - senkou_span_b_period : i].max()
                low_val = low[i - senkou_span_b_period : i].min()
                senkou_span_b[i] = ((high_val + low_val) / 2).shift(kijun_period)

            # Chikou Span (Lagging Span)
            chikou_span = close.shift(-kijun_period)

            return {
                "tenkan": tenkan_sen,
                "kijun": kijun_sen,
                "senkou_a": senkou_span_a,
                "senkou_b": senkou_span_b,
                "chikou": chikou_span,
                "cloud_strength": abs(senkou_span_a - senkou_span_b) / close,
            }

        except Exception as e:
            logger.error(f"Erreur calcul Ichimoku: {e}")
            return None

    def _trix(self, data: pd.DataFrame, period: int = 15) -> pd.Series:
        close = data["close"]
        ema1 = close.ewm(span=period, min_periods=period).mean()
        ema2 = ema1.ewm(span=period, min_periods=period).mean()
        ema3 = ema2.ewm(span=period, min_periods=period).mean()
        trix = ema3.pct_change() * 100
        return trix

    def _vwma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calcule la moyenne mobile pondérée par volume"""
        try:
            return (data["close"] * data["volume"]).rolling(period).sum() / data[
                "volume"
            ].rolling(period).sum()
        except Exception as e:
            logger.error(f"Erreur calcul VWMA: {e}")
            return None

    def _kama(
        self, data: pd.DataFrame, period: int = 20, fast: int = 2, slow: int = 30
    ) -> pd.Series:
        """Calcule la moyenne mobile adaptative de Kaufman"""
        try:
            close = data["close"]
            change = abs(close - close.shift(period))
            volatility = pd.Series(dtype=float)

            for i in range(period, len(close)):
                vol = abs(close[i] - close[i - 1]).sum()
                volatility[i] = vol

            er = change / volatility
            fast_sc = 2 / (fast + 1)
            slow_sc = 2 / (slow + 1)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            kama = pd.Series(index=close.index, dtype=float)
            kama[period - 1] = close[period - 1]

            for i in range(period, len(close)):
                kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])

            return kama

        except Exception as e:
            logger.error(f"Erreur calcul KAMA: {e}")
            return None

    def _psar(self, data: pd.DataFrame, iaf: float = 0.02, maxaf: float = 0.2) -> Dict:
        """Calcule l'indicateur Parabolic SAR"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]

            psar = pd.Series(index=data.index, dtype=float)
            trend = pd.Series(index=data.index, dtype=int)
            af = pd.Series(index=data.index, dtype=float)
            ep = pd.Series(index=data.index, dtype=float)

            psar[0] = high[0]
            trend[0] = 1
            af[0] = iaf
            ep[0] = low[0]

            for i in range(1, len(data)):
                if trend[i - 1] == 1:
                    psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])

                    if low[i] < psar[i]:
                        trend[i] = -1
                        psar[i] = ep[i - 1]
                        af[i] = iaf
                        ep[i] = low[i]
                    else:
                        trend[i] = 1
                        if high[i] > ep[i - 1]:
                            ep[i] = high[i]
                            af[i] = min(af[i - 1] + iaf, maxaf)
                        else:
                            ep[i] = ep[i - 1]
                            af[i] = af[i - 1]
                else:
                    psar[i] = psar[i - 1] - af[i - 1] * (psar[i - 1] - ep[i - 1])

                    if high[i] > psar[i]:
                        trend[i] = 1
                        psar[i] = ep[i - 1]
                        af[i] = iaf
                        ep[i] = high[i]
                    else:
                        trend[i] = -1
                        if low[i] < ep[i - 1]:
                            ep[i] = low[i]
                            af[i] = min(af[i - 1] + iaf, maxaf)
                        else:
                            ep[i] = ep[i - 1]
                            af[i] = af[i - 1]

            return {
                "value": psar,
                "trend": trend,
                "strength": abs(close - psar) / close,
            }

        except Exception as e:
            logger.error(f"Erreur calcul PSAR: {e}")
            return None

    # Ajoutez ici les autres méthodes (_trix, _awesome_oscillator, etc.)

    def analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Analyse tous les indicateurs pour un timeframe donné"""
        try:
            results = {}
            for category, indicators in self.indicators.items():
                results[category] = {}
                for name, func in indicators.items():
                    results[category][name] = func(data)
            return results
        except Exception as e:
            logger.error(f"Erreur analyse timeframe {timeframe}: {e}")
            return None
