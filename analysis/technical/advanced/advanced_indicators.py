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

            tenkan_period = 9
            kijun_period = 26
            senkou_span_b_period = 52

            tenkan_sen = pd.Series(dtype=float)
            for i in range(tenkan_period, len(data)):
                high_val = high[i - tenkan_period : i].max()
                low_val = low[i - tenkan_period : i].min()
                tenkan_sen[i] = (high_val + low_val) / 2

            kijun_sen = pd.Series(dtype=float)
            for i in range(kijun_period, len(data)):
                high_val = high[i - kijun_period : i].max()
                low_val = low[i - kijun_period : i].min()
                kijun_sen[i] = (high_val + low_val) / 2

            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

            senkou_span_b = pd.Series(dtype=float)
            for i in range(senkou_span_b_period, len(data)):
                high_val = high[i - senkou_span_b_period : i].max()
                low_val = low[i - senkou_span_b_period : i].min()
                senkou_span_b[i] = ((high_val + low_val) / 2).shift(kijun_period)

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

    def _awesome_oscillator(
        self, data: pd.DataFrame, fast: int = 5, slow: int = 34
    ) -> pd.Series:
        hl2 = (data["high"] + data["low"]) / 2
        ao = hl2.rolling(window=fast).mean() - hl2.rolling(window=slow).mean()
        return ao

    def _trix(self, data: pd.DataFrame, period: int = 15) -> pd.Series:
        close = data["close"]
        ema1 = close.ewm(span=period, min_periods=period).mean()
        ema2 = ema1.ewm(span=period, min_periods=period).mean()
        ema3 = ema2.ewm(span=period, min_periods=period).mean()
        trix = ema3.pct_change() * 100
        return trix

    def _vwma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
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

    # --- Momentum ---

    def _williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            willr = -100 * (highest_high - close) / (highest_high - lowest_low)
            return willr
        except Exception as e:
            logger.error(f"Erreur calcul Williams %R: {e}")
            return pd.Series(np.nan, index=data.index)

    def _cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        try:
            tp = (data["high"] + data["low"] + data["close"]) / 3
            cci = (tp - tp.rolling(window=period).mean()) / (
                0.015 * tp.rolling(window=period).std()
            )
            return cci
        except Exception as e:
            logger.error(f"Erreur calcul CCI: {e}")
            return pd.Series(np.nan, index=data.index)

    # --- Volatility ---

    def _parkinson(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        try:
            high = data["high"]
            low = data["low"]
            ln_hl = np.log(high / low)
            parkinson = (1 / (4 * period * np.log(2))) * (ln_hl**2).rolling(
                period
            ).sum()
            return parkinson
        except Exception as e:
            logger.error(f"Erreur calcul Parkinson volatility: {e}")
            return pd.Series(np.nan, index=data.index)

    def _yang_zhang(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        try:
            open_ = data["open"]
            close = data["close"]
            high = data["high"]
            low = data["low"]
            k = 0.34 / (1.34 + (period + 1) / (period - 1))
            log_ho = np.log(high / open_)
            log_lo = np.log(low / open_)
            log_co = np.log(close / open_)
            rs = (log_ho * log_lo).rolling(window=period).mean()
            open_vol = (np.log(open_ / close.shift(1))) ** 2
            close_vol = (np.log(close / open_)) ** 2
            yang_zhang = (
                open_vol.rolling(window=period).mean()
                + k * close_vol.rolling(window=period).mean()
                + (1 - k) * rs
            )
            return yang_zhang
        except Exception as e:
            logger.error(f"Erreur calcul Yang-Zhang volatility: {e}")
            return pd.Series(np.nan, index=data.index)

    # --- Volume ---

    def _accumulation(self, data: pd.DataFrame) -> pd.Series:
        try:
            close = data["close"]
            low = data["low"]
            high = data["high"]
            volume = data["volume"]
            money_flow = (
                ((close - low) - (high - close)) / (high - low + 1e-10) * volume
            )
            acc = money_flow.cumsum()
            return acc
        except Exception as e:
            logger.error(f"Erreur calcul Accumulation/Distribution: {e}")
            return pd.Series(np.nan, index=data.index)

    def _chaikin_money_flow(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            volume = data["volume"]
            mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
            mf_volume = mf_multiplier * volume
            cmf = (
                mf_volume.rolling(window=period).sum()
                / volume.rolling(window=period).sum()
            )
            return cmf
        except Exception as e:
            logger.error(f"Erreur calcul Chaikin Money Flow: {e}")
            return pd.Series(np.nan, index=data.index)

    def _ease_of_movement(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            high = data["high"]
            low = data["low"]
            volume = data["volume"]
            distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
            box_ratio = volume / (high - low + 1e-10)
            eom = distance_moved / box_ratio
            eom = eom.rolling(window=period).mean()
            return eom
        except Exception as e:
            logger.error(f"Erreur calcul Ease of Movement: {e}")
            return pd.Series(np.nan, index=data.index)

    # --- Orderflow ---

    def _delta_volume(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["volume"].diff()
        except Exception as e:
            logger.error(f"Erreur calcul Delta Volume: {e}")
            return pd.Series(np.nan, index=data.index)

    def _imbalance(self, data: pd.DataFrame) -> pd.Series:
        try:
            # Placeholder: you should implement real orderbook imbalance if you have bid/ask
            # Here, proxy using up/down volume
            close = data["close"]
            volume = data["volume"]
            imbalance = np.where(close.diff() > 0, volume, -volume)
            return pd.Series(imbalance, index=data.index)
        except Exception as e:
            logger.error(f"Erreur calcul Imbalance: {e}")
            return pd.Series(np.nan, index=data.index)

    def _smart_money_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            close = data["close"]
            open_ = data["open"]
            smi = close - open_
            smi = smi.rolling(window=period).sum()
            return smi
        except Exception as e:
            logger.error(f"Erreur calcul Smart Money Index: {e}")
            return pd.Series(np.nan, index=data.index)

    def _liquidity_wave(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        try:
            # Proxy: high-low relative to volume, as liquidity wave
            high = data["high"]
            low = data["low"]
            volume = data["volume"]
            lw = (high - low) / (volume + 1e-10)
            lw = lw.rolling(window=period).mean()
            return lw
        except Exception as e:
            logger.error(f"Erreur calcul Liquidity Wave: {e}")
            return pd.Series(np.nan, index=data.index)

    def _bid_ask_ratio(self, data: pd.DataFrame) -> float:
        # This is a placeholder, real bid/ask needs orderbook
        try:
            # Use up-volume/down-volume as proxy
            close = data["close"]
            volume = data["volume"]
            buy_volume = volume[close.diff() > 0].sum()
            sell_volume = volume[close.diff() < 0].sum()
            total = buy_volume + abs(sell_volume)
            if total == 0:
                return 0.5
            return float(buy_volume / total)
        except Exception as e:
            logger.error(f"Erreur calcul Bid/Ask Ratio: {e}")
            return 0.5

    def analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict:
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
