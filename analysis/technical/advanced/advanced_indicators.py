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
                "ema": self._ema,
                "sma": self._sma,
                "donchian": self._donchian,
                "ichimoku": self._ichimoku,
            },
            "momentum": {
                "ao": self._awesome_oscillator,
                "williams_r": self._williams_r,
                "cci": self._cci,
                "rsi": self._rsi,
                "macd": self._macd,
                "stoch": self._stoch,
            },
            "volatility": {
                "parkinson": self._parkinson,
                "yang_zhang": self._yang_zhang,
                "bbands": self._bbands,
                "atr": self._atr,
                "kc": self._kc,
            },
            "volume": {
                "accumulation": self._accumulation,
                "cmf": self._chaikin_money_flow,
                "eom": self._ease_of_movement,
                "obv": self._obv,
                "vwap": self._vwap,
            },
            "orderflow": {
                "delta_volume": self._delta_volume,
                "imbalance": self._imbalance,
                "smart_money_index": self._smart_money_index,
                "liquidity_wave": self._liquidity_wave,
                "bid_ask_ratio": self._bid_ask_ratio,
            },
        }

    def _debug_nan(self, df: pd.DataFrame, name: str):
        """Affiche le nombre de NaN, de valeurs infinies, et la taille du DataFrame pour le debug."""
        print(f"[DEBUG NaN] {name}: len={len(df)}")
        for col in df.columns:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(df[col]).sum()
            print(f"  - {col}: NaN={n_nan} | Inf={n_inf}")

    # --- Trend indicators ---
    def _supertrend(
        self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> Dict:
        self._debug_nan(data, "Supertrend (entrée)")
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join="inner").max(axis=1)
            atr = tr.ewm(span=period, min_periods=period).mean()
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
            print(
                f"[DEBUG NaN] Supertrend (sortie): NaN={supertrend.isna().sum()} | Inf={np.isinf(supertrend).sum()}"
            )
            return {
                "value": supertrend,
                "direction": direction,
                "strength": abs(close - supertrend) / close,
            }
        except Exception as e:
            logger.error(f"Erreur calcul Supertrend: {e}")
            return None

    def _vwma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        self._debug_nan(data, "VWMA (entrée)")
        try:
            result = (data["close"] * data["volume"]).rolling(period).sum() / data[
                "volume"
            ].rolling(period).sum()
            print(
                f"[DEBUG NaN] VWMA (sortie): NaN={result.isna().sum()} | Inf={np.isinf(result).sum()}"
            )
            return result
        except Exception as e:
            logger.error(f"Erreur calcul VWMA: {e}")
            return pd.Series(np.nan, index=data.index)

    def _kama(
        self, data: pd.DataFrame, period: int = 20, fast: int = 2, slow: int = 30
    ) -> pd.Series:
        self._debug_nan(data, "KAMA (entrée)")
        try:
            close = np.array(data["close"], dtype=float)
            close = np.nan_to_num(
                close,
                nan=np.nanmean(close),
                posinf=np.nanmax(close),
                neginf=np.nanmin(close),
            )
            change = np.abs(close - np.roll(close, period))
            volatility = np.zeros_like(close)
            for i in range(period, len(close)):
                volatility[i] = np.sum(
                    np.abs(
                        close[i - period + 1 : i + 1]
                        - np.roll(close[i - period : i], 1)
                    )
                )
            volatility[volatility == 0] = 1e-8
            er = change / volatility
            fast_sc = 2 / (fast + 1)
            slow_sc = 2 / (slow + 1)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            sc = np.clip(sc, 0, 1)
            kama = np.zeros_like(close)
            kama[:period] = close[:period]
            for i in range(period, len(close)):
                try:
                    kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
                    if np.isnan(kama[i]) or np.isinf(kama[i]):
                        kama[i] = close[i]
                except Exception:
                    kama[i] = close[i]
            kama_series = pd.Series(kama, index=data.index)
            print(
                f"[DEBUG NaN] KAMA (sortie): NaN={kama_series.isna().sum()} | Inf={np.isinf(kama_series).sum()}"
            )
            return kama_series
        except Exception as e:
            logger.error(f"Erreur calcul KAMA: {e}")
            return pd.Series(np.nan, index=data.index)

    def _psar(self, data: pd.DataFrame, iaf: float = 0.02, maxaf: float = 0.2) -> Dict:
        self._debug_nan(data, "PSAR (entrée)")
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
            print(
                f"[DEBUG NaN] PSAR (sortie): NaN={psar.isna().sum()} | Inf={np.isinf(psar).sum()}"
            )
            return {
                "value": psar,
                "trend": trend,
                "strength": abs(close - psar) / close,
            }
        except Exception as e:
            logger.error(f"Erreur calcul PSAR: {e}")
            return None

    def _trix(self, data: pd.DataFrame, period: int = 15) -> pd.Series:
        self._debug_nan(data, "TRIX (entrée)")
        close = data["close"]
        ema1 = close.ewm(span=period, min_periods=period).mean()
        ema2 = ema1.ewm(span=period, min_periods=period).mean()
        ema3 = ema2.ewm(span=period, min_periods=period).mean()
        trix = ema3.pct_change() * 100
        print(
            f"[DEBUG NaN] TRIX (sortie): NaN={trix.isna().sum()} | Inf={np.isinf(trix).sum()}"
        )
        return trix

    def _ema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        self._debug_nan(data, "EMA (entrée)")
        try:
            result = data["close"].ewm(span=period, min_periods=period).mean()
            print(
                f"[DEBUG NaN] EMA (sortie): NaN={result.isna().sum()} | Inf={np.isinf(result).sum()}"
            )
            return result
        except Exception as e:
            logger.error(f"Erreur calcul EMA: {e}")
            return pd.Series(np.nan, index=data.index)

    def _sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        self._debug_nan(data, "SMA (entrée)")
        try:
            result = data["close"].rolling(window=period).mean()
            print(
                f"[DEBUG NaN] SMA (sortie): NaN={result.isna().sum()} | Inf={np.isinf(result).sum()}"
            )
            return result
        except Exception as e:
            logger.error(f"Erreur calcul SMA: {e}")
            return pd.Series(np.nan, index=data.index)

    def _donchian(self, data: pd.DataFrame, period: int = 20) -> Dict:
        self._debug_nan(data, "Donchian (entrée)")
        try:
            high = data["high"].rolling(window=period).max()
            low = data["low"].rolling(window=period).min()
            print(
                f"[DEBUG NaN] Donchian high: NaN={high.isna().sum()} | Inf={np.isinf(high).sum()}"
            )
            print(
                f"[DEBUG NaN] Donchian low: NaN={low.isna().sum()} | Inf={np.isinf(low).sum()}"
            )
            return {"high": high, "low": low}
        except Exception as e:
            logger.error(f"Erreur calcul Donchian: {e}")
            return {
                "high": pd.Series(np.nan, index=data.index),
                "low": pd.Series(np.nan, index=data.index),
            }

    def _ichimoku(self, data: pd.DataFrame) -> Dict:
        self._debug_nan(data, "Ichimoku (entrée)")
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
            print(
                f"[DEBUG NaN] Ichimoku tenkan: NaN={tenkan_sen.isna().sum()} | Inf={np.isinf(tenkan_sen).sum()}"
            )
            print(
                f"[DEBUG NaN] Ichimoku kijun: NaN={kijun_sen.isna().sum()} | Inf={np.isinf(kijun_sen).sum()}"
            )
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

    # --- Momentum indicators ---
    def _awesome_oscillator(
        self, data: pd.DataFrame, fast: int = 5, slow: int = 34
    ) -> pd.Series:
        self._debug_nan(data, "AO (entrée)")
        hl2 = (data["high"] + data["low"]) / 2
        ao = hl2.rolling(window=fast).mean() - hl2.rolling(window=slow).mean()
        print(
            f"[DEBUG NaN] AO (sortie): NaN={ao.isna().sum()} | Inf={np.isinf(ao).sum()}"
        )
        return ao

    def _williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        self._debug_nan(data, "Williams_R (entrée)")
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            willr = -100 * (highest_high - close) / (highest_high - lowest_low)
            print(
                f"[DEBUG NaN] Williams_R (sortie): NaN={willr.isna().sum()} | Inf={np.isinf(willr).sum()}"
            )
            return willr
        except Exception as e:
            logger.error(f"Erreur calcul Williams %R: {e}")
            return pd.Series(np.nan, index=data.index)

    def _cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        self._debug_nan(data, "CCI (entrée)")
        try:
            tp = (data["high"] + data["low"] + data["close"]) / 3
            cci = (tp - tp.rolling(window=period).mean()) / (
                0.015 * tp.rolling(window=period).std()
            )
            print(
                f"[DEBUG NaN] CCI (sortie): NaN={cci.isna().sum()} | Inf={np.isinf(cci).sum()}"
            )
            return cci
        except Exception as e:
            logger.error(f"Erreur calcul CCI: {e}")
            return pd.Series(np.nan, index=data.index)

    def _rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        self._debug_nan(data, "RSI (entrée)")
        try:
            close = data["close"]
            result = pd.Series(
                talib.RSI(close.values, timeperiod=period), index=data.index
            )
            print(
                f"[DEBUG NaN] RSI (sortie): NaN={result.isna().sum()} | Inf={np.isinf(result).sum()}"
            )
            return result
        except Exception as e:
            logger.error(f"Erreur calcul RSI: {e}")
            return pd.Series(np.nan, index=data.index)

    def _macd(
        self,
        data: pd.DataFrame,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> Dict:
        self._debug_nan(data, "MACD (entrée)")
        try:
            close = data["close"]
            macd, macdsignal, macdhist = talib.MACD(
                close.values,
                fastperiod=fastperiod,
                slowperiod=slowperiod,
                signalperiod=signalperiod,
            )
            macd_series = pd.Series(macd, index=data.index)
            macdsignal_series = pd.Series(macdsignal, index=data.index)
            macdhist_series = pd.Series(macdhist, index=data.index)
            print(
                f"[DEBUG NaN] MACD macd: NaN={macd_series.isna().sum()} | Inf={np.isinf(macd_series).sum()}"
            )
            print(
                f"[DEBUG NaN] MACD signal: NaN={macdsignal_series.isna().sum()} | Inf={np.isinf(macdsignal_series).sum()}"
            )
            print(
                f"[DEBUG NaN] MACD hist: NaN={macdhist_series.isna().sum()} | Inf={np.isinf(macdhist_series).sum()}"
            )
            return {
                "macd": macd_series,
                "macd_signal": macdsignal_series,
                "macd_hist": macdhist_series,
            }
        except Exception as e:
            logger.error(f"Erreur calcul MACD: {e}")
            return {
                "macd": pd.Series(np.nan, index=data.index),
                "macd_signal": pd.Series(np.nan, index=data.index),
                "macd_hist": pd.Series(np.nan, index=data.index),
            }

    def _stoch(
        self,
        data: pd.DataFrame,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> Dict:
        self._debug_nan(data, "Stoch (entrée)")
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            slowk, slowd = talib.STOCH(
                high.values,
                low.values,
                close.values,
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowd_period=slowd_period,
            )
            slowk_series = pd.Series(slowk, index=data.index)
            slowd_series = pd.Series(slowd, index=data.index)
            print(
                f"[DEBUG NaN] Stoch slowk: NaN={slowk_series.isna().sum()} | Inf={np.isinf(slowk_series).sum()}"
            )
            print(
                f"[DEBUG NaN] Stoch slowd: NaN={slowd_series.isna().sum()} | Inf={np.isinf(slowd_series).sum()}"
            )
            return {
                "slowk": slowk_series,
                "slowd": slowd_series,
            }
        except Exception as e:
            logger.error(f"Erreur calcul Stochastique: {e}")
            return {
                "slowk": pd.Series(np.nan, index=data.index),
                "slowd": pd.Series(np.nan, index=data.index),
            }

    # --- Volatility indicators ---
    def _parkinson(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        self._debug_nan(data, "Parkinson (entrée)")
        try:
            high = data["high"]
            low = data["low"]
            ln_hl = np.log(high / low)
            parkinson = (1 / (4 * period * np.log(2))) * (ln_hl**2).rolling(
                period
            ).sum()
            print(
                f"[DEBUG NaN] Parkinson (sortie): NaN={parkinson.isna().sum()} | Inf={np.isinf(parkinson).sum()}"
            )
            return parkinson
        except Exception as e:
            logger.error(f"Erreur calcul Parkinson volatility: {e}")
            return pd.Series(np.nan, index=data.index)

    def _yang_zhang(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        self._debug_nan(data, "YangZhang (entrée)")
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
            print(
                f"[DEBUG NaN] YangZhang (sortie): NaN={yang_zhang.isna().sum()} | Inf={np.isinf(yang_zhang).sum()}"
            )
            return yang_zhang
        except Exception as e:
            logger.error(f"Erreur calcul Yang-Zhang volatility: {e}")
            return pd.Series(np.nan, index=data.index)

    def _bbands(
        self, data: pd.DataFrame, period: int = 20, stddev: float = 2.0
    ) -> Dict:
        self._debug_nan(data, "BBANDS (entrée)")
        try:
            close = data["close"]
            upper, middle, lower = talib.BBANDS(
                close.values,
                timeperiod=period,
                nbdevup=stddev,
                nbdevdn=stddev,
                matype=0,
            )
            upper_series = pd.Series(upper, index=data.index)
            middle_series = pd.Series(middle, index=data.index)
            lower_series = pd.Series(lower, index=data.index)
            print(
                f"[DEBUG NaN] BBANDS upper: NaN={upper_series.isna().sum()} | Inf={np.isinf(upper_series).sum()}"
            )
            print(
                f"[DEBUG NaN] BBANDS middle: NaN={middle_series.isna().sum()} | Inf={np.isinf(middle_series).sum()}"
            )
            print(
                f"[DEBUG NaN] BBANDS lower: NaN={lower_series.isna().sum()} | Inf={np.isinf(lower_series).sum()}"
            )
            return {
                "upper": upper_series,
                "middle": middle_series,
                "lower": lower_series,
            }
        except Exception as e:
            logger.error(f"Erreur calcul Bollinger Bands: {e}")
            return {
                "upper": pd.Series(np.nan, index=data.index),
                "middle": pd.Series(np.nan, index=data.index),
                "lower": pd.Series(np.nan, index=data.index),
            }

    def _atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        self._debug_nan(data, "ATR (entrée)")
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            atr_series = pd.Series(atr, index=data.index)
            print(
                f"[DEBUG NaN] ATR (sortie): NaN={atr_series.isna().sum()} | Inf={np.isinf(atr_series).sum()}"
            )
            return atr_series
        except Exception as e:
            logger.error(f"Erreur calcul ATR: {e}")
            return pd.Series(np.nan, index=data.index)

    def _kc(
        self, data: pd.DataFrame, period: int = 20, multiplier: float = 2.0
    ) -> Dict:
        self._debug_nan(data, "KC (entrée)")
        try:
            sma = data["close"].rolling(window=period).mean()
            atr = self._atr(data, period)
            upper = sma + multiplier * atr
            lower = sma - multiplier * atr
            print(
                f"[DEBUG NaN] KC upper: NaN={upper.isna().sum()} | Inf={np.isinf(upper).sum()}"
            )
            print(
                f"[DEBUG NaN] KC lower: NaN={lower.isna().sum()} | Inf={np.isinf(lower).sum()}"
            )
            return {
                "upper": upper,
                "lower": lower,
            }
        except Exception as e:
            logger.error(f"Erreur calcul Keltner Channel: {e}")
            return {
                "upper": pd.Series(np.nan, index=data.index),
                "lower": pd.Series(np.nan, index=data.index),
            }

    # --- Volume indicators ---
    def _accumulation(self, data: pd.DataFrame) -> pd.Series:
        self._debug_nan(data, "Accumulation (entrée)")
        try:
            close = data["close"]
            low = data["low"]
            high = data["high"]
            volume = data["volume"]
            money_flow = (
                ((close - low) - (high - close)) / (high - low + 1e-10) * volume
            )
            acc = money_flow.cumsum()
            print(
                f"[DEBUG NaN] Accumulation (sortie): NaN={acc.isna().sum()} | Inf={np.isinf(acc).sum()}"
            )
            return acc
        except Exception as e:
            logger.error(f"Erreur calcul Accumulation/Distribution: {e}")
            return pd.Series(np.nan, index=data.index)

    def _chaikin_money_flow(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        self._debug_nan(data, "CMF (entrée)")
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
            print(
                f"[DEBUG NaN] CMF (sortie): NaN={cmf.isna().sum()} | Inf={np.isinf(cmf).sum()}"
            )
            return cmf
        except Exception as e:
            logger.error(f"Erreur calcul Chaikin Money Flow: {e}")
            return pd.Series(np.nan, index=data.index)

    def _ease_of_movement(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        self._debug_nan(data, "EoM (entrée)")
        try:
            high = data["high"]
            low = data["low"]
            volume = data["volume"]
            distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
            box_ratio = volume / (high - low + 1e-10)
            eom = distance_moved / box_ratio
            eom = eom.rolling(window=period).mean()
            print(
                f"[DEBUG NaN] EoM (sortie): NaN={eom.isna().sum()} | Inf={np.isinf(eom).sum()}"
            )
            return eom
        except Exception as e:
            logger.error(f"Erreur calcul Ease of Movement: {e}")
            return pd.Series(np.nan, index=data.index)

    def _obv(self, data: pd.DataFrame) -> pd.Series:
        self._debug_nan(data, "OBV (entrée)")
        try:
            close = data["close"]
            volume = data["volume"]
            obv = pd.Series(talib.OBV(close.values, volume.values), index=data.index)
            print(
                f"[DEBUG NaN] OBV (sortie): NaN={obv.isna().sum()} | Inf={np.isinf(obv).sum()}"
            )
            return obv
        except Exception as e:
            logger.error(f"Erreur calcul OBV: {e}")
            return pd.Series(np.nan, index=data.index)

    def _vwap(self, data: pd.DataFrame) -> pd.Series:
        self._debug_nan(data, "VWAP (entrée)")
        try:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            volume = data["volume"]
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            print(
                f"[DEBUG NaN] VWAP (sortie): NaN={vwap.isna().sum()} | Inf={np.isinf(vwap).sum()}"
            )
            return vwap
        except Exception as e:
            logger.error(f"Erreur calcul VWAP: {e}")
            return pd.Series(np.nan, index=data.index)

    # --- Orderflow indicators ---
    def _delta_volume(self, data: pd.DataFrame) -> pd.Series:
        self._debug_nan(data, "Delta Volume (entrée)")
        try:
            delta = data["volume"].diff()
            print(
                f"[DEBUG NaN] Delta Volume (sortie): NaN={delta.isna().sum()} | Inf={np.isinf(delta).sum()}"
            )
            return delta
        except Exception as e:
            logger.error(f"Erreur calcul Delta Volume: {e}")
            return pd.Series(np.nan, index=data.index)

    def _imbalance(self, data: pd.DataFrame) -> pd.Series:
        self._debug_nan(data, "Imbalance (entrée)")
        try:
            close = data["close"]
            volume = data["volume"]
            imbalance = np.where(close.diff() > 0, volume, -volume)
            imbalance_series = pd.Series(imbalance, index=data.index)
            print(
                f"[DEBUG NaN] Imbalance (sortie): NaN={imbalance_series.isna().sum()} | Inf={np.isinf(imbalance_series).sum()}"
            )
            return imbalance_series
        except Exception as e:
            logger.error(f"Erreur calcul Imbalance: {e}")
            return pd.Series(np.nan, index=data.index)

    def _smart_money_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        self._debug_nan(data, "SMI (entrée)")
        try:
            close = data["close"]
            open_ = data["open"]
            smi = close - open_
            smi = smi.rolling(window=period).sum()
            print(
                f"[DEBUG NaN] SMI (sortie): NaN={smi.isna().sum()} | Inf={np.isinf(smi).sum()}"
            )
            return smi
        except Exception as e:
            logger.error(f"Erreur calcul Smart Money Index: {e}")
            return pd.Series(np.nan, index=data.index)

    def _liquidity_wave(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        self._debug_nan(data, "Liquidity Wave (entrée)")
        try:
            high = data["high"]
            low = data["low"]
            volume = data["volume"]
            lw = (high - low) / (volume + 1e-10)
            lw = lw.rolling(window=period).mean()
            print(
                f"[DEBUG NaN] Liquidity Wave (sortie): NaN={lw.isna().sum()} | Inf={np.isinf(lw).sum()}"
            )
            return lw
        except Exception as e:
            logger.error(f"Erreur calcul Liquidity Wave: {e}")
            return pd.Series(np.nan, index=data.index)

    def _bid_ask_ratio(self, data: pd.DataFrame) -> float:
        self._debug_nan(data, "Bid/Ask Ratio (entrée)")
        try:
            close = data["close"]
            volume = data["volume"]
            buy_volume = volume[close.diff() > 0].sum()
            sell_volume = volume[close.diff() < 0].sum()
            total = buy_volume + abs(sell_volume)
            print(
                f"[DEBUG NaN] Bid/Ask Ratio: buy_volume={buy_volume}, sell_volume={sell_volume}, total={total}"
            )
            if total == 0:
                return 0.5
            ratio = float(buy_volume / total)
            print(f"[DEBUG NaN] Bid/Ask Ratio (sortie): ratio={ratio}")
            return ratio
        except Exception as e:
            logger.error(f"Erreur calcul Bid/Ask Ratio: {e}")
            return 0.5

    def analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict:
        self._debug_nan(data, f"Analyze Timeframe ({timeframe}) - entrée")
        try:
            results = {}
            for category, indicators in self.indicators.items():
                results[category] = {}
                for name, func in indicators.items():
                    results[category][name] = func(data)
            print(
                f"[DEBUG] Analyze Timeframe ({timeframe}) - résultats: {list(results.keys())}"
            )
            return results
        except Exception as e:
            logger.error(f"Erreur analyse timeframe {timeframe}: {e}")
            return None
