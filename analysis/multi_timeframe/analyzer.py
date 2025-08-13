"""
Ce module complète les fonctionnalités existantes en permettant
l'analyse de données sur plusieurs timeframes simultanément.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
class MultiTimeframeAnalyzer:
    """
    Analyse des données de marché sur plusieurs timeframes simultanément.
    Cette classe s'intègre avec les modules d'analyse existants.
    """
    def __init__(self, timeframes: List[str] = None):
        """
        Initialise l'analyseur multi-timeframe.
        Args:
            timeframes: Liste des timeframes à analyser (défaut: ["1m", "5m", "15m", "1h", "4h", "1d"])
        """
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.data_frames: Dict[str, pd.DataFrame] = {}
    def add_data(self, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Ajoute des données pour un timeframe spécifique.
        Args:
            timeframe: Période temporelle ("1m", "5m", etc.)
            data: DataFrame avec colonnes OHLCV
        Returns:
            Succès de l'opération
        """
        if timeframe in self.timeframes:
            self.data_frames[timeframe] = data
            return True
        return False
    def analyze_confluence(self, indicators: List[str] = None) -> Dict[str, float]:
        """
        Analyse la confluence des signaux sur différents timeframes.
        Args:
            indicators: Liste d'indicateurs à calculer (défaut: ["ema", "rsi", "macd"])
        Returns:
            Dictionnaire avec scores de confluence pour chaque indicateur
        """
        indicators = indicators or ["ema", "rsi", "macd"]
        results = {}
        # Vérifier qu'on a suffisamment de timeframes
        if len(self.data_frames) < 2:
            return {"error": "Pas assez de timeframes disponibles"}
        # Calculer la confluence pour chaque indicateur
        for indicator in indicators:
            signals = []
            for tf, df in self.data_frames.items():
                # Calculer le signal pour cet indicateur sur ce timeframe
                signal = self._calculate_indicator_signal(df, indicator)
                signals.append(signal)
            # Calculer le score de confluence
            if signals:
                # Un score élevé indique une forte confluence (même direction sur plusieurs TF)
                # Un score proche de zéro indique des signaux contradictoires
                score = np.mean(signals)
                strength = abs(score)
                direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
                results[indicator] = {
                    "score": score,
                    "strength": strength,
                    "direction": direction
                }
        return results
    def _calculate_indicator_signal(self, df: pd.DataFrame, indicator: str) -> float:
        """
        Calcule un signal normalisé (-1 à 1) pour un indicateur sur un dataframe.
        Args:
            df: DataFrame avec données OHLCV
            indicator: Nom de l'indicateur
        Returns:
            Signal normalisé (-1 = très bearish, +1 = très bullish)
        """
        import talib
        if indicator == "ema":
            # Calcul de l'EMA
            ema20 = talib.EMA(df["close"].values, timeperiod=20)
            ema50 = talib.EMA(df["close"].values, timeperiod=50)
            # Signal basé sur la position relative des EMAs
            last_price = df["close"].iloc[-1]
            last_ema20 = ema20[-1]
            last_ema50 = ema50[-1]
            # Signal normalisé
            if last_ema20 > last_ema50:
                # Bullish - force basée sur l'écart
                return min(1.0, (last_ema20 - last_ema50) / last_price * 20)
            else:
                # Bearish - force basée sur l'écart
                return max(-1.0, (last_ema20 - last_ema50) / last_price * 20)
        elif indicator == "rsi":
            # Calcul du RSI
            rsi = talib.RSI(df["close"].values)
            last_rsi = rsi[-1]
            # Signal normalisé (-1 à +1)
            if last_rsi > 70:
                return -1.0  # Très suracheté (bearish)
            elif last_rsi < 30:
                return 1.0   # Très survendu (bullish)
            else:
                # Entre 30 et 70, normaliser sur [-0.5, 0.5]
                return (50 - last_rsi) / 40  # 50=neutre, <50=bullish, >50=bearish
        elif indicator == "macd":
            # Calcul du MACD
            macd, signal, hist = talib.MACD(df["close"].values)
            # Signal basé sur l'histogramme récent
            last_hist = hist[-1]
            prev_hist = hist[-2]
            # Signal de base depuis l'histogramme
            base_signal = 1.0 if last_hist > 0 else -1.0
            # Ajustement en fonction de la tendance de l'histogramme
            if last_hist > prev_hist:
                # Histogramme en hausse = renforcement du signal
                return base_signal * min(1.0, abs(last_hist / 100))
            else:
                # Histogramme en baisse = affaiblissement du signal
                return base_signal * min(0.5, abs(last_hist / 200))
        # Indicateur non reconnu
        return 0.0
