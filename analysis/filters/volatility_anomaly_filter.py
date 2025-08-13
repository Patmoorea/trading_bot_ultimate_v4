import numpy as np
import pandas as pd

def rolling_volatility(df, window=24, price_col="close"):
    """
    Calcule la volatilité (écart-type des returns) sur une fenêtre glissante.
    """
    returns = df[price_col].pct_change()
    vol = returns.rolling(window=window, min_periods=window//2).std()
    return vol

def zscore_anomaly(series, window=48):
    """
    Détecte les anomalies (z-score élevé) dans une série temporelle.
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    zscores = (series - rolling_mean) / (rolling_std + 1e-8)
    return zscores.abs()

def filter_market(df, vol_threshold=0.12, anomaly_threshold=4.0, price_col="close"):
    """
    Renvoie True si le marché est 'sain', False si trop volatil ou anormal.
    """
    vol = rolling_volatility(df, price_col=price_col)
    zscores = zscore_anomaly(df[price_col])
    if vol.iloc[-1] > vol_threshold:
        return False
    if zscores.iloc[-1] > anomaly_threshold:
        return False
    return True