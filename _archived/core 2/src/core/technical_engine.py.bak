# -*- coding: utf-8 -*-
"""
Technical Analysis Engine - Enhanced Version
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
import logging

# ---- Core Indicators ----
def calculate_rsi_enhanced(prices, window=14):
    """Enhanced RSI calculation with boundary checks"""
    if not isinstance(prices, (np.ndarray, pd.Series, list)):
        raise TypeError("Prices must be array-like")
    
    prices = np.asarray(prices)
    if len(prices) < window + 1:
        return 50.0  # Neutral value
    
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    if np.isclose(avg_loss, 0):
        return 100.0 if not np.isclose(avg_gain, 0) else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.clip(rsi, 0, 100)

# [Add other existing functions below]
