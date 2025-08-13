import pandas as pd

def mean_reversion_strategy(data: pd.DataFrame, window: int = 20, threshold: float = 2.0, **kwargs) -> pd.Series:
    """
    Mean Reversion: short si close > SMA + threshold*std, long si close < SMA - threshold*std, flat sinon.
    """
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper = sma + threshold * std
    lower = sma - threshold * std
    signal = pd.Series(0, index=data.index)
    signal[data['close'] > upper] = -1
    signal[data['close'] < lower] = 1
    return signal
