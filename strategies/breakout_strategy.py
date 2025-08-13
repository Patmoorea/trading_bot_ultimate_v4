import pandas as pd

def breakout_strategy(data: pd.DataFrame, window: int = 50, **kwargs) -> pd.Series:
    """
    Breakout: long si close > max(high N jours), flat sinon.
    """
    high_roll = data['high'].rolling(window=window).max()
    signal = (data['close'] > high_roll.shift(1)).astype(int)
    return signal
