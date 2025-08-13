import pandas as pd

def donchian_strategy(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Donchian: long si close > plus haut N, short si close < plus bas N, flat sinon
    """
    highest_high = data['high'].rolling(window=window).max()
    lowest_low = data['low'].rolling(window=window).min()
    signal = pd.Series(0, index=data.index)
    signal[data['close'] > highest_high.shift(1)] = 1
    signal[data['close'] < lowest_low.shift(1)] = -1
    return signal
