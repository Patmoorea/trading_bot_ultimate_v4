import pandas as pd
import pandas_ta as ta

def bollinger_strategy(data: pd.DataFrame, window: int = 20, std: float = 2.0, **kwargs) -> pd.Series:
    """
    Bollinger: long si close < lower, short si close > upper, flat sinon
    """
    bb = ta.bbands(data['close'], length=window, std=std)
    signal = pd.Series(0, index=data.index)
    signal[data['close'] < bb['BBL_20_2.0']] = 1
    signal[data['close'] > bb['BBU_20_2.0']] = -1
    return signal
