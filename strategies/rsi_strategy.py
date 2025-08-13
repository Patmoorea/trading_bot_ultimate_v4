import pandas as pd
import pandas_ta as ta

def rsi_strategy(data: pd.DataFrame, window: int = 14, overbought: float = 70, oversold: float = 30, **kwargs) -> pd.Series:
    """
    RSI: long si RSI < oversold, short si RSI > overbought, flat sinon.
    """
    rsi = ta.rsi(data['close'], length=window)
    signal = pd.Series(0, index=data.index)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = -1
    return signal
