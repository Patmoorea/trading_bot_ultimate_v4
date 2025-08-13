import pandas as pd
import pandas_ta as ta

def macd_strategy(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal_: int = 9, **kwargs) -> pd.Series:
    """
    MACD: long si MACD > Signal, short sinon.
    """
    macd = ta.macd(data['close'], fast=fast, slow=slow, signal=signal_)
    signal = pd.Series(0, index=data.index)
    signal[macd['MACD_12_26_9'] > macd['MACDs_12_26_9']] = 1
    signal[macd['MACD_12_26_9'] < macd['MACDs_12_26_9']] = -1
    return signal
