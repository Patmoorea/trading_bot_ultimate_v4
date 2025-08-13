import pandas as pd
import pandas_ta as ta

def ema_strategy(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    EMA: long si close > EMA, flat sinon.
    """
    ema = ta.ema(data['close'], length=window)
    signal = (data['close'] > ema).astype(int)
    return signal
