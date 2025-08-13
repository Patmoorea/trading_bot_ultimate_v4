import pandas as pd
import pandas_ta as ta

def parabolic_sar_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Parabolic SAR: long si close > PSAR, short sinon.
    """
    psar = ta.psar(data['high'], data['low'], data['close'])
    signal = pd.Series(0, index=data.index)
    signal[data['close'] > psar['PSARl_0.02_0.2']] = 1
    signal[data['close'] < psar['PSARs_0.02_0.2']] = -1
    return signal
