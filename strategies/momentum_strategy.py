import pandas as pd

def momentum_strategy(data: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
    """
    Momentum: long si momentum positif, short si négatif, flat sinon.
    """
    momentum = data['close'].diff(window)
    signal = pd.Series(0, index=data.index)
    signal[momentum > 0] = 1
    signal[momentum < 0] = -1
    return signal
