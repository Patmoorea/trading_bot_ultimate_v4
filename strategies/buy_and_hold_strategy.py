import pandas as pd

def buy_and_hold_strategy(data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Buy & Hold: long du début à la fin.
    """
    signal = pd.Series(1, index=data.index)
    return signal
