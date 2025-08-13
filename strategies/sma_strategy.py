def sma_strategy(df, fast_window=10, slow_window=50):
    """
    Stratégie SMA crossover.
    Retourne 1 (long) si la SMA rapide croise au-dessus de la lente, -1 sinon.
    """
    fast = df["close"].rolling(fast_window).mean()
    slow = df["close"].rolling(slow_window).mean()
    signals = (fast > slow).astype(int)
    signals = signals.where(signals == 1, -1)
    signals.index = df.index
    return signals
