def arbitrage_strategy(df, spread_threshold=0.5):
    """
    Stratégie Arbitrage simple : signal 1 si spread > seuil, -1 sinon.
    On suppose ici que df contient une colonne 'spread'.
    """
    signals = (df["spread"] > spread_threshold).astype(int)
    signals = signals.where(signals == 1, -1)
    signals.index = df.index
    return signals
