def momentum_strategy(df, period=14):
    """Stratégie momentum avec confirmation volume"""
    df['momentum'] = df['close'].pct_change(period)
    df['vol_confirm'] = (df['volume'] > df['volume'].rolling(50).mean())
    df['signal'] = (df['momentum'] > 0) & df['vol_confirm']
    return df
def mean_reversion_strategy(df, window=20, threshold=2):
    """Stratégie mean-reversion"""
    rolling_mean = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()
    df['z_score'] = (df['close'] - rolling_mean) / rolling_std
    df['signal'] = df['z_score'].abs() > threshold
    return df
