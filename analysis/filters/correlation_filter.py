import numpy as np
import pandas as pd


def compute_pairwise_correlations(market_data, pairs, timeframe="1h", window=50):
    """
    Calcule la matrice de corrélation des variations logarithmiques sur un window recent.
    market_data: dict {PAIR_KEY: {tf: {"close": [...]}}}
    pairs: liste de paires (ex: ["BTC/USDC", ...])
    timeframe: str ("1h" par défaut)
    window: taille de la fenêtre glissante
    Retourne: DataFrame de corrélations (index et colonnes = paires)
    """
    closes = {}
    for pair in pairs:
        pair_key = pair.replace("/", "").upper()
        series = market_data.get(pair_key, {}).get(timeframe, {}).get("close", [])
        if series and len(series) >= window:
            closes[pair] = np.log(pd.Series(series)[-window:]).diff().dropna()
    if len(closes) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(closes)
    return df.corr()


def filter_uncorrelated_pairs(
    market_data, pairs, timeframe="1h", window=50, corr_threshold=0.85, top_n=5
):
    """
    Sélectionne les paires les moins corrélées entre elles (algorithme greedy).
    - corr_threshold: corrélation max tolérée entre deux paires dans la sélection
    - top_n: nombre de paires max à retourner
    Retourne une liste de paires sélectionnées.
    """
    corr_matrix = compute_pairwise_correlations(market_data, pairs, timeframe, window)
    if corr_matrix.empty:
        return pairs[:top_n]  # fallback: aucune exclusion
    selected = []
    for pair in corr_matrix.columns:
        too_corr = False
        for sel in selected:
            if abs(corr_matrix.loc[pair, sel]) > corr_threshold:
                too_corr = True
                break
        if not too_corr:
            selected.append(pair)
        if len(selected) >= top_n:
            break
    return selected
