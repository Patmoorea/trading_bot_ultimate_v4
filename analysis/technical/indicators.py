class TechnicalAnalysis:
    def __init__(self):
        pass
    def calculate(self):
        pass
    def supertrend(self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """Calcule l'indicateur Supertrend
        Args:
            data (pd.DataFrame): DataFrame avec colonnes high, low, close
            period (int): Période pour l'ATR, par défaut 10
            multiplier (float): Multiplicateur pour les bandes, par défaut 3.0
        Returns:
            pd.Series: Valeurs du Supertrend
        """
        hl2 = (data['high'] + data['low']) / 2
        atr = self._calculate_atr(data, period)
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        supertrend = pd.Series(index=data.index)
        direction = pd.Series(1, index=data.index)
        for i in range(1, len(data.index)):
            curr, prev = data.index[i], data.index[i-1]
            if data['close'][curr] > upperband[prev]:
                direction[curr] = 1
            elif data['close'][curr] < lowerband[prev]:
                direction[curr] = -1
            else:
                direction[curr] = direction[prev]
                if direction[curr] == 1 and lowerband[curr] < lowerband[prev]:
                    lowerband[curr] = lowerband[prev]
                if direction[curr] == -1 and upperband[curr] > upperband[prev]:
                    upperband[curr] = upperband[prev]
            if direction[curr] == 1:
                supertrend[curr] = lowerband[curr]
            else:
                supertrend[curr] = upperband[curr]
        return supertrend
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calcule l'Average True Range (ATR)"""
        tr = pd.DataFrame(index=data.index)
        tr['h-l'] = data['high'] - data['low']
        tr['h-pc'] = abs(data['high'] - data['close'].shift(1))
        tr['l-pc'] = abs(data['low'] - data['close'].shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return tr['tr'].rolling(period).mean()
