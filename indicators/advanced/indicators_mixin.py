import pandas as pd
import numpy as np
class IndicatorsMixin:
    def _calculate_supertrend(self, data, period=10, multiplier=3):
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            # Calcul de l'ATR
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=period).mean()
            # Calcul des bandes
            upperband = ((high + low) / 2) + (multiplier * atr)
            lowerband = ((high + low) / 2) - (multiplier * atr)
            # Calcul du Supertrend
            supertrend = pd.Series(index=close.index)
            direction = pd.Series(index=close.index)
            for i in range(period, len(close)):
                if close[i] > upperband[i-1]:
                    direction[i] = 1
                elif close[i] < lowerband[i-1]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i-1]
                if direction[i] == 1:
                    supertrend[i] = lowerband[i]
                else:
                    supertrend[i] = upperband[i]
            return {
                'value': supertrend,
                'direction': direction,
                'strength': abs(close - supertrend) / close
            }
        except Exception as e:
            print(f"Erreur calcul Supertrend: {e}")
            return None
    # Ajout des autres méthodes de calcul...
