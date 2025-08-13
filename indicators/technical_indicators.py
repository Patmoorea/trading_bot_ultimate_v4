import pandas as pd
from ta import add_all_ta_features
from ta.trend import (SMAIndicator, EMAIndicator, MACD, ADXIndicator, 
                     PSARIndicator, IchimokuIndicator, VortexIndicator)
from ta.momentum import (RSIIndicator, StochasticOscillator, WilliamsRIndicator, 
                        ROCIndicator, StochRSIIndicator)
from ta.volatility import (BollingerBands, AverageTrueRange, KeltnerChannel, 
                          DonchianChannel)
from ta.volume import (OnBalanceVolumeIndicator, ForceIndexIndicator, 
                      EaseOfMovementIndicator, VolumeWeightedAveragePrice,
                      AccDistIndexIndicator, ChaikinMoneyFlowIndicator)
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
class TechnicalIndicators:
    def __init__(self):
        """Initialisation des indicateurs techniques"""
        self.window_params = {
            'short': 14,
            'medium': 50,
            'long': 200
        }
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute tous les indicateurs techniques (130+) au DataFrame"""
        # Vérifie que les colonnes nécessaires existent
        required_columns = ['high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame doit contenir les colonnes: {required_columns}")
        return add_all_ta_features(
            df,
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True
        )
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs de tendance"""
        # Moyennes Mobiles
        df['sma_short'] = SMAIndicator(close=df['close'], window=self.window_params['short']).sma_indicator()
        df['sma_medium'] = SMAIndicator(close=df['close'], window=self.window_params['medium']).sma_indicator()
        df['sma_long'] = SMAIndicator(close=df['close'], window=self.window_params['long']).sma_indicator()
        df['ema_short'] = EMAIndicator(close=df['close'], window=self.window_params['short']).ema_indicator()
        df['ema_medium'] = EMAIndicator(close=df['close'], window=self.window_params['medium']).ema_indicator()
        df['ema_long'] = EMAIndicator(close=df['close'], window=self.window_params['long']).ema_indicator()
        # MACD
        macd = MACD(close=df['close'])
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        # ADX
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        # Parabolic SAR
        psar = PSARIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['psar'] = psar.psar()
        df['psar_up'] = psar.psar_up()
        df['psar_down'] = psar.psar_down()
        # Ichimoku
        ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        # Vortex
        vortex = VortexIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['vortex_pos'] = vortex.vortex_indicator_pos()
        df['vortex_neg'] = vortex.vortex_indicator_neg()
        df['vortex_diff'] = vortex.vortex_indicator_diff()
        return df
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs de momentum"""
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        # Stochastique
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch'] = stoch.stoch()
        df['stoch_signal'] = stoch.stoch_signal()
        # Stochastique RSI
        stoch_rsi = StochRSIIndicator(close=df['close'])
        df['stoch_rsi'] = stoch_rsi.stochrsi()
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
        # ROC
        df['roc'] = ROCIndicator(close=df['close']).roc()
        return df
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs de volatilité"""
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        # ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        # Keltner Channel
        kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
        df['kc_high'] = kc.keltner_channel_hband()
        df['kc_mid'] = kc.keltner_channel_mband()
        df['kc_low'] = kc.keltner_channel_lband()
        df['kc_width'] = kc.keltner_channel_wband()
        # Donchian Channel
        dc = DonchianChannel(high=df['high'], low=df['low'], close=df['close'])
        df['dc_high'] = dc.donchian_channel_hband()
        df['dc_mid'] = dc.donchian_channel_mband()
        df['dc_low'] = dc.donchian_channel_lband()
        df['dc_width'] = dc.donchian_channel_wband()
        return df
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs de volume"""
        # On-Balance Volume
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        # Force Index
        df['force_index'] = ForceIndexIndicator(close=df['close'], volume=df['volume']).force_index()
        # Ease of Movement
        df['eom'] = EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume']).ease_of_movement()
        # Volume-Weighted Average Price
        df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
        # Accumulation/Distribution Index
        df['adi'] = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()
        # Chaikin Money Flow
        df['cmf'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).chaikin_money_flow()
        return df
    def add_other_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute d'autres indicateurs techniques"""
        # Daily Return
        df['daily_return'] = DailyReturnIndicator(close=df['close']).daily_return()
        # Cumulative Return
        df['cumulative_return'] = CumulativeReturnIndicator(close=df['close']).cumulative_return()
        return df
    def get_all_signal_indicators(self, df: pd.DataFrame) -> dict:
        """Retourne un dictionnaire avec tous les signaux des indicateurs"""
        signals = {}
        # Signaux de tendance
        signals['trend'] = {
            'sma_cross': df['sma_short'] > df['sma_medium'],
            'ema_cross': df['ema_short'] > df['ema_medium'],
            'macd_cross': df['macd_line'] > df['macd_signal'],
            'adx_trend': (df['adx'] > 25) & (df['adx_pos'] > df['adx_neg']),
            'psar_trend': df['close'] > df['psar'],
            'ichimoku_trend': df['close'] > df['ichimoku_base']
        }
        # Signaux de momentum
        signals['momentum'] = {
            'rsi_overbought': df['rsi'] > 70,
            'rsi_oversold': df['rsi'] < 30,
            'stoch_overbought': df['stoch'] > 80,
            'stoch_oversold': df['stoch'] < 20,
            'williams_r_overbought': df['williams_r'] > -20,
            'williams_r_oversold': df['williams_r'] < -80
        }
        # Signaux de volatilité
        signals['volatility'] = {
            'bb_upper_break': df['close'] > df['bb_high'],
            'bb_lower_break': df['close'] < df['bb_low'],
            'kc_upper_break': df['close'] > df['kc_high'],
            'kc_lower_break': df['close'] < df['kc_low']
        }
        # Signaux de volume
        signals['volume'] = {
            'obv_increasing': df['obv'] > df['obv'].shift(1),
            'force_index_positive': df['force_index'] > 0,
            'cmf_positive': df['cmf'] > 0
        }
        return signals
