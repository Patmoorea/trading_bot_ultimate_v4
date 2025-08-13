"""
Pattern Detector Module
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Union
import logging
from src.utils.datetime_utils import format_timestamp, get_utc_now
from src.config.constants import TIMEZONE
class Pattern:
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    TRIANGLE = "triangle"
    BREAKOUT = "breakout"
class PatternDetector:
    def __init__(self):
        """Initialize pattern detector"""
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        self.patterns = {
            Pattern.DOUBLE_TOP: self._detect_double_top,
            Pattern.DOUBLE_BOTTOM: self._detect_double_bottom,
            Pattern.HEAD_SHOULDERS: self._detect_head_shoulders,
            Pattern.TRIANGLE: self._detect_triangle,
            Pattern.BREAKOUT: self._detect_breakout
        }
        self.volume_threshold = 1.5  # Volume should be 50% above average
    def detect_all(self, df: pd.DataFrame) -> List[Dict]:
        """Detect all patterns in the data"""
        if not self._validate_data(df):
            raise ValueError("Invalid data format")
        patterns_found = []
        for pattern_name, detect_func in self.patterns.items():
            try:
                patterns = detect_func(df)
                if patterns:
                    # Verify volume for each pattern
                    patterns = [p for p in patterns if self.confirm_volume(df, p)]
                    patterns_found.extend(patterns)
            except Exception as e:
                logging.error(f"Error detecting {pattern_name}: {str(e)}")
        return sorted(patterns_found, key=lambda x: x['timestamp'], reverse=True)
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data"""
        return all(col in df.columns for col in self.required_columns)
    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile"""
        return df['volume'].rolling(window=20).mean()
    def confirm_volume(self, df: pd.DataFrame, pattern: Dict) -> bool:
        """Confirm pattern with volume analysis"""
        try:
            if df.empty or 'volume' not in df.columns:
                return False
            # Calculate average volume
            volume_ma = self._calculate_volume_profile(df)
            if volume_ma.empty:
                return False
            # Get current volume
            current_volume = df['volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            # Volume should be above threshold
            return bool(current_volume > self.volume_threshold * avg_volume)
        except Exception as e:
            logging.error(f"Error in volume confirmation: {str(e)}")
            return False
    def _detect_double_top(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double top pattern"""
        patterns = []
        try:
            # Calculate peaks
            df['peaks'] = df['high'].rolling(window=5, center=True).apply(
                lambda x: 1 if x.iloc[2] == max(x) else 0
            )
            # Find consecutive peaks
            peaks = df[df['peaks'] == 1].index
            for i in range(len(peaks)-1):
                peak1, peak2 = peaks[i], peaks[i+1]
                if abs(df.loc[peak1, 'high'] - df.loc[peak2, 'high']) < 0.01 * df.loc[peak1, 'high']:
                    patterns.append({
                        'pattern': Pattern.DOUBLE_TOP,
                        'timestamp': format_timestamp(get_utc_now()),
                        'confidence': 0.8,
                        'price_level': df.loc[peak2, 'high']
                    })
        except Exception as e:
            logging.error(f"Error in double top detection: {str(e)}")
        return patterns
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double bottom pattern"""
        patterns = []
        try:
            # Calculate troughs
            df['troughs'] = df['low'].rolling(window=5, center=True).apply(
                lambda x: 1 if x.iloc[2] == min(x) else 0
            )
            # Find consecutive troughs
            troughs = df[df['troughs'] == 1].index
            for i in range(len(troughs)-1):
                trough1, trough2 = troughs[i], troughs[i+1]
                if abs(df.loc[trough1, 'low'] - df.loc[trough2, 'low']) < 0.01 * df.loc[trough1, 'low']:
                    patterns.append({
                        'pattern': Pattern.DOUBLE_BOTTOM,
                        'timestamp': format_timestamp(get_utc_now()),
                        'confidence': 0.8,
                        'price_level': df.loc[trough2, 'low']
                    })
        except Exception as e:
            logging.error(f"Error in double bottom detection: {str(e)}")
        return patterns
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """Detect head and shoulders pattern"""
        patterns = []
        try:
            # Calculate peaks
            df['peaks'] = df['high'].rolling(window=5, center=True).apply(
                lambda x: 1 if x.iloc[2] == max(x) else 0
            )
            peaks = df[df['peaks'] == 1].index
            for i in range(len(peaks)-2):
                if len(peaks[i:i+3]) == 3:
                    left, head, right = peaks[i:i+3]
                    if (df.loc[head, 'high'] > df.loc[left, 'high'] and 
                        df.loc[head, 'high'] > df.loc[right, 'high'] and
                        abs(df.loc[left, 'high'] - df.loc[right, 'high']) < 
                        0.02 * df.loc[left, 'high']):
                        patterns.append({
                            'pattern': Pattern.HEAD_SHOULDERS,
                            'timestamp': format_timestamp(get_utc_now()),
                            'confidence': 0.7,
                            'price_level': df.loc[right, 'high']
                        })
        except Exception as e:
            logging.error(f"Error in head and shoulders detection: {str(e)}")
        return patterns
    def _detect_triangle(self, df: pd.DataFrame) -> List[Dict]:
        """Detect triangle patterns"""
        patterns = []
        try:
            window = 20
            highs = df['high'].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )
            lows = df['low'].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )
            if not highs.empty and not lows.empty:
                last_high_slope = highs.iloc[-1]
                last_low_slope = lows.iloc[-1]
                if abs(last_high_slope) < 0.001 and abs(last_low_slope) < 0.001:
                    patterns.append({
                        'pattern': Pattern.TRIANGLE,
                        'timestamp': format_timestamp(get_utc_now()),
                        'confidence': 0.6,
                        'price_level': df['close'].iloc[-1]
                    })
        except Exception as e:
            logging.error(f"Error in triangle detection: {str(e)}")
        return patterns
    def _detect_breakout(self, df: pd.DataFrame) -> List[Dict]:
        """Detect breakout patterns"""
        patterns = []
        try:
            window = 20
            resistance = df['high'].rolling(window=window).max()
            support = df['low'].rolling(window=window).min()
            if df['close'].iloc[-1] > resistance.iloc[-2]:
                patterns.append({
                    'pattern': Pattern.BREAKOUT,
                    'direction': 'up',
                    'timestamp': format_timestamp(get_utc_now()),
                    'confidence': 0.75,
                    'price_level': df['close'].iloc[-1]
                })
            elif df['close'].iloc[-1] < support.iloc[-2]:
                patterns.append({
                    'pattern': Pattern.BREAKOUT,
                    'direction': 'down',
                    'timestamp': format_timestamp(get_utc_now()),
                    'confidence': 0.75,
                    'price_level': df['close'].iloc[-1]
                })
        except Exception as e:
            logging.error(f"Error in breakout detection: {str(e)}")
        return patterns
