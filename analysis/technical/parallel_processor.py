import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
class ParallelTechnicalAnalyzer:
    def __init__(self, use_gpu: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_gpu = use_gpu
        self.device = cp if use_gpu else np
        self._check_gpu()
    def _check_gpu(self) -> None:
        if self.use_gpu:
            try:
                cp.cuda.runtime.getDeviceCount()
                self.logger.info("GPU acceleration enabled")
            except:
                self.logger.warning("GPU not available, falling back to CPU")
                self.use_gpu = False
                self.device = np
    def parallel_macd(self, data: np.ndarray, 
                     fast_period: int = 12,
                     slow_period: int = 26,
                     signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD in parallel"""
        data_gpu = self.device.asarray(data)
        # EMA Calculations
        fast_ema = self._parallel_ema(data_gpu, fast_period)
        slow_ema = self._parallel_ema(data_gpu, slow_period)
        # MACD Line
        macd_line = fast_ema - slow_ema
        # Signal Line
        signal_line = self._parallel_ema(macd_line, signal_period)
        # Histogram
        histogram = macd_line - signal_line
        if self.use_gpu:
            return cp.asnumpy(macd_line), cp.asnumpy(signal_line), cp.asnumpy(histogram)
        return macd_line, signal_line, histogram
    def parallel_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI in parallel"""
        data_gpu = self.device.asarray(data)
        deltas = self.device.diff(data_gpu)
        gains = self.device.where(deltas > 0, deltas, 0)
        losses = self.device.where(deltas < 0, -deltas, 0)
        avg_gains = self._parallel_ema(gains[1:], period)
        avg_losses = self._parallel_ema(losses[1:], period)
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        if self.use_gpu:
            return cp.asnumpy(rsi)
        return rsi
    def _parallel_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA using parallel processing"""
        alpha = 2 / (period + 1)
        kernel = self.device.power(1 - alpha, self.device.arange(len(data)))
        normalized_kernel = kernel / kernel.sum()
        if len(data.shape) > 1:
            normalized_kernel = normalized_kernel.reshape(-1, 1)
        return self.device.convolve(data, normalized_kernel, mode='valid')
    def parallel_bollinger_bands(self, data: np.ndarray, 
                               period: int = 20, 
                               num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands in parallel"""
        data_gpu = self.device.asarray(data)
        # Middle band (SMA)
        middle_band = self.device.convolve(data_gpu, 
                                         self.device.ones(period)/period,
                                         mode='valid')
        # Calculate standard deviation
        rolling_std = self.device.sqrt(
            self.device.convolve(
                (data_gpu - middle_band)**2,
                self.device.ones(period)/period,
                mode='valid'
            )
        )
        # Calculate bands
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        if self.use_gpu:
            return (cp.asnumpy(upper_band), 
                   cp.asnumpy(middle_band),
                   cp.asnumpy(lower_band))
        return upper_band, middle_band, lower_band
    def batch_process_indicators(self, 
                               data: Dict[str, np.ndarray],
                               timeframes: List[str],
                               indicators: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Process multiple indicators across multiple timeframes in parallel"""
        results = {}
        for timeframe in timeframes:
            results[timeframe] = {}
            timeframe_data = data[timeframe]
            for indicator in indicators:
                if indicator == 'macd':
                    macd, signal, hist = self.parallel_macd(timeframe_data)
                    results[timeframe]['macd'] = {
                        'macd_line': macd,
                        'signal_line': signal,
                        'histogram': hist
                    }
                elif indicator == 'rsi':
                    results[timeframe]['rsi'] = self.parallel_rsi(timeframe_data)
                elif indicator == 'bollinger':
                    upper, middle, lower = self.parallel_bollinger_bands(timeframe_data)
                    results[timeframe]['bollinger'] = {
                        'upper': upper,
                        'middle': middle,
                        'lower': lower
                    }
        return results
    @staticmethod
    def get_timeframe_multiplier(timeframe: str) -> int:
        """Convert timeframe string to multiplier"""
        multipliers = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15,
            '30m': 30, '1h': 60, '2h': 120, '4h': 240,
            '1d': 1440, '1w': 10080
        }
        return multipliers.get(timeframe, 1)
