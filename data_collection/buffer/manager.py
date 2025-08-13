import lz4.frame
from collections import deque
from typing import Dict, List
import numpy as np
from datetime import datetime
import logging
class CircularBufferManager:
    def __init__(self, max_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.buffers: Dict[str, deque] = {}
        self.max_size = max_size
        self.compression_stats = {
            'total_raw_size': 0,
            'total_compressed_size': 0,
            'compression_ratio': 0
        }
    def create_buffer(self, symbol: str, timeframe: str) -> None:
        key = f"{symbol}_{timeframe}"
        if key not in self.buffers:
            self.buffers[key] = deque(maxlen=self.max_size)
            self.logger.info(f"Buffer créé pour {key}")
    def add_data(self, symbol: str, timeframe: str, data: Dict) -> None:
        key = f"{symbol}_{timeframe}"
        if key not in self.buffers:
            self.create_buffer(symbol, timeframe)
        try:
            raw_data = str(data).encode()
            self.compression_stats['total_raw_size'] += len(raw_data)
            compressed_data = lz4.frame.compress(raw_data)
            self.compression_stats['total_compressed_size'] += len(compressed_data)
            self.buffers[key].append({
                'data': compressed_data,
                'timestamp': datetime.utcnow().timestamp(),
                'size': len(compressed_data)
            })
            # Mise à jour du ratio de compression
            if self.compression_stats['total_raw_size'] > 0:
                self.compression_stats['compression_ratio'] = (
                    self.compression_stats['total_compressed_size'] / 
                    self.compression_stats['total_raw_size']
                )
        except Exception as e:
            self.logger.error(f"Erreur lors de la compression: {str(e)}")
            raise
    def get_data(self, symbol: str, timeframe: str, n: int = None) -> List[Dict]:
        key = f"{symbol}_{timeframe}"
        if key not in self.buffers:
            return []
        buffer = self.buffers[key]
        if n is None:
            n = len(buffer)
        result = []
        for item in list(buffer)[-n:]:
            try:
                decompressed = lz4.frame.decompress(item['data'])
                data = eval(decompressed.decode())  # Attention: eval() à utiliser avec précaution
                result.append(data)
            except Exception as e:
                self.logger.error(f"Erreur lors de la décompression: {str(e)}")
                continue
        return result
    def get_stats(self) -> Dict:
        return {
            'buffers': len(self.buffers),
            'compression': self.compression_stats,
            'memory_usage': sum(
                sum(item['size'] for item in buffer)
                for buffer in self.buffers.values()
            )
        }
