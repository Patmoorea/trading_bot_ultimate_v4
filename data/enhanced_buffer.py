import lz4.frame
from collections import deque
from typing import Dict, Any
class CircularLZ4Buffer:
    def __init__(self, max_size: int = 1_000_000):
        self.buffer = deque(maxlen=max_size)
        self.compression_context = lz4.frame.create_compression_context()
    def add_data(self, data: Dict[str, Any]):
        compressed = lz4.frame.compress(str(data).encode(), 
                                     compression_level=3,
                                     block_size=lz4.frame.BLOCKSIZE_MAX1MB)
        self.buffer.append(compressed)
    def get_latest(self, n: int = 100) -> list:
        result = []
        for i in range(min(n, len(self.buffer))):
            data = lz4.frame.decompress(self.buffer[-(i+1)])
            result.append(eval(data.decode()))
        return result
    def clear(self):
        self.buffer.clear()
