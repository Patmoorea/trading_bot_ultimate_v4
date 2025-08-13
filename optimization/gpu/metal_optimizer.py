import numpy as np
import tensorflow as tf
import torch
import lz4.frame
from dataclasses import dataclass
from typing import Dict, List
@dataclass
class GPUConfig:
    use_metal: bool = True
    memory_fraction: float = 0.9
    buffer_size: int = 1_000_000
    compression: str = 'lz4'
class MetalOptimizer:
    def __init__(self, config: GPUConfig = GPUConfig()):
        self.config = config
        self._setup_metal()
        self._init_circular_buffer()
    def _setup_metal(self):
        """Configure Metal pour M1/M2"""
        if self.config.use_metal:
            # TensorFlow Metal
            tf.config.experimental.set_memory_growth(
                tf.config.list_physical_devices('GPU')[0], 
                True
            )
            # PyTorch Metal
            if torch.backends.mps.is_available():
                torch.backends.mps.enable_mps()
    def _init_circular_buffer(self):
        """Buffer circulaire optimisé (~15ms)"""
        self.buffer = np.zeros(self.config.buffer_size, dtype=np.float32)
        self.buffer_position = 0
    def optimize_array(self, data: np.ndarray) -> np.ndarray:
        """Optimise avec compression LZ4"""
        if self.config.compression == 'lz4':
            compressed = lz4.frame.compress(data.tobytes())
            return np.frombuffer(compressed, dtype=data.dtype)
        return data
    def process_batch(self, batch: np.ndarray) -> np.ndarray:
        """Traitement optimisé des batchs"""
        # Utilise le buffer circulaire
        end_pos = self.buffer_position + len(batch)
        if end_pos > self.config.buffer_size:
            end_pos = self.config.buffer_size
            self.buffer_position = 0
        self.buffer[self.buffer_position:end_pos] = batch
        self.buffer_position = end_pos
        return self.optimize_array(batch)
