import numpy as np
import tensorflow as tf
from dataclasses import dataclass
import lz4.frame
from typing import Any, Dict
@dataclass
class GPUConfig:
    use_metal: bool = True
    memory_fraction: float = 0.9
    compression: str = "lz4"
class GPUBuffer:
    def __init__(self, config: GPUConfig = GPUConfig()):
        self.config = config
        self._setup_metal()
        self._init_buffer()
    def _setup_metal(self):
        """Configure Metal pour M4"""
        if self.config.use_metal:
            # Configuration TensorFlow Metal
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                    tf.config.experimental.set_virtual_device_configuration(
                        physical_devices[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=self.config.memory_fraction * 16000
                        )]
                    )
                except RuntimeError as e:
                    print(f"Erreur configuration GPU: {e}")
    def _init_buffer(self):
        """Initialise le buffer circulaire"""
        self.buffer_size = 1_000_000  # Pour ~15ms de latence
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.position = 0
    def add_data(self, data: np.ndarray):
        """Ajoute des données avec compression LZ4"""
        if self.config.compression == "lz4":
            compressed = lz4.frame.compress(data.tobytes())
            data = np.frombuffer(compressed, dtype=data.dtype)
        end_pos = self.position + len(data)
        if end_pos > self.buffer_size:
            # Rotation du buffer
            self.position = 0
            end_pos = len(data)
        self.buffer[self.position:end_pos] = data
        self.position = end_pos
