import tensorflow as tf
from src.core_merged.gpu_setup import GPU_AVAILABLE
class TradingModel:
    def __init__(self):
        self.device = '/GPU:0' if GPU_AVAILABLE else '/CPU:0'
    def predict(self, data):
        with tf.device(self.device):
            # Exemple simple
            return tf.constant([0.5], dtype=tf.float32)
