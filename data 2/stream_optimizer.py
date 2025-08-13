import lz4.frame
import pyarrow as pa
from threading import Lock
class StreamOptimizer:
    def __init__(self):
        self.lock = Lock()
    def process(self, data):
        with self.lock:
            # Compression LZ4 + Sérialisation Arrow
            compressed = lz4.frame.compress(
                pa.serialize(data).to_buffer()
            )
            return {
                'original_size': len(data),
                'compressed_size': len(compressed),
                'data': compressed
            }
# Solution de repli si PyArrow non disponible
try:
    import pyarrow as pa
    serialize = pa.serialize
except (ImportError, AttributeError):
    import pickle
    serialize = pickle.dumps
