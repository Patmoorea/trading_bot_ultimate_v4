import lz4.frame
import pandas as pd
class CompressedStorage:
    def __init__(self):
        self.compression = lz4.frame
    def save(self, data: pd.DataFrame, path: str):
        compressed = self.compression.compress(data.to_json().encode())
        with open(path, 'wb') as f:
            f.write(compressed)
    def load(self, path: str) -> pd.DataFrame:
        with open(path, 'rb') as f:
            decompressed = self.compression.decompress(f.read())
        return pd.read_json(decompressed.decode())
