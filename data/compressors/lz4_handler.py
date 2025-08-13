import lz4.frame
class LZ4Compressor:
    """Compression/décompression optimisée pour les flux temps-réel"""
    def __init__(self):
        self.compression_level = 4
    def compress(self, data: bytes) -> bytes:
        return lz4.frame.compress(data, compression_level=self.compression_level)
    def decompress(self, compressed_data: bytes) -> bytes:
        return lz4.frame.decompress(compressed_data)
