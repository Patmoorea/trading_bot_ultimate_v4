class EnhancedDataCollector:
    def __init__(self):
        self.ws_connections = 12
        self.buffer = CircularBuffer(size=1000)
        self.compression = 'LZ4'
    def parallel_download(self, timeframes):
        """Téléchargement parallèle sur 8 timeframes"""
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self._download, timeframes))
        return self._convert_to_arrow(results)
