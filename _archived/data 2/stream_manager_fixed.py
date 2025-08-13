from binance import ThreadedWebsocketManager
import lz4.frame
import threading
class StreamManager:
    def __init__(self):
        self.twm = ThreadedWebsocketManager()
        self.buffer = []
        self.lock = threading.Lock()
        self.symbols = ['BTCUSDT', 'ETHUSDT']
    def start(self):
        self.twm.start()
        streams = [f"{s.lower()}@kline_1m" for s in self.symbols]
        self.twm.start_multiplex_socket(
            callback=self._handle_message,
            streams=streams
        )
    def _handle_message(self, msg):
        with self.lock:
            compressed = lz4.frame.compress(str(msg).encode())
            self.buffer.append(compressed)
if __name__ == "__main__":
    sm = StreamManager()
    sm.start()
