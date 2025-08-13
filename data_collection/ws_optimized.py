import lz4.frame
import websockets
import asyncio
from signal import SIGINT, SIGTERM
class OptimizedWSClient:
    def __init__(self):
        self._should_stop = False
        asyncio.get_event_loop().add_signal_handler(SIGINT, self.stop)
        asyncio.get_event_loop().add_signal_handler(SIGTERM, self.stop)
    def stop(self):
        self._should_stop = True
    async def connect(self, urls):
        async with websockets.connect(urls[0]) as ws:
            while not self._should_stop:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=1)
                    compressed = lz4.frame.compress(data.encode())
                    print(f"Reçu {len(compressed)} octets (compressés)")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Erreur: {e}")
                    break
