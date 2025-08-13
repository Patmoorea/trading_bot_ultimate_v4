import websockets
import asyncio
import lz4.frame
from collections import deque
class WebSocketClient:
    def __init__(self, max_buffer_size=1000):
        self.buffer = deque(maxlen=max_buffer_size)
    async def connect(self, uri):
        async with websockets.connect(uri) as websocket:
            while True:
                compressed = await websocket.recv()
                data = lz4.frame.decompress(compressed)
                self.buffer.append(data)
    def get_latest(self):
        return list(self.buffer)
