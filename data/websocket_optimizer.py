import asyncio
import websockets
from typing import Dict, List
from .enhanced_buffer import CircularLZ4Buffer
class OptimizedWebSocketManager:
    def __init__(self, max_connections: int = 12):
        self.max_connections = max_connections
        self.active_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.buffers: Dict[str, CircularLZ4Buffer] = {}
        self.connection_semaphore = asyncio.Semaphore(max_connections)
    async def add_feed(self, symbol: str, ws_url: str):
        async with self.connection_semaphore:
            if symbol not in self.buffers:
                self.buffers[symbol] = CircularLZ4Buffer()
            try:
                ws = await websockets.connect(ws_url)
                self.active_connections[symbol] = ws
                asyncio.create_task(self._handle_messages(symbol, ws))
            except Exception as e:
                print(f"Error connecting to {symbol}: {e}")
    async def _handle_messages(self, symbol: str, ws: websockets.WebSocketClientProtocol):
        try:
            while True:
                message = await ws.recv()
                self.buffers[symbol].add_data(message)
        except Exception as e:
            print(f"Connection lost for {symbol}: {e}")
            await self.reconnect(symbol)
    async def reconnect(self, symbol: str):
        await asyncio.sleep(1)  # Backoff
        if symbol in self.active_connections:
            ws_url = self.active_connections[symbol].url
            await self.add_feed(symbol, ws_url)
