import websockets
import asyncio
import json
import lz4.frame
from collections import deque
from typing import List, Dict  # Ajout des imports de types

class WebSocketManager:
    def __init__(self, symbols: List[str], max_buffer=1000):
        self.symbols = symbols
        self.buffers = {sym: deque(maxlen=max_buffer) for sym in symbols}
        self.connections = {}

    async def connect(self):
        """Établit les connexions WebSocket pour tous les symboles"""
        base_url = "wss://stream.binance.com:9443/ws/"
        streams = [f"{sym.lower()}@kline_1m" for sym in self.symbols]
        
        async with websockets.connect(base_url + "/".join(streams)) as ws:
            while True:
                msg = await ws.recv()
                data = json.loads(lz4.frame.decompress(msg))
                self.buffers[data['s']].append(data)

    def get_latest(self, symbol: str) -> Dict:
        """Récupère les dernières données pour un symbole"""
        return self.buffers.get(symbol, {})

    def start(self):
        """Démarre le manager dans un event loop séparé"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect())
