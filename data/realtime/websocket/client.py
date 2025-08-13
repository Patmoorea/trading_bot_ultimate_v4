import logging
import asyncio
import json
from datetime import datetime
import websockets
logger = logging.getLogger(__name__)
class StreamConfig:
    def __init__(self, max_connections=10, reconnect_delay=1.0, buffer_size=1000):
        self.max_connections = max_connections
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
class MultiStreamManager:
    def __init__(self, config: StreamConfig):
        self.config = config
        self.active_streams = {}
        self.callbacks = {}
        self._running = False
        self._tasks = []
        self._ws_connections = {}
    async def connect(self):
        """Initialise les connexions"""
        self._running = True
        logger.info("✅ Initialisation du Stream Manager")
        return self
    async def subscribe_stream(self, pair, timeframe, stream_name, callback):
        """Souscrit à un stream spécifique"""
        try:
            if not self._running:
                await self.connect()
            ws = await self._create_websocket_connection(stream_name)
            self.callbacks[stream_name] = callback
            self.active_streams[f"{pair}_{timeframe}"] = {
                "ws": ws,
                "stream": stream_name,
                "last_message": datetime.utcnow()
            }
            task = asyncio.create_task(self._process_messages(ws, stream_name))
            self._tasks.append(task)
            logger.info(f"✅ Souscrit au stream {stream_name}")
        except Exception as e:
            logger.error(f"❌ Erreur souscription stream {stream_name}: {e}")
            raise
    async def _create_websocket_connection(self, stream_name):
        """Crée une connexion websocket"""
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        ws = await websockets.connect(ws_url)
        self._ws_connections[stream_name] = ws
        return ws
    async def _process_messages(self, ws, stream_name):
        """Traite les messages reçus sur le websocket"""
        try:
            while self._running:
                message = await ws.recv()
                data = json.loads(message)
                if stream_name in self.callbacks:
                    await self.callbacks[stream_name](data)
                stream_key = next(k for k, v in self.active_streams.items() if v["stream"] == stream_name)
                self.active_streams[stream_key]["last_message"] = datetime.utcnow()
        except websockets.ConnectionClosed:
            logger.warning(f"Connection fermée pour {stream_name}, tentative de reconnexion...")
            await self._reconnect(stream_name)
        except Exception as e:
            logger.error(f"Erreur traitement messages pour {stream_name}: {e}")
            await self._reconnect(stream_name)
    async def _reconnect(self, stream_name):
        """Tente de reconnecter un stream"""
        while self._running:
            try:
                await asyncio.sleep(self.config.reconnect_delay)
                stream_key = next(k for k, v in self.active_streams.items() if v["stream"] == stream_name)
                pair, timeframe = stream_key.split("_")
                await self.subscribe_stream(
                    pair,
                    timeframe,
                    stream_name,
                    self.callbacks[stream_name]
                )
                break
            except Exception as e:
                logger.error(f"Erreur reconnexion {stream_name}: {e}")
                continue
    async def disconnect(self):
        """Déconnecte tous les streams"""
        self._running = False
        for stream_name, ws in self._ws_connections.items():
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Erreur fermeture stream {stream_name}: {e}")
        for task in self._tasks:
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass
        self.active_streams.clear()
        self.callbacks.clear()
        self._tasks.clear()
        self._ws_connections.clear()
    def is_connected(self):
        """Vérifie si le manager est connecté"""
        return self._running and bool(self._ws_connections)
