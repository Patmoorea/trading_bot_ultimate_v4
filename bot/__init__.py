from src.data.realtime.websocket.client import MultiStreamManager


class MonBot:
    def __init__(self):
        self.stream_config = ...  # initialise ici ta config
        self.ws_manager = MultiStreamManager(self.stream_config)
        self.ws_manager = None
