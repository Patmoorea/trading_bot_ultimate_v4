class StreamManager:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = []
    def add_data(self, data):
        """Add data with type checking"""
        if not isinstance(data, (dict, list)):
            raise ValueError("Data must be dict or list")
        self.buffer.append(data)
        self._trim_buffer()
    def _trim_buffer(self):
        """Maintain buffer size"""
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
