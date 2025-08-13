import psutil
class MemoryOptimizer:
    """Gestion mémoire pour M4"""
    def __init__(self):
        self.max_mem = 12  # GB (80% de 16GB)
    def check_memory(self):
        used = psutil.virtual_memory().used / (1024**3)
        return used < self.max_mem
    def optimize_trading(self):
        import gc
        gc.collect()
        if not self.check_memory():
            raise MemoryError("Memory limit reached")
