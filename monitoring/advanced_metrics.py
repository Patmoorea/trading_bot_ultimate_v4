class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            'latency': LatencyTracker(),
            'memory': MemoryMonitor(),
            'throughput': ThroughputAnalyzer()
        }
    def track_all(self):
        return {
            name: tracker.get_current()
            for name, tracker in self.metrics.items()
        }
