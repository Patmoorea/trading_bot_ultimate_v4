import metal
class M4MetalOptimizer:
    def __init__(self):
        self.device = metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()
        # Configuration spécifique M4
        self.max_threads = 10  # Cœurs M4
        self.cache_config = {
            'texture': metal.MTLResourceStorageModePrivate,
            'buffer': metal.MTLResourceStorageModeShared
        }
    def configure_for_ml(self):
        """Optimise Metal pour le machine learning"""
        pipeline = metal.MTLComputePipelineState.new()
        pipeline.threadExecutionWidth = min(1024, self.device.maxThreadsPerThreadgroup)
        return pipeline
