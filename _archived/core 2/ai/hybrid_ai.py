try:
    from torch import nn
except ImportError:
    # Fallback pour torch.nn
    class nn:
        Module = object
        Linear = lambda *args, **kwargs: None
class HybridAIEnhanced:
    """Version compatible avec ou sans Torch"""
    def __init__(self):
        self.nn = nn
