import sys
import warnings
class TorchMock:
    """Mock pour contourner les vérifications de docstring"""
    class _C:
        class _VariableFunctions:
            @staticmethod
            def _has_torch_function(*args, **kwargs):
                return False
if 'torch' not in sys.modules:
    sys.modules['torch'] = TorchMock()
    warnings.warn("Torch mock activated for docstring checks")
