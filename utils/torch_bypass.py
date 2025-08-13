import sys
import warnings
class TorchStub:
    """Stub complet pour PyTorch"""
    class _C:
        def _set_docstring_check_enabled(self, _): pass
    class overrides:
        @staticmethod
        def has_torch_function(*_): return False
# Remplace complètement torch
sys.modules['torch'] = TorchStub()
warnings.warn("Torch stub activated - limited functionality")
