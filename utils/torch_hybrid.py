import sys
import warnings
import importlib
class TorchStub:
    """Stub partiel pour PyTorch avec fallback"""
    def __getattr__(self, name):
        # Retourne un mock pour les attributs non critiques
        if name in ['_C', 'overrides']:
            return self
        return lambda *args, **kwargs: None
# Charge le vrai torch mais avec des fallbacks
_real_torch = importlib.import_module('torch')
sys.modules['torch'] = _real_torch
# Applique les patches nécessaires
try:
    if hasattr(_real_torch._C, '_set_docstring_check_enabled'):
        _real_torch._C._set_docstring_check_enabled(False)
except Exception as e:
    warnings.warn(f"Could not configure torch: {str(e)}")
    sys.modules['torch'] = TorchStub()
