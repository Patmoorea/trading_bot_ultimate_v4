import sys
import warnings
class TorchSafetyWrapper:
    """Wrapper de sécurité pour Torch"""
    def __getattr__(self, name):
        if name in ['_has_torch_function', '_set_docstring_check_enabled']:
            return lambda *args, **kwargs: None
        raise AttributeError(f"Torch safety wrapper: {name}")
# Applique le wrapper
sys.modules['torch.overrides'] = TorchSafetyWrapper()
sys.modules['torch._C'] = TorchSafetyWrapper()
warnings.warn("Torch safety wrapper activated")
