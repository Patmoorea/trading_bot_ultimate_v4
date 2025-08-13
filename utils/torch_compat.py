import torch
import warnings
def ensure_torch_compatibility():
    """Solution ultime pour la compatibilité Torch"""
    try:
        # Solution pour les nouvelles versions de Torch
        if hasattr(torch, 'overrides') and hasattr(torch.overrides, 'has_torch_function'):
            original = torch.overrides.has_torch_function
            if original.__doc__:
                torch.overrides.has_torch_function.__doc__ = None
        # Solution alternative pour torch._C
        if hasattr(torch, '_C'):
            torch._C._set_docstring_check_enabled(False)
        return True
    except Exception as e:
        warnings.warn(f"Torch compatibility warning: {str(e)}")
        return False
