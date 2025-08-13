import torch
import warnings
import functools
def safe_patch_torch():
    """Patch Torch de manière non-destructive"""
    if not hasattr(torch._C._VariableFunctions, '_has_torch_function'):
        return False
    original = torch._C._VariableFunctions._has_torch_function
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)
    torch._C._VariableFunctions._has_torch_function = wrapped
    return True
# Applique le patch au chargement
success = safe_patch_torch()
if not success:
    warnings.warn("Could not apply torch _has_torch_function patch")
