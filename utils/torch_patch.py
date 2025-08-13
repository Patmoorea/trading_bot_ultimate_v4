import torch
import warnings
import types
def apply_torch_fixes():
    """Solution ultime pour les problèmes Torch"""
    try:
        # Solution pour torch._C._VariableFunctions
        if hasattr(torch._C, '_VariableFunctions'):
            original = torch._C._VariableFunctions._has_torch_function
            new_func = types.FunctionType(
                original.__code__,
                original.__globals__,
                name='_has_torch_function',
                argdefs=original.__defaults__,
                closure=original.__closure__
            )
            new_func.__doc__ = "Torch function patch"
            torch._C._VariableFunctions._has_torch_function = new_func
        return True
    except Exception as e:
        warnings.warn(f"Torch patch error: {str(e)}")
        return False
