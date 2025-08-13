import os
os.environ['TORCH_DISABLE_DOCSTRING_CHECK'] = '1'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
def disable_torch_checks():
    """Désactive les vérifications problématiques de Torch"""
    import torch
    torch._C._set_docstring_check_enabled(False)
