import numpy as np
from numpy.core import _multiarray_umath
# Activation des accélérations M4
_multiarray_umath.set_m4_optimization_flags(
    enable_neon=True,
    enable_amx=True,  # Matrix extensions
    enable_fp16=True
)
np.show_config()  # Vérifier
