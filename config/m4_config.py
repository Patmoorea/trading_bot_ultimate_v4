import os
# Configuration M4 optimale
os.environ.update({
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
    'PYTORCH_ENABLE_MPS_FALLBACK': '1'
})
TORCH_MM_PRECISION = 'high'  # Pour les calculs matriciels
