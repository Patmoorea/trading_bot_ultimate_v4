import torch
import platform
def safe_mps_device():
    """Retourne un device MPS configuré de manière stable"""
    if not torch.backends.mps.is_available():
        return torch.device('cpu')
    # Désactive les vérifications problématiques
    torch._C._set_mps_allocator_settings("NONE")
    # Configuration spécifique macOS 15+ et M4
    if platform.mac_ver()[0] >= '15.0':
        torch.backends.mps.enable_flash_attention(True)
        torch._C._mps_setLowWatermarkRatio(0.0)  # Désactive le watermark
    return torch.device('mps')
device = safe_mps_device()
print(f"Device stable configuré : {device}")
