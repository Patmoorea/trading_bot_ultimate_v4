import torch
import warnings
class DeviceManager:
    def __init__(self):
        self._device = self._init_device()
    def _init_device(self):
        # Essayer MPS en premier
        if torch.backends.mps.is_available():
            try:
                # Test minimal avec MPS
                test_tensor = torch.rand(2, device='mps')
                if test_tensor.sum() >= 0:  # Simple vérification
                    print("MPS activé avec succès")
                    return torch.device("mps")
            except RuntimeError as e:
                warnings.warn(f"Échec MPS: {str(e)}")
        # Fallback CPU
        print("Utilisation du CPU comme fallback")
        return torch.device("cpu")
    @property
    def device(self):
        return self._device
dm = DeviceManager()
def force_mps():
    """Tente de forcer l'utilisation de MPS"""
    if torch.backends.mps.is_available():
        try:
            torch._C._set_mps_allocator_settings("NONE")
            torch._C._mps_setLowWatermarkRatio(0.0)
            return torch.device("mps")
        except:
            pass
    return torch.device("cpu")
dm._device = force_mps()
print(f"Device final: {dm.device}")
