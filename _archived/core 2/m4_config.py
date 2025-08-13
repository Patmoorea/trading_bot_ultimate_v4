import torch
class M4Config:
    def __init__(self):
        self.device = self._setup_device()
        self._configure()
    def _setup_device(self):
        if torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(0.8)
            return torch.device("mps")
        return torch.device("cpu")
    def _configure(self):
        if str(self.device) == "mps":
            # Paramètres spécifiques M4
            torch.backends.mps.enable_flash_attention(True)
            torch._C._set_mps_allocator_settings(
                strategy="simple",
                max_split_size_mb=128
            )
config = M4Config()
print(f"M4 configuré : {config.device}")
