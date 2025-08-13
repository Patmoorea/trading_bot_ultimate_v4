import torch
def configure_mps():
    if not torch.backends.mps.is_available():
        return torch.device("cpu")
    # Nouvelle API pour M4
    if hasattr(torch.backends.mps, 'is_macos13_or_newer'):
        torch.backends.mps.set_memory_fraction(0.8)  # Nouvelle méthode
    # Désactiver les vérifications problématiques
    torch._C._set_mps_allocator_settings("NONE") 
    return torch.device("mps")
device = configure_mps()
print(f"Device configuré: {device}")
