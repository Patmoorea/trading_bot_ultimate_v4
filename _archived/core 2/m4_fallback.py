import torch
def get_device():
    if torch.backends.mps.is_available():
        try:
            # Solution de repli
            torch._C._set_mps_allocator_settings("NONE")
            return torch.device("mps")
        except:
            pass
    return torch.device("cpu")
device = get_device()
print(f"Using device: {device}")
