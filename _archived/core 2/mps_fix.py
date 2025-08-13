import torch
torch.mps.set_per_process_memory_fraction(0.8)  # Valeur valide entre 0.0 et 1.0
print(f"MPS mémoire configurée : {torch.mps.current_allocated_memory()/1e6:.1f}MB")
