import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name)
    print("vram_gb", round(p.total_memory / 1e9, 1))
