"""
Library for generating embedding of videos for a given folder.
This file will run automatically when the library is imported.
"""

# For importing * from this library
__all__ = ['Window', 'Dataset', 'EmbeddingModel', 'device', 'this_os']

import torch
import os

this_os = os.name

print("Operating System:", this_os)

# Set the device to use (e.g., 'cpu', 'cuda', 'mps')
device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu")
                      )

# Select Device According to Availability
print("Device selected:", device)

# If the device is CUDA, print the device capability
if device.type == "cuda":
    os.system("nvidia-smi")
    print()
    print("Device type:", device.type)
    print("Capability:", torch.cuda.get_device_capability(device))
else:
    print("Device capabilities are limited on MPSs and CPUs.")
