"""Computing the WECT of the Stanford Bunny"""

import torch
import os
from pyect import (
    mesh_to_complex,
    sample_directions_3d,
    WECT
)

# Get the absolute file path for the bunny
script_dir = os.path.dirname(os.path.abspath(__file__))
bunny_path = os.path.join(script_dir, "bunny.obj")

# Set the device
device = torch.device("cpu")

# Construct a simplicial complex out of the obj file
bunny_complex = mesh_to_complex(bunny_path, device=device, centering=True)

# Sample directions
num_dirs = 100
directions = sample_directions_3d(num_dirs=num_dirs, device=device)

# Set the number of height values to sample
num_heights = 1000

# Initialize the WECT module
wect = WECT(dirs=directions, num_heights=num_heights).eval()

# Compute the WECT of the bunny
bunny_wect = wect(bunny_complex)
print(bunny_wect)