import math
import torch

golden_angle = math.pi * (3.0 - math.sqrt(5.0))

def sample_directions_2d(num_dirs: int, *, device=None):
    """
    Sample num_dirs directions evenly from S^1.
    """

    angles = 2 * math.pi * torch.arange(num_dirs, dtype=torch.float32, device=device) / num_dirs
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return directions.contiguous()

def sample_directions_3d(num_dirs: int, *, device=None):
    """
    Sample num_dirs directions from S^2 using the Fibonacci spiral method.
    """

    i = torch.arange(num_dirs, dtype=torch.float32, device=device)
    theta = golden_angle * i
    y = 1.0 - (2.0 * (i + 0.5) / num_dirs)
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    directions = torch.stack([x, y, z], dim=-1)

    return directions.contiguous()