import torch

pi = torch.pi
golden_angle = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))

def sample_directions_2d(num_dirs):
    angles = 2 * pi * torch.arange(num_dirs, dtype=torch.float32) / num_dirs
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return directions

def sample_directions_3d(num_dirs):

    i = torch.arange(num_dirs, dtype=torch.float32)
    theta = golden_angle * i
    y = torch.linspace(1.0, -1.0, num_dirs)
    r = torch.sqrt(1.0 - y**2)
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    directions = torch.stack([x, y, z], dim=1)

    return directions