import torch

pi = torch.pi

def sample_directions_2d(num_dirs):
    angles = 2 * pi * torch.arange(num_dirs, dtype=torch.float32) / num_dirs
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return directions

def sample_directions_3d(num_dirs):

    return