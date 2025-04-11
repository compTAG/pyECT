import torch
import timeit
from pyect import image_to_grayscale_tensor, weighted_freudenthal, compute_wect, compute_diwect_2d, sample_directions_2d, sample_directions_3d

arr = torch.rand([400,400])
vertices, higher_simplices = weighted_freudenthal(arr)
directions = sample_directions_2d(20)
num_heights = 1000

num_simps = vertices[1].size(dim=0) + higher_simplices[0][1].size(dim=0) + higher_simplices[1][1].size(dim=0)
print("Number of simplices:")
print(num_simps)

time_taken = timeit.timeit('compute_wect(vertices, higher_simplices, directions, num_heights)', globals=globals(), number=5)
print(f"Average time per run: {time_taken / 5:.6f} seconds")

dirs = sample_directions_3d(5)
print(dirs.norm())