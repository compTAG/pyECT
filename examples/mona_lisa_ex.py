import os
import torch
import timeit
from pyect import image_to_grayscale_tensor, weighted_freudenthal, compute_wect, compute_diwect_2d, sample_directions_2d

# Get the directory where this script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the image file
image_path = os.path.join(script_dir, "mona_lisa.jpg")

mona_lisa = image_to_grayscale_tensor(image_path)

vertices, higher_simplices = weighted_freudenthal(mona_lisa)
directions = sample_directions_2d(5)
num_heights = 1000

wect = compute_wect(vertices, higher_simplices, directions, num_heights)
print(wect)