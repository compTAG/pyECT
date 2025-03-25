import os
import torch
from pyect import image_to_grayscale_tensor, weighted_freudenthal, compute_wect, compute_differentiated_wect, compute_diwect_2d, sample_directions_2d, scatter_wect

# Get the directory where this script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the image file
image_path = os.path.join(script_dir, "mona_lisa.jpg")

mona_lisa = image_to_grayscale_tensor(image_path)

vertices, higher_simplices = weighted_freudenthal(mona_lisa)
directions = sample_directions_2d(20)
num_heights = 100000

wect = compute_wect(vertices, higher_simplices, directions, num_heights) # The WECT computed the old way
scat_wect = scatter_wect(vertices, higher_simplices, directions, num_heights) # The WECT computed the new way

dif = torch.amax(wect - scat_wect) # The difference between the new WECT function and the old one. When num_heights is small, this can be substantial.
print("The max difference between the old wect and the new WECT is:")
print(dif)


# Checking that the last column of the WECT is (approximately) equal to the WEC

v_weights = vertices[1]
e_weights = higher_simplices[0][1]
t_weights = higher_simplices[1][1]

wec = v_weights.sum() - e_weights.sum() + t_weights.sum() 
wec.unsqueeze_(-1)

print("The max difference between the last column of the old WECT and the WEC is:")
print(torch.amax(wect[:, -1] - wec))

print("The max difference between the last column of the new WECT and the WEC is:")
print(torch.amax(scat_wect[:, -1] - wec))