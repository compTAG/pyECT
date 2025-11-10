"""Example showing how to apply the WECT and image ECF to an image array."""

import torch
from pyect import (
    weighted_freudenthal,
    sample_directions_2d,
    WECT,
    Image_ECF_2D,
    image_to_grayscale_tensor
)

def main():
    # First, choose a device.
    # Leave one of the following three lines uncommented to pick that device (if you have it):

    device = torch.device("cpu")       # default option
    #device = torch.device("cuda")     # if you have a Nvidia GPU
    #device = torch.device("mps")      # if you have an Apple m-series chip


    # We'll use a randomized array rather than an image file.
    # To apply this to an image file, use the function image_to_grayscale_tensor.
    # The resulting tensor can then be used in place of img_arr.

    img_size = (500, 500)
    img_arr = torch.rand(img_size, device=device)

    # Next, we'll choose the parameters for the image ECF and WECT.

    num_bins = 100          # The number of bins to discretize (W)ECFs over.
    num_directions = 25     # The number of directions to sample the WECT over.
    directions = sample_directions_2d(num_directions, device=device)

    ### Image ECF ###

    # Now we initialize the Image_ECF_2D module.

    ecf = Image_ECF_2D(num_bins).eval()

    # Then, we compute the image ECF of img_arr.

    ecf_result = ecf(img_arr)
    print(f"ECF: {ecf_result}")

    ### WECT ###

    # We can also compute the WECT of img_arr.
    # We first initialize the WECT module.

    wect = WECT(directions, num_bins).eval()

    # Now we compute the weighted Freudenthal complex of img_arr.

    img_complex = weighted_freudenthal(img_arr, device=device)

    # Finally, we compute the wect of the resulting simplicial complex.

    wect_result = wect(img_complex)
    print(f"WECT: {wect_result}")

if __name__ == "__main__":
    main()