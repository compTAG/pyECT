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

    img_arr = torch.rand((500,500), device=device)

    # From here we can then take the image ECF.
    # We first initialize the Image_ECF_2D module.
    # Here we're choosing to sample the ECF over 6 bins.

    ecf = Image_ECF_2D(6).eval()

    # Next, we compute the image ECF of img_arr.

    ecf_result = ecf(img_arr)
    print(f"ECF: {ecf_result}")



    # We can also compute the WECT of img_arr.
    # To do so, we first sample direction vectors.
    # We'll pick 5 directions for this example.

    directions = sample_directions_2d(5, device=device)

    # Next, we initialize the WECT module with the direction vectors we just sampled.
    # We're choosing to sample 6 height values here.

    wect = WECT(directions, 6).eval()

    # Now we compute the weighted Freudenthal complex of img_arr.

    img_complex = weighted_freudenthal(img_arr, device=device)

    # Finally, we compute the wect of the resulting simplicial complex.

    wect_result = wect(img_complex)
    print(f"WECT: {wect_result}")

if __name__ == "__main__":
    main()