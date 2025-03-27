import os
import torch
import timeit
from typing import List, Generator
from pyect import (
    image_to_grayscale_tensor,
    weighted_freudenthal,
    WECT,
    compute_diwect_2d,
    sample_directions_2d,
)

devices = (
    [torch.device("cpu"), torch.device("cuda")]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)

# Get the directory where this script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the image file
image_path = os.path.join(script_dir, "mona_lisa.jpg")

for device in devices:
    mona_lisa = image_to_grayscale_tensor(image_path, device)

    complex = weighted_freudenthal(mona_lisa)

    directions = sample_directions_2d(5).to(device)
    num_heights = 1000

    wect = WECT(directions, num_heights).eval()

    start = timeit.default_timer()
    uncompiled_output = wect.forward(list(complex.dimensions))
    end = timeit.default_timer()
    print("Uncompiled output: ", uncompiled_output)
    print(f"Time on {device} uncompiled: {end - start}")

    scripted_wect = torch.jit.script(wect)

    start = timeit.default_timer()
    scripted_output = scripted_wect.forward(list(complex.dimensions))
    end = timeit.default_timer()
    print("Scripted output: ", scripted_output)
    print(f"Time on {device} scripted: {end - start}")

    # traced_wect = torch.jit.trace(wect, (list(complex.dimensions),))
    #
    # start = timeit.default_timer()
    # traced_output = traced_wect.forward(list(complex.dimensions))
    # end = timeit.default_timer()
    # print(f"Time on {device} traced: {end - start}")
    #
    # compiled_wect = torch.compile(wect, backend="tensorrt")
    # compiled_output = compiled_wect.forward(list(complex.dimensions))
    # end = timeit.default_timer()
    # print(f"Time on {device} compiled: {end - start}")


def image_wect(
    grayscaletensors: List[torch.Tensor], wect_module: WECT, device: torch.device
) -> Generator[torch.Tensor, None, None]:
    """Compute the WECT of a sequence of grayscale images.

    Args:
        grayscaletensors: A sequence of grayscale images.
        wect_module (WECT): The WECT module to use for computation.

    Yields:
        torch.Tensor: The WECT of each grayscale image.
    """
    # if device == torch.device("cuda"): # PERF: At one point I added this, can't remember why so uncommenting for now
    #     torch.set_float32_matmul_precision("high")

    for gt in grayscaletensors:
        image_tensor = torch.squeeze(gt.div(255.0))

        complex = weighted_freudenthal(
            image_tensor,
            device=device,
        )

        simplices = complex.dimensions
        im_wect = wect_module.forward(list(simplices))

        yield im_wect
