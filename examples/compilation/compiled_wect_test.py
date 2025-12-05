"""Testing the effects of compiling the WECT"""

import torch
import timeit
from pyect import (
    weighted_freudenthal,
    sample_directions_2d,
    WECT
)

torch.set_grad_enabled(False)
device = torch.device("cpu")

img_sizes = [(30, 30), (100, 100), (300, 300), (500, 500), (800, 800), (1000, 1000), (1500,1500), (2000, 2000)]
num_heights = 1000
num_directions = 50
directions = sample_directions_2d(num_directions, device=device)
n_tests = 10

wect = WECT(directions, num_heights).eval()
compiled_wect = torch.compile(wect, backend="inductor", mode="max-autotune", dynamic=True)

for img_size in img_sizes:
    print(f"Testing on {img_size} images:")
    img_complexes = []
    for i in range(n_tests):
        img_complex = weighted_freudenthal(torch.rand(img_size), device=device)
        img_complexes.append(img_complex)

    ###### Compiled ######

    # Warmup
    for i in range(n_tests):
        compiled_output = compiled_wect(img_complexes[i])

    # Test
    start = timeit.default_timer()
    for i in range(n_tests):
        compiled_output = compiled_wect(img_complexes[i])
    end = timeit.default_timer()

    print(f"Compiled avg time: {(end - start)/n_tests}")

    ###### Uncompiled ######

    # Warmup
    for i in range(n_tests):
        uncompiled_output = wect(img_complexes[i])

    # Test
    start = timeit.default_timer()
    for i in range(n_tests):
        uncompiled_output = wect(img_complexes[i])
    end = timeit.default_timer()

    print(f"Uncompiled avg time: {(end - start)/n_tests}")
    print()