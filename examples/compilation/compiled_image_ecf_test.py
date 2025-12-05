"""Testing the effects of compiling the Image ECF"""

import torch
import timeit
from pyect import Image_ECF_2D

torch.set_grad_enabled(False)
device = torch.device("cpu")

img_sizes = [(500, 500), (1000, 1000), (3000, 3000), (5000, 5000), (10000, 10000), (15000, 15000)]
num_vals = 256
n_tests = 10

ecf = Image_ECF_2D(num_vals).eval()
compiled_ecf = torch.compile(ecf, backend="inductor", mode="max-autotune", dynamic=False)

print()

for img_size in img_sizes:
    print(f"Testing on {img_size} images:")
    imgs = []
    for i in range(n_tests):
        imgs.append(torch.rand((img_size), device=device))

    ###### Compiled ######

    # Warmup
    for img in imgs:
        compiled_output = compiled_ecf(img)

    # Test
    start = timeit.default_timer()
    for img in imgs:
        compiled_output = compiled_ecf(img)
    end = timeit.default_timer()

    print(f"Compiled avg time: {(end - start)/n_tests}")

    ###### Uncompiled ######

    # Warmup
    for img in imgs:
        uncompiled_output = ecf(img)

    # Test
    start = timeit.default_timer()
    for img in imgs:
        uncompiled_output = ecf(img)
    end = timeit.default_timer()

    print(f"Uncompiled avg time: {(end - start)/n_tests}")
    print()