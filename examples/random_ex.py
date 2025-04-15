import torch
import timeit
from typing import List, Generator
from pyect import (
    weighted_freudenthal,
    WECT,
    compute_diwect_2d,
    sample_directions_2d,
)

image_size = (300, 300)
directions = sample_directions_2d(20)
num_heights = 10000

img_arr = torch.rand(image_size)

#### CPU timing

device = torch.device("cpu")
complex = weighted_freudenthal(img_arr, device=device)
directions = directions.to(device)

## Unscripted
wect = WECT(directions, num_heights).eval()
# burn in
for i in range(5):
    wect.forward(list(complex.dimensions))

start = timeit.default_timer()
uncompiled_output = wect.forward(list(complex.dimensions))
end = timeit.default_timer()
print("Uncompiled output: ", uncompiled_output)
print(f"Time on {device} uncompiled: {end - start}")

## Scripted
scripted_wect = torch.jit.script(wect)
# Burn in
for i in range(5):
    scripted_wect.forward(list(complex.dimensions))

start = timeit.default_timer()
scripted_output = scripted_wect.forward(list(complex.dimensions))
end = timeit.default_timer()
print("Scripted output: ", scripted_output)
print(f"Time on {device} scripted: {end - start}")

#### CUDA

if torch.cuda.is_available():
    device = torch.device("cuda")
    complex = weighted_freudenthal(img_arr, device=device)
    directions = directions.to(device)

    ## Unscripted
    wect = WECT(directions, num_heights).eval()
    # Burn in
    for i in range(5):
        wect.forward(list(complex.dimensions))

    torch.cuda.synchronize()
    start = timeit.default_timer()
    uncompiled_output = wect.forward(list(complex.dimensions))
    torch.cuda.synchronize()
    end = timeit.default_timer()

    print("Uncompiled output: ", uncompiled_output)
    print(f"Time on {device} uncompiled: {end - start}")

    ## Scripted
    scripted_wect = torch.jit.script(wect)
    # Burn in
    for i in range(5):
        scripted_wect.forward(list(complex.dimensions))
    
    torch.cuda.synchronize()

    start = timeit.default_timer()

    scripted_output = scripted_wect.forward(list(complex.dimensions))
    torch.cuda.synchronize()

    end = timeit.default_timer()

    print("Scripted output: ", scripted_output)
    print(f"Time on {device} scripted: {end - start}")


#### Apple

if torch.backends.mps.is_available():
    device = torch.device("mps")
    complex = weighted_freudenthal(img_arr, device=device)
    directions = directions.to(device)

    ## Unscripted
    wect = WECT(directions, num_heights).eval()
    # Burn in
    for i in range(5):
        wect.forward(list(complex.dimensions))
        torch.mps.synchronize()

    start = timeit.default_timer()
    uncompiled_output = wect.forward(list(complex.dimensions))
    torch.mps.synchronize()
    end = timeit.default_timer()

    print("Uncompiled output: ", uncompiled_output)
    print(f"Time on {device} uncompiled: {end - start}")

    ## Scripted
    scripted_wect = torch.jit.script(wect)
    # Burn in
    for i in range(5):
        scripted_wect.forward(list(complex.dimensions))
        torch.mps.synchronize()

    start = timeit.default_timer()

    scripted_output = scripted_wect.forward(list(complex.dimensions))
    torch.mps.synchronize()

    end = timeit.default_timer()

    print("Scripted output: ", scripted_output)
    print(f"Time on {device} scripted: {end - start}")