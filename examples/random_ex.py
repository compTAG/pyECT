import torch
import timeit
from typing import List, Generator
from pyect import (
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

for device in devices:
    arr = torch.rand([4000,4000],device=device)

    complex = weighted_freudenthal(arr)

    directions = sample_directions_2d(20).to(device)
    num_heights = 1000

    ########################################################
    ## Run the WECT with different compilation strategies ##
    ########################################################

    ## Uncompiled
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

    # ## Traced
    # traced_wect = torch.jit.trace(wect, (list(complex.dimensions),))
    # # Burn in
    # for i in range(5):
    #     traced_wect.forward(list(complex.dimensions))
    #
    # start = timeit.default_timer()
    # traced_output = traced_wect.forward(list(complex.dimensions))
    # end = timeit.default_timer()
    # print(f"Time on {device} traced: {end - start}")