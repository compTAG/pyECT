import torch
import pytest

from pyect import WECT
from pyect import Complex


def build_triangle_complexes(device="cpu"):
    vcoords = torch.tensor(
        [[-1.0, 0.0],
         [ 0.0, 1.0],
         [ 1.0, 0.0]], device=device
    )
    vweights_ones = torch.ones(3, device=device)
    vweights = torch.tensor([0.5, 1.0, 1.5], device=device)

    ecoords = torch.tensor(
        [[0, 1],
         [1, 2],
         [2, 0]], device=device
    )
    eweights_ones = torch.ones(3, device=device)
    eweights = torch.tensor([.5, 1.0, 0.5], device=device)

    fcoords = torch.tensor([[0, 1, 2]], device=device)
    fweights_ones = torch.ones(1, device=device)
    fweights = torch.tensor([0.5], device=device)

    return Complex(
        (vcoords, vweights_ones),
        (ecoords, eweights_ones),
        (fcoords, fweights_ones),
    ), Complex(
        (vcoords, vweights),
        (ecoords, eweights),
        (fcoords, fweights),
    ), 


# ----------------------------------------------------------
# WECT TEST
# ----------------------------------------------------------

def test_wect_exact_triangle_direction_10():
    device = torch.device("cpu")

    complex_unweighted, complex_weighted = build_triangle_complexes(device=device)


    # UNWEIGHTED COMPLEX
    # single direction (1,0)
    dirs = torch.tensor([[1.0, 0.0]], device=device)

    num_bins = 3
    wect = WECT(dirs, num_bins).eval()

    result = wect(complex_unweighted)  # shape (1,3)

    assert result.shape == (1, num_bins)
    assert torch.isfinite(result).all()

    expected = torch.tensor([1.0, 1.0, 1.0], device=device)

    assert torch.allclose(result[0], expected, atol=1e-6)

    # WEIGHTED COMPLEX
    result = wect(complex_weighted)  # shape (1,3)
    assert result.shape == (1, num_bins)
    assert torch.isfinite(result).all()

    expected = torch.tensor([0.5, 1.0, 1.5], device=device)
    assert torch.allclose(result[0], expected, atol=1e-6)
