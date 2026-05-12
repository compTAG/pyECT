"""For computing the WECF of an arbitrary (not neccesarily lower-star) filtration.
If your filtrations are lower-star (they usually are), then use wecfs.py instead.
"""

import torch
from typing import List, Tuple

def compute_wecfs_general(
    filtration_data: List[Tuple[torch.Tensor, torch.Tensor]],
    num_vals: int
) -> torch.Tensor:
    """Calculates WECFs for filtrations with values assigned to every simplex.

    Args:
        filtration_data: A weighted simplicial or cubical complex with a collection
            of filter functions defined on each simplex, represented as a list of
            pairs of tensors. The list index is the simplex dimension.

            filtration_data[i] = (simplex_filters, simplex_weights):
                simplex_filters (torch.Tensor): A tensor of shape (k_i, m), where
                    k_i is the number of i-simplices and m is the number of filter
                    functions. Each row contains the filter values of one simplex.

                simplex_weights (torch.Tensor): A tensor of shape (k_i). Values
                    are the weights of the i-simplices.

    Returns:
        wecfs (torch.Tensor): A 2d tensor of shape (m, num_vals)
            containing the WECFs.
    """

    if num_vals <= 0:
        raise ValueError("num_vals must be positive.")

    if len(filtration_data) == 0:
        raise ValueError("filtration_data must be non-empty.")

    device = filtration_data[0][0].device
    m = filtration_data[0][0].size(dim=1)
    eps = torch.finfo(torch.float32).eps

    max_val = torch.cat([f.reshape(-1) for f, _ in filtration_data]).max()
    min_val = torch.cat([f.reshape(-1) for f, _ in filtration_data]).min()
    val_range = torch.clamp(max_val - min_val, min=eps)

    diff_wecfs = torch.zeros((m, num_vals), dtype=torch.float32, device=device)

    for i, (simplex_filters, simplex_weights) in enumerate(filtration_data):
        simplex_indices = torch.ceil(
            (num_vals - 1) * (simplex_filters - min_val) / (val_range)
        ).clamp(0, num_vals-1).long()

        expanded_simplex_weights = (
            (-1) ** i * simplex_weights.unsqueeze(0).expand(m, -1)
        )

        diff_wecfs.scatter_add_(1, simplex_indices.T, expanded_simplex_weights)

    return torch.cumsum(diff_wecfs, dim=1)