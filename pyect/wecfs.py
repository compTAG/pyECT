""" For computing the WECFs of lower-star filtrations of
weighted simplicial/cubical complex with respect to a set of filter functions."""

import torch
from typing import List, Tuple

def compute_wecfs(
    complex_data: List[Tuple[torch.Tensor, torch.Tensor]],
    num_vals: int
) -> torch.Tensor:
    """Calculates a discretization of the WECFs of a weighted complex with respect to a set of filter functions.

    Args:
        complex_data: A weighted simplicial or cubical complex with a collection of filter functions,
        represented as a list of pairs of tensors.
            complex_data[0] = (filters, v_weights):
                filters (torch.Tensor): A tensor of shape (k_0, m) where k_0 is the
                    number of vertices and m is the number of filter functions.
                    Each column contains the values of a filter function on the vertices.

                v_weights (torch.Tensor): A tensor of shape (k_0). Values are the weights of the vertices.

            for i > 0:
                complex_data[i] = (simp_verts, simp_weights):
                    simp_verts (torch.Tensor): A tensor of shape (k_i, i+1) where k_i is the number of i-simplices.
                        Rows are the vertex sets of the i-simplices.

                    simp_weights (torch.Tensor): A tensor of shape (k_i). Values are the weights of the i-simplices.

    Returns:
        wecfs (torch.Tensor): A 2d tensor of shape (m, num_vals)
            containing the WECFs.
    """

    filters = complex_data[0][0].float()
    m = filters.size(dim=1)
    device = filters.device
    v_weights = complex_data[0][1].to(device=device, dtype=torch.float32)

    expanded_v_weights = v_weights.unsqueeze(0).expand(m, -1)  # Expand to shape (m, k_0)

    # Map the values of the filter functions to indices in range(num_vals)
    max_val = filters.abs().amax()
    v_indices = torch.ceil(
        (num_vals - 1) * (max_val + filters) / (2.0 * max_val)
    ).clamp(0, num_vals-1).long()

    # Initialize the differentiated WECFs
    diff_wecfs = torch.zeros((m, num_vals), dtype=torch.float32, device=device)

    # Add the contribution of the vertices to the differentiated WECFs
    diff_wecfs.scatter_add_(1, v_indices.T, expanded_v_weights)

    for i in range(1, len(complex_data)):
        simp_verts = complex_data[i][0].to(device=device, dtype=torch.long)
        simp_weights = complex_data[i][1].to(device=device, dtype=torch.float32)

        expanded_simp_weights = (-1) ** i * simp_weights.unsqueeze(0).expand(m, -1)

        simp_indices = v_indices[simp_verts]
        max_simp_indices = torch.amax(simp_indices, dim=1)

        diff_wecfs.scatter_add_(1, max_simp_indices.T, expanded_simp_weights)

    return torch.cumsum(diff_wecfs, dim=1)