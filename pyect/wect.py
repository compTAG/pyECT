"""For computing the WECT of a weighted geometric simplicial/cubical complex embedded in R^n."""

import torch
from typing import List, Tuple


class WECT(torch.nn.Module):
    """A torch module for computing the Weighted Euler Characteristic Transform (WECT) of a simplicial complex discretized over a grid.

    This module may be used just for computing the WECT, or used as a layer in a neural network.
    Internally, the module stores the directions and number of heights used for sampling, so repeated forward calls
    do not require these parameters to be passed in, and allow streamlined loading/saving of the module for consistent
    computation.

    This module can also be converted to TorchScript using torch.jit.script for use
    outside of Python.
    """

    def __init__(self, dirs: torch.Tensor, num_heights: int) -> None:
        """Initializes the WECT module.

        The initialized module is designed to compute the WECT of a simplicial complex
        embedded in R^[dirs.shape[1]], using dirs.shape[0] directions for sampling.
        The discretization of the WECT is parameterized by num_heights distinct height values.

        Args:
            dirs: An (d x n) tensor of directions to use for sampling.
            num_heights: A constant tensor, with the number of distinct height
                values to round to as an integer
        """
        super().__init__()
        dirs = torch.nn.functional.normalize(dirs, p=2, dim=1, eps=1e-12)
        self.register_buffer("dirs", dirs)
        self.num_heights: int = int(num_heights)

    def _vertex_indices(
        self,
        vertex_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the height values of each vertex and converts them to an index in range(num_heights).

        Args:
            vertex_coords (torch.Tensor): A tensor of shape (k_0, n) with rows representing the coordinates of the vertices.

        Returns:
            torch.Tensor: A tensor of shape (k_0, d) with the height indices of each vertex in each direction.
        """

        v_norms = torch.norm(vertex_coords, dim=1)
        max_height = torch.amax(v_norms)
        v_heights = torch.matmul(vertex_coords, self.dirs.T)

        # The case where all vertices are at the origin
        if max_height.item() == 0.0:
            return torch.zeros((v_heights.size(0), self.dirs.size(0)), dtype=torch.long, device=self.dirs.device)

        v_indices = torch.ceil(
            (self.num_heights - 1) * (max_height + v_heights) / (2.0 * max_height)
        ).clamp(0, self.num_heights - 1).long()

        return v_indices

    def forward(
        self,
        complex_data: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Calculates a discretization of the WECT of a complex embedded in n-dimensional space.

        Args:
            complex_data: A weighted simplicial or cubical complex, represented as a list of pairs of tensors.
                complex_data[0] = (v_coords, v_weights):
                    v_coords (torch.Tensor): A tensor of shape (k_0, n) where k_0 is the number of vertices.
                    Rows are the coordinates of the vertices.

                    v_weights (torch.Tensor): A tensor of shape (k_0). Values are the weights of the vertices.

                for i > 0:
                    complex_data[i] = (simp_verts, simp_weights):
                        simp_verts (torch.Tensor): A tensor of shape (k_i, i+1) where k_i is the number of i-simplices.
                        Rows are the vertex sets of the i-simplices.

                        simp_weights (torch.Tensor): A tensor of shape (k_i). Values are the weights of the i-simplices.

        Returns:
            wect (torch.Tensor): A 2d tensor of shape (self.dirs.shape[0], self.num_heights)
                containing the WECT.
        """

        d = self.dirs.size(dim=0)
        h = self.num_heights

        if h <= 0:
            raise ValueError("num_heights must be positive.")
        
        device = self.dirs.device
        v_coords  = complex_data[0][0].to(device=device, dtype=torch.float32)
        v_weights = complex_data[0][1].to(device=device, dtype=torch.float32)

        # Check for empty inputs
        if v_coords.size(0) == 0:
            return torch.zeros((d, h), dtype=torch.float32, device=device)

        expanded_v_weights = v_weights.unsqueeze(0).expand(
            d, -1
        )  # Expand to shape (d, k_0)

        # Initialize the differentiated WECT
        diff_wect = torch.zeros((d, h), dtype=torch.float32, device=device)

        # Compute the height index of each vertex
        v_indices = self._vertex_indices(v_coords)

        # Add the contribution of the vertices to the differentiated WECT
        diff_wect.scatter_add_(1, v_indices.T, expanded_v_weights)

        for i in range(1, len(complex_data)):
            simp_verts = complex_data[i][0].to(device=device, dtype=torch.long)
            simp_weights = complex_data[i][1].to(device=device, dtype=torch.float32)

            # Expand to shape (d, k_i)
            expanded_simp_weights = (-1) ** i * simp_weights.unsqueeze(0).expand(d, -1)

            # Compute the maximum index for each simplex's vertices
            simp_indices = v_indices[simp_verts]
            max_simp_indices = torch.amax(simp_indices, dim=1)

            # Add the contribution of the i-simplices to the differentiated WECT
            diff_wect.scatter_add_(1, max_simp_indices.T, expanded_simp_weights)

        return torch.cumsum(diff_wect, dim=1)