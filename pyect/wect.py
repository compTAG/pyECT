import torch
from typing import List, Tuple

# Automatically select the best available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WECT(torch.nn.Module):
    """A torch module for computing the Weighted Euler Characteristic Transform (WECT) of a simplicial complex discretized over a grid.

    This module may be used just for computing the WECT, or used as a layer in a neural network.
    Internally, the module stores the directions and number of heights used for sampling, so repeated forward calls
    do not require these parameters to be passed in, and allow streamlined loading/saving of the module for consistent
    computation.

    This module can also be converted to TorchScrpt using torch.jit.script for use
    outside of Python.
    """

    def __init__(self, dirs: torch.Tensor, num_heights: int) -> None:
        """Initializes the WECT module.

        The initialized module is designed to compute the WECT of a simplicial complex
        embedded in R^[dirs.shape[1]], using dirs.shape[0] directiions for sampling.
        The discretization of the WECT is parameterized by num_heights distinct height values.

        Args:
            dirs: An (n x d) tensor of directions to use for sampling.
            num_heights: A constant tensor, with the number of distinct height
                values to round to as an integer
        """
        super().__init__()
        self.dirs = torch.nn.Parameter(dirs, requires_grad=False)
        height_tensor = torch.Tensor([num_heights], device=dirs.device)
        self.num_heights = torch.nn.Parameter(height_tensor, requires_grad=False)

    def _vertex_indices(
        self,
        vertex_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the height values of each vertex and converts them to an index in range(num_heights).

        Assumes that directions are unit vectors.

        Args:
            vertex_coords (torch.Tensor): A tensor of shape (k_0, n+1) with rows representing the coordinates of the vertices.

        Returns:
            torch.Tensor: A tensor of shape (k_0, d) with the height indices of each vertex in each direction.
        """
        v_norms = torch.norm(vertex_coords, dim=1)
        max_height = v_norms.max()

        v_heights = torch.matmul(vertex_coords, self.dirs.T)
        v_indices = torch.ceil(
            (self.num_heights - 1) * (max_height + v_heights) / (2.0 * max_height)
        ).long()
        return v_indices

    def forward(
        self,
        complex: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Calculates a discretization of the differentiated WECT of a simplicial complex embedded in (n+1)-dimensional space.

        Args:
            complex: A weighted simplicial complex, represented as a list (simplex, weight) pairs of tensors.
                The first element of the list is the vertex set, and the remaining elements are simplices of increasing dimension.

        Returns:
            wect (tf.Tensor): A 2d tensor of shape (self.directions.shape[0], self.num_heights)
                containing the WECT.
        """

        d = self.dirs.size(dim=0)
        h = int(self.num_heights)

        v_coords, v_weights = complex[0]
        expanded_v_weights = v_weights.unsqueeze(0).expand(
            d, -1
        )  # Expand to shape (d, k_0)

        # Initialize the differentiated WECT
        diff_wect = torch.zeros((d, h), dtype=v_weights.dtype, device=v_weights.device)

        # Compute the height index of each vertex
        v_indices = self._vertex_indices(v_coords)

        # Add the contribution of the vertices to the differentiated WECT
        diff_wect.scatter_add_(1, v_indices.t(), expanded_v_weights)

        for i in range(1, len(complex)):
            simp_verts, simp_weights = complex[i]
            expanded_simp_weights = (-1) ** (i + 1) * simp_weights.unsqueeze(0).expand(
                d, -1
            )

            simp_indices = v_indices[simp_verts.to(v_indices.device)]
            max_simp_indices = torch.amax(simp_indices, dim=1)

            diff_wect.scatter_add_(1, max_simp_indices.t(), expanded_simp_weights)

        return torch.cumsum(diff_wect, dim=1)


"""
TODO: rewrite the forward function arguments

Args:
    vertices (torch.Tensor, torch.Tensor): A tuple of tensors:
        vertices[0]: A tensor of shape (k_0, n+1) with rows the coordinates of the vertices.
        vertices[1]: A tensor of shape (k_0) containing the vertex weights.
    higher_simplices: A tuple of tuples of tensors:
        higher_simplices[i] (torch.Tensor, torch.Tensor):
            higher_simplices[i,0]: A tensor of shape (k_{i+1}, i+2) containing the vertices of the (i+1)-simplices.
            higher_simplices[i,1]: A tensor of shape (k_{i+1}) containing the weights of the (i+1)-simplices.
    directions (torch.Tensor): A tensor of shape (d, n+1) with rows the sampled direction vectors.
    num_heights (Int): The number of height values to sample.

Returns:
    diff_wect (torch.Tensor): A tensor of shape (d, num_heights) containing a discretization of the differentiated WECT.
"""
