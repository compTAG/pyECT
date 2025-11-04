"""For computing the ECF of 2- and 3-dimensional images filtered by pixel intensity"""

import torch
from typing import List, Tuple

class image_ECF_2D(torch.nn.Module):
    """A torch module for computing the ECF of an image (2- or 3-dimensional) filtered by pixel intensity.

    This module may be used just for computing the ECF of images, or used as a layer in a neural network.
    Internally, the module stores the number of values used for sampling, so repeated forward calls
    do not require this parameters to be passed in, and allow streamlined loading/saving of the module for consistent
    computation.

    This module can also be converted to TorchScript using torch.jit.script for use
    outside of Python.
    """

    def __init__(self, num_vals: int) -> None:
        """Initializes the image_ECF module.

        The initialized module is designed to compute the ECF of an image, discretized by sampling num_vals values.

        Args:
            num_vals: The number of values to discretize the ECF over.
        """
        super().__init__()
        self.num_vals: int = int(num_vals)

    def cell_values_2D(arr):
        """
        Creates a cubical complex with a function on its cells from a 2D tensor.
        The structure of the cubical complex is ignored with only the function values on the cells
        being recorded.

        Args:
            arr (torch.Tensor): A 2D tensor.

        Returns:
            vertex_values (torch.Tensor): A 1D tensor containing the function values of each vertex.
            edge_values (torch.Tensor): A 1D tensor containing the function values of each edge.
            square_values (torch.Tensor): A 1D tensor containing the function values of each square.
        """
        arr = arr.float()

        vertex_values = arr.reshape(-1)

        x_edge_values = torch.maximum(arr[1:, :], arr[:-1, :])
        y_edge_values = torch.maximum(arr[:, 1:], arr[:, :-1])
        edge_values = torch.cat([
            x_edge_values.reshape(-1),
            y_edge_values.reshape(-1)
        ], dim=0)

        square_values = torch.maximum(y_edge_values[1:, :], y_edge_values[:-1, :]).reshape(-1)

        return [vertex_values, edge_values, square_values]










def image_ecf(complex_values, filt_steps):
    """
    Calculates a discretization of the ECF of a sublevel set filtration.

    Args:
        complex_values: Function values on a cell complex, represented as a list
                        [0_cell_values, 1_cell_values, ...] of 1D tensors.
                        We assume all function values are between 0 and 1.
        filt_steps: The number of filtration steps to use in the discretization.

    Returns:
        ecf (torch.Tensor): A 1D tensor of shape (filt_steps) containing the sublevel set ECF.
    """
    diff_ecf = torch.zeros(filt_steps, dtype=torch.int32)

    for i, cell_values in enumerate(complex_values):
        cell_indices = torch.ceil(cell_values * (filt_steps-1)).long()
        diff_ecf.scatter_add_(
            0,
            cell_indices,
            ((-1) ** i) * torch.ones_like(cell_indices, dtype=torch.int32)
        )
    return torch.cumsum(diff_ecf, dim=0)


def cell_values_2D(arr):
    """
    Creates a cubical complex with a function on its cells from a 2D tensor.
    The structure of the cubical complex is ignored with only the function values on the cells
    being recorded.

    Args:
        arr (torch.Tensor): A 2D tensor.

    Returns:
        vertex_values (torch.Tensor): A 1D tensor containing the function values of each vertex.
        edge_values (torch.Tensor): A 1D tensor containing the function values of each edge.
        square_values (torch.Tensor): A 1D tensor containing the function values of each square.
    """
    arr = arr.float()

    vertex_values = arr.reshape(-1)

    x_edge_values = torch.maximum(arr[1:, :], arr[:-1, :])
    y_edge_values = torch.maximum(arr[:, 1:], arr[:, :-1])
    edge_values = torch.cat([
        x_edge_values.reshape(-1),
        y_edge_values.reshape(-1)
    ], dim=0)

    square_values = torch.maximum(y_edge_values[1:, :], y_edge_values[:-1, :]).reshape(-1)

    return [vertex_values, edge_values, square_values]

def cell_values_3D(arr):
    """
    Creates a cubical complex with a function on its cells from a 3D tensor.
    The structure of the cubical complex is ignored with only the function values on the cells
    being recorded.

    Args:
        arr (torch.Tensor): A 3D tensor.

    Returns:
        vertex_values (torch.Tensor): A 1D tensor containing the function values of each vertex.
        edge_values (torch.Tensor): A 1D tensor containing the function values of each edge.
        square_values (torch.Tensor): A 1D tensor containing the function values of each square.
        cube_values (torch.Tensor): A 1D tensor containing the function values of each cube.
    """
    arr = arr.float()

    vertex_values = arr.reshape(-1)

    x_edge_values = torch.maximum(arr[1:, ...], arr[:-1, ...])
    y_edge_values = torch.maximum(arr[:, 1:, :], arr[:, :-1, :])
    z_edge_values = torch.maximum(arr[..., 1:], arr[..., :-1])
    edge_values = torch.cat([
        x_edge_values.reshape(-1),
        y_edge_values.reshape(-1),
        z_edge_values.reshape(-1)
        ], dim=0)

    x_square_values = torch.maximum(y_edge_values[..., 1:], y_edge_values[..., :-1])
    y_square_values = torch.maximum(z_edge_values[1:, ...], z_edge_values[:-1, ...])
    z_square_values = torch.maximum(x_edge_values[:, 1:, :], x_edge_values[:, :-1, :])
    square_values = torch.cat([
        x_square_values.reshape(-1),
        y_square_values.reshape(-1),
        z_square_values.reshape(-1)
    ], dim=0)

    cube_values = torch.maximum(x_square_values[1:, ...], x_square_values[:-1, ...]).reshape(-1)

    return [vertex_values, edge_values, square_values, cube_values]
