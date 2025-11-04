"""For computing the ECF of 2- and 3-dimensional images filtered by pixel intensity"""

import torch
from typing import List

class Image_ECF_2D(torch.nn.Module):
    """A torch module for computing the ECF of a 2D image filtered by pixel intensity.

    This module may be used just for computing the ECF of images, or used as a layer in a neural network.
    Internally, the module stores the number of values used for sampling, so repeated forward calls
    do not require this parameters to be passed in, and allow streamlined loading/saving of the module for consistent
    computation.

    This module can also be converted to TorchScript using torch.jit.script for use
    outside of Python.
    """

    def __init__(self, num_vals: int) -> None:
        """Initializes the image_ECF module.

        The initialized module is designed to compute the ECF of a 2D image, discretized by sampling num_vals values.

        Args:
            num_vals: The number of values to discretize the ECF over.
        """
        super().__init__()
        self.num_vals: int = int(num_vals)

    @staticmethod
    def cell_values_2D(arr: torch.Tensor) -> List[torch.Tensor]:
        """
        Creates a cubical complex with a function on its cells from a 2D tensor.
        The structure of the cubical complex is ignored with only the function values on the cells
        being recorded.

        Args:
            arr (torch.Tensor): A 2D tensor with values between 0 and 1.

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
    
    def forward(self, img_arr: torch.Tensor) -> torch.Tensor:
        """
        Calculates a discretization of the ECF of a 2D image.

        Args:
            img_arr (torch.Tensor): a 2D tensor with values between 0 and 1.

        Returns:
            ecf (torch.Tensor): A 1D tensor of shape (self.num_vals) containing the ECF.
        """

        device = img_arr.device
        n = self.num_vals
        vertex_values, edge_values, square_values = self.cell_values_2D(img_arr)

        vertex_indices = torch.ceil(vertex_values * (n-1)).long()
        edge_indices = torch.ceil(edge_values * (n-1)).long()
        square_indices = torch.ceil(square_values * (n-1)).long()

        diff_ecf = torch.zeros(n, dtype=torch.int32, device=device)

        # Add the contribution of the vertices
        diff_ecf.scatter_add_(
            0,
            vertex_indices,
            torch.ones_like(vertex_indices, dtype=torch.int32)
        )

        # Add the contribution of the edges
        diff_ecf.scatter_add_(
            0,
            edge_indices,
            -1 * torch.ones_like(edge_indices, dtype=torch.int32)
        )

        # Add the contribution of the squares
        diff_ecf.scatter_add_(
            0,
            square_indices,
            torch.ones_like(square_indices, dtype=torch.int32)
        )

        return torch.cumsum(diff_ecf, dim=0)


class Image_ECF_3D(torch.nn.Module):
    """A torch module for computing the ECF of a 3D image filtered by pixel intensity.

    This module may be used just for computing the ECF of images, or used as a layer in a neural network.
    Internally, the module stores the number of values used for sampling, so repeated forward calls
    do not require this parameters to be passed in, and allow streamlined loading/saving of the module for consistent
    computation.

    This module can also be converted to TorchScript using torch.jit.script for use
    outside of Python.
    """

    def __init__(self, num_vals: int) -> None:
        """Initializes the image_ECF module.

        The initialized module is designed to compute the ECF of a 3D image, discretized by sampling num_vals values.

        Args:
            num_vals: The number of values to discretize the ECF over.
        """
        super().__init__()
        self.num_vals: int = int(num_vals)

    @staticmethod
    def cell_values_3D(arr: torch.Tensor) -> List[torch.Tensor]:
        """
        Creates a cubical complex with a function on its cells from a 3D tensor.
        The structure of the cubical complex is ignored with only the function values on the cells
        being recorded.

        Args:
            arr (torch.Tensor): A 3D tensor with values between 0 and 1.

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

    
    def forward(self, img_arr: torch.Tensor) -> torch.Tensor:
        """
        Calculates a discretization of the ECF of a 3D image.

        Args:
            img_arr (torch.Tensor): A 3D tensor with values between 0 and 1.

        Returns:
            ecf (torch.Tensor): A 1D tensor of shape (self.num_vals) containing the sublevel set ECF.
        """

        device = img_arr.device
        n = self.num_vals
        vertex_values, edge_values, square_values, cube_values = self.cell_values_3D(img_arr)

        vertex_indices = torch.ceil(vertex_values * (n-1)).long()
        edge_indices = torch.ceil(edge_values * (n-1)).long()
        square_indices = torch.ceil(square_values * (n-1)).long()
        cube_indices = torch.ceil(cube_values * (n-1)).long()

        diff_ecf = torch.zeros(n, dtype=torch.int32, device=device)

        # Add the contribution of the vertices
        diff_ecf.scatter_add_(
            0,
            vertex_indices,
            torch.ones_like(vertex_indices, dtype=torch.int32)
        )

        # Add the contribution of the edges
        diff_ecf.scatter_add_(
            0,
            edge_indices,
            -1 * torch.ones_like(edge_indices, dtype=torch.int32)
        )

        # Add the contribution of the squares
        diff_ecf.scatter_add_(
            0,
            square_indices,
            torch.ones_like(square_indices, dtype=torch.int32)
        )

        # Add the contribution of the cubes
        diff_ecf.scatter_add_(
            0,
            cube_indices,
            -1 * torch.ones_like(cube_indices, dtype=torch.int32)
        )

        return torch.cumsum(diff_ecf, dim=0)