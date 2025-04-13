import torch

def sublevelset_ecf(complex_values, num_values):
    """
    Calculates a discretization of the ECF of a sublevel set filtration.

    Args:
        complex_values: Function values on a cell complex, represented as a list
                        [0_cell_values, 1_cell_values, ...] of 1D tensors.
                        We assume all function values are between 0 and 1.
        num_values: The number of filtration steps to use in the discretization.

    Returns:
        ecf (torch.Tensor): A 1D tensor of shape (num_values) containing the sublevel set ECF.
    """
    diff_ecf = torch.zeros(num_values, dtype=torch.float32)
    for i, cell_values in enumerate(complex_values):
        cell_indices = _cell_indices(cell_values, num_values)
        diff_ecf.scatter_add_(0, cell_indices, ((-1) ** i) * torch.ones_like(cell_indices, dtype=torch.float32))
    return torch.cumsum(diff_ecf, dim=0)

def _cell_indices(cell_values, num_values):
    """
    Converts function values between 0 and 1 to indices in [0,1,... num_values - 1].
    """
    return torch.ceil(cell_values * (num_values-1)).long()

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
    z_square_values = torch.maximum(y_edge_values[1:, ...], y_edge_values[:-1, ...])
    square_values = torch.cat([
        x_square_values.reshape(-1),
        y_square_values.reshape(-1),
        z_square_values.reshape(-1)
    ], dim=0)

    cube_values = torch.maximum(x_square_values[1:, ...], x_square_values[:-1, ...]).reshape(-1)

    return [vertex_values, edge_values, square_values, cube_values]