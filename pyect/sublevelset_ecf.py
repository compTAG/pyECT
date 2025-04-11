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

def image_cell_values(img_arr):
    """
    Creates a cubical complex with a function on its cells from an image array.
    The structure of the cubical complex is ignored with only the function values on the cells
    being recorded.

    Args:
        img_arr (torch.Tensor): A 2D tensor with values between 0 and 1.

    Returns:
        vertex_values (torch.Tensor): A 1D tensor containing the function values of each vertex.
        edge_values (torch.Tensor): A 1D tensor containing the function values of each edge.
        square_values (torch.Tensor): A 1D tensor containing the function values of each square.
    """
    img_arr = img_arr.float()

    vertex_values = img_arr.reshape(-1)

    horizontal_edge_values = torch.maximum(img_arr[:, 1:], img_arr[:, :-1]).reshape(-1)
    vertical_edge_values = torch.maximum(img_arr[1:, :], img_arr[:-1, :]).reshape(-1)
    edge_values = torch.cat([horizontal_edge_values, vertical_edge_values], dim=0)

    # torch.maximum can only handle two tensors at once so the max is split into several steps.
    squares_1 = torch.maximum(img_arr[1:, 1:], img_arr[1:, :-1])
    squares_2 = torch.maximum(img_arr[:-1, 1:], img_arr[:-1, :-1])
    square_values = torch.maximum(squares_1, squares_2).reshape(-1) 

    return [vertex_values, edge_values, square_values]
