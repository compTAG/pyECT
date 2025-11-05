from typing import Optional
import torch
import torchvision.transforms as transforms
from pyect import Complex
from PIL import Image


def image_to_grayscale_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    # Open the image using PIL
    image = Image.open(image_path)
    # Convert the image to grayscale (mode 'L')
    grayscale_image = image.convert("L")
    # Convert the grayscale image to a tensor with values in [0,1]
    tensor = transforms.ToTensor()(grayscale_image).squeeze(dim=0)
    # The resulting tensor will have shape (H, W)
    return tensor.to(device)


def weighted_freudenthal(
    img_arr: torch.Tensor, device: Optional[torch.device] = None
) -> Complex:
    """
    Creates the weighted Freudenthal complex of an image array using a max function extension.
    Discards edges and triangles that have a vertex with a zero weight.
    By default, the device of the input tensor is used unless a different device is specified.

    The vertices are a (h*w, 2) tensor with recentered pixel coordinates.
    The vertex weights are a (h*w,) tensor containing the pixel intensities.
    The edges are a (num_valid_edges, 2) tensor of vertex indices.
    The edge weights are a (num_valid_edges,) tensor with the maximum weight on the edge.
    The triangles are a (num_valid_triangles, 3) tensor of vertex indices.
    The triangle weights are a (num_valid_triangles,) tensor with the maximum weight on the triangle.

    Args:
        img_arr (torch.Tensor): A grayscale image of shape (h, w).
        device (torch.device, optional): The device to create tensors on.
                If None, the device of the input tensor is used.

    Returns:
        Complex: A complex containing the weighted vertices, weighted edges, and weighted triangles.
    """

    device = img_arr.device if device is None else device
    img_arr = img_arr.float().to(device)
    h, w = img_arr.shape

    # Create a mask of the nonzero pixels
    img_mask = img_arr != 0 

    # Indices of nonzero pixels (vertices)
    nonzero_vertices = torch.nonzero(img_mask, as_tuple=True)

    # Enumerate the nonzero vertices in the index array with all other values set to 0
    vertex_numbers = torch.zeros_like(img_arr, dtype=torch.int64, device=device)
    vertex_numbers[nonzero_vertices] = torch.arange(
        nonzero_vertices[0].size(0), dtype=torch.int64, device=device
    )

    # Construct the vertex coords and weights
    vertex_coords = torch.stack([
        nonzero_vertices[1] - (w - 1) / 2.0,
        (h - 1) / 2.0 - nonzero_vertices[0]
    ], dim=1)
    vertex_weights = img_arr[nonzero_vertices]
    vertices = (vertex_coords, vertex_weights)

    ### Horizontal Edges
    # Remove the first and last columns of img_mask and check where the resulting arrays are both nonzero
    horizontal_edge_mask = img_mask[:, :-1] & img_mask[:, 1:] 
    horizontal_edge_indices = torch.nonzero(horizontal_edge_mask, as_tuple=True)

    # Get the vertex numbers of the endpoints of each horizontal edge
    horizontal_edge_vertices = torch.stack([
        vertex_numbers[horizontal_edge_indices],
        vertex_numbers[:, 1:][horizontal_edge_indices]
    ], dim=1)
    horizontal_edge_weights = vertex_weights[horizontal_edge_vertices].amax(dim=1)

    ### Vertical Edges
    # Remove the first and last rows of img_mask and check where the resulting arrays are both nonzero
    vertical_edge_mask = img_mask[:-1, :] & img_mask[1:, :]
    vertical_edge_indices = torch.nonzero(vertical_edge_mask, as_tuple=True)

    # Get the vertex numbers of the endpoints of each vertical edge
    vertical_edge_vertices = torch.stack([
        vertex_numbers[vertical_edge_indices],
        vertex_numbers[1:, :][vertical_edge_indices]
    ], dim=1)
    vertical_edge_weights = vertex_weights[vertical_edge_vertices].amax(dim=1)

    ### Diagonal Edges
    diagonal_edge_mask = img_mask[:-1, :-1] & img_mask[1:, 1:]
    diagonal_edge_indices = torch.nonzero(diagonal_edge_mask, as_tuple=True)
    diagonal_edge_vertices = torch.stack([
        vertex_numbers[diagonal_edge_indices],
        vertex_numbers[1:, 1:][diagonal_edge_indices]
    ], dim=1)
    diagonal_edge_weights = vertex_weights[diagonal_edge_vertices].amax(dim=1)

    # Concatenate the horizontal, vertical, and diagonal edges
    edge_vertices = torch.cat([
        horizontal_edge_vertices,
        vertical_edge_vertices,
        diagonal_edge_vertices
    ], dim=0)
    edge_weights = torch.cat([
        horizontal_edge_weights,
        vertical_edge_weights,
        diagonal_edge_weights
    ], dim=0)
    edges = (edge_vertices, edge_weights)

    ### Upper Triangles
    upper_triangle_mask = img_mask[:-1, :-1] & img_mask[:-1, 1:] & img_mask[1:, 1:]
    upper_triangle_indices = torch.nonzero(upper_triangle_mask, as_tuple=True)
    upper_triangle_vertices = torch.stack([
        vertex_numbers[upper_triangle_indices],
        vertex_numbers[:, 1:][upper_triangle_indices],
        vertex_numbers[1:, 1:][upper_triangle_indices]
    ], dim=1)
    upper_triangle_weights = vertex_weights[upper_triangle_vertices].amax(dim=1)

    ### Lower Triangles
    lower_triangle_mask = img_mask[:-1, :-1] & img_mask[1:, :-1] & img_mask[1:, 1:]
    lower_triangle_indices = torch.nonzero(lower_triangle_mask, as_tuple=True)
    lower_triangle_vertices = torch.stack([
        vertex_numbers[lower_triangle_indices],
        vertex_numbers[1:, :][lower_triangle_indices],
        vertex_numbers[1:, 1:][lower_triangle_indices]
    ], dim=1)
    lower_triangle_weights = vertex_weights[lower_triangle_vertices].amax(dim=1)

    ### Concatenate the upper and lower triangles
    triangle_vertices = torch.cat([
        upper_triangle_vertices,
        lower_triangle_vertices
    ], dim=0)
    triangle_weights = torch.cat([
        upper_triangle_weights,
        lower_triangle_weights
    ], dim=0)
    triangles = (triangle_vertices, triangle_weights)

    return Complex(vertices, edges, triangles, device=device)


def weighted_cubical(
    img_arr: torch.Tensor, device: Optional[torch.device] = None
) -> Complex:
    """
    Creates the weighted cubical complex of an image array.
    Discards edges and squares that have a vertex with zero weight.

    The vertices are a (h*w, 2) tensor with recentered pixel coordinates.
    The vertex weights are a (h*w,) tensor containing the pixel intensities.
    The edges are a (num_valid_edges, 2) tensor of vertex indices.
    The edge weights are a (num_valid_edges,) tensor with the maximum weight on the edge.
    The squares are a (num_valid_squares, 4) tensor of vertex indices.
    The square weights are a (num_valid_squares,) tensor with the maximum weight on
    the square.

    Args:
        img_arr (torch.Tensor): A grayscale image of shape (h, w).
        device (torch.device, optional): The device to create tensors on.
                If None, the device of the input tensor is used.

    Returns:
        Complex: A complex containing the weighted vertices, weighted edges, and weighted squares.
    """

    device = img_arr.device if device is None else device
    img_arr = img_arr.float().to(device)
    h, w = img_arr.shape

    # Create a mask of the nonzero pixels
    img_mask = img_arr != 0

    # Indices of nonzero pixels (vertices)
    nonzero_vertices = torch.nonzero(img_mask, as_tuple=True)

    # Create an array enumerating the nonzero vertices with all other values 0
    vertex_numbers = torch.zeros_like(img_arr, dtype=torch.int64, device=device)
    vertex_numbers[nonzero_vertices] = torch.arange(
        nonzero_vertices[0].size(0), dtype=torch.int64, device=device
    )

    # Construct the vertex coords and weights
    vertex_coords = torch.stack([
        nonzero_vertices[1] - (w - 1) / 2.0,
        (h - 1) / 2.0 - nonzero_vertices[0]
    ], dim=1)
    vertex_weights = img_arr[nonzero_vertices]
    vertices = (vertex_coords, vertex_weights)

    ### Horizontal Edges
    # Remove the first and last columns of img_mask and check where the resulting arrays are both nonzero
    horizontal_edge_mask = img_mask[:, :-1] & img_mask[:, 1:]
    horizontal_edge_indices = torch.nonzero(horizontal_edge_mask, as_tuple=True)

    # Get the vertex numbers of the endpoints of each horizontal edge
    horizontal_edge_vertices = torch.stack([
        vertex_numbers[horizontal_edge_indices],
        vertex_numbers[:, 1:][horizontal_edge_indices]
    ], dim=1)
    horizontal_edge_weights = vertex_weights[horizontal_edge_vertices].amax(dim=1)

    ### Vertical Edges
    # Remove the first and last rows of img_mask and check where the resulting arrays are both nonzero
    vertical_edge_mask = img_mask[:-1, :] & img_mask[1:, :]
    vertical_edge_indices = torch.nonzero(vertical_edge_mask, as_tuple=True)

    # Get the vertex numbers of the endpoints of each vertical edge
    vertical_edge_vertices = torch.stack([
        vertex_numbers[vertical_edge_indices],
        vertex_numbers[1:, :][vertical_edge_indices]
    ], dim=1)
    vertical_edge_weights = vertex_weights[vertical_edge_vertices].amax(dim=1)

    # Concatenate the horizontal and vertical edges
    edge_vertices = torch.cat([
        horizontal_edge_vertices,
        vertical_edge_vertices
    ], dim=0)
    edge_weights = torch.cat([
        horizontal_edge_weights,
        vertical_edge_weights
    ], dim=0)
    edges = (edge_vertices, edge_weights)

    ###Squares
    square_mask = horizontal_edge_mask[:-1, :] & horizontal_edge_mask[1:, :]
    square_indices = torch.nonzero(square_mask, as_tuple=True)
    square_vertices = torch.stack([
        vertex_numbers[square_indices],
        vertex_numbers[1:, :][square_indices],
        vertex_numbers[:, 1:][square_indices],
        vertex_numbers[1:, 1:][square_indices]
    ], dim=1)
    square_weights = vertex_weights[square_vertices].amax(dim=1)
    squares = (square_vertices, square_weights)

    return Complex(vertices, edges, squares, n_type="cubical", device=device)