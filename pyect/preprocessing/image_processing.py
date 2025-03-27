from typing import Optional, Tuple
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
    # The resulting tensor will have shape (1, H, W)
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
    h, w = img_arr.shape
    device = img_arr.device if device is None else device
    img_arr = img_arr.to(device)

    # Compute pixel coordinates recentered around (0,0)
    i = torch.arange(h, dtype=torch.float32, device=device)
    j = torch.arange(w, dtype=torch.float32, device=device)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    # x axis: shift columns; y axis: invert rows so that higher rows correspond to lower y values
    x = j_grid - (w - 1) / 2.0
    y = (h - 1) / 2.0 - i_grid
    vertex_coords = torch.stack([x, y], dim=-1).reshape(-1, 2)

    # Flatten the image to get a weight per vertex (ensure float type)
    flat_img = img_arr.reshape(-1).float()
    vertices = (vertex_coords, flat_img)

    # Compute the Freudenthal triangulation (edges and triangles)
    edges, triangles = _full_freudenthal((h, w), device)

    # ----- Process Edges -----
    # For each edge (pair of vertex indices), get the vertex weights.
    edge_vert_weights = flat_img[edges.to(device)]  # shape: (num_edges, 2)
    # Compute the max and min weights along each edge.
    max_edge_weights = edge_vert_weights.amax(dim=1)
    min_edge_weights = edge_vert_weights.amin(dim=1)
    # Keep edges only if both vertex weights are non-zero (min != 0)
    positive_edge_mask = min_edge_weights != 0
    positive_edges = edges[positive_edge_mask]
    positive_edge_weights = max_edge_weights[positive_edge_mask]
    weighted_edges = (positive_edges, positive_edge_weights)

    # ----- Process Triangles -----
    # For each triangle (triplet of vertex indices), get the vertex weights.
    tri_vert_weights = flat_img[triangles]  # shape: (num_triangles, 3)
    max_tri_weights = tri_vert_weights.amax(dim=1)
    min_tri_weights = tri_vert_weights.amin(dim=1)
    # Keep triangles only if all vertex weights are non-zero.
    positive_tri_mask = min_tri_weights != 0
    positive_triangles = triangles[positive_tri_mask]
    positive_tri_weights = max_tri_weights[positive_tri_mask]
    weighted_triangles = (positive_triangles, positive_tri_weights)

    return Complex(vertices, weighted_edges, weighted_triangles, device=device)


def weighted_cubical(img_arr: torch.Tensor, device: Optional[torch.device]) -> Complex:
    """
    Creates the weighted cubical complex of an image array.
    Discards edges and squares that have a vertex with a zero weight.

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
    h, w = img_arr.shape
    device = img_arr.device if device is None else device
    img_arr = img_arr.to(device)

    # Compute recentered vertex coordinates.
    i = torch.arange(h, dtype=torch.float32, device=device)
    j = torch.arange(w, dtype=torch.float32, device=device)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    x = j_grid - (w - 1) / 2.0
    y = (h - 1) / 2.0 - i_grid
    vertex_coords = torch.stack([x, y], dim=-1).reshape(-1, 2)

    # Flatten image to get vertex weights.
    flat_img = img_arr.reshape(-1).float()
    vertices = (vertex_coords, flat_img)

    # Compute cubical complex: edges and squares.
    edges = _cubical_edges(h, w, device)
    squares = _cubical_squares(h, w, device)

    # ----- Process Edges -----
    # For each edge, get the vertex weights.
    edge_vert_weights = flat_img[edges.to(device)]  # shape: (num_edges, 2)
    max_edge_weights = edge_vert_weights.amax(dim=1)
    min_edge_weights = edge_vert_weights.amin(dim=1)
    # Keep only edges with both vertex weights non-zero.
    positive_edge_mask = min_edge_weights != 0
    positive_edges = edges[positive_edge_mask]
    positive_edge_weights = max_edge_weights[positive_edge_mask]
    weighted_edges = (positive_edges, positive_edge_weights)

    # ----- Process Squares -----
    # For each square, get the vertex weights.
    square_vert_weights = flat_img[squares]  # shape: (num_squares, 4)
    max_square_weights = square_vert_weights.amax(dim=1)
    min_square_weights = square_vert_weights.amin(dim=1)
    # Keep only squares with all vertex weights non-zero.
    positive_square_mask = min_square_weights != 0
    positive_squares = squares[positive_square_mask]
    positive_square_weights = max_square_weights[positive_square_mask]
    weighted_squares = (positive_squares, positive_square_weights)

    return Complex(vertices, weighted_edges, weighted_squares, device=device)


def get_shifted_vertices(vertices, base_coords, shift):
    """
    Helper to shift base coordinates by a given shift and gather the corresponding vertices.

    Args:
        vertices (Tensor): 2D grid of vertex indices of shape (h, w).
        base_coords (Tensor): Tensor of base coordinates with shape (..., 2).
        shift (list or tuple): The [row, col] shift to apply.

    Returns:
        Tensor: Gathered vertices after applying the shift.
    """
    # Convert the shift to a tensor of shape (1, 1, 2)
    shift_tensor = torch.tensor(shift, dtype=torch.long).view(1, 1, 2)
    # Apply the shift
    shifted_coords = base_coords + shift_tensor
    return vertices[shifted_coords[..., 0], shifted_coords[..., 1]]


def _freudenthal_edges(vertices, h, w):
    """
    Constructs horizontal, vertical, and diagonal edges.
    """
    # Horizontal edges
    i = torch.arange(h)
    j = torch.arange(w - 1)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    coords = torch.stack([i_grid, j_grid], dim=-1)  # shape (h, w-1, 2)
    left_vertices = vertices[coords[..., 0], coords[..., 1]]
    right_vertices = get_shifted_vertices(vertices, coords, [0, 1])
    h_edges = torch.stack([left_vertices, right_vertices], dim=-1).reshape(-1, 2)

    # Vertical edges
    i = torch.arange(h - 1)
    j = torch.arange(w)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    coords = torch.stack([i_grid, j_grid], dim=-1)  # shape (h-1, w, 2)
    top_vertices = vertices[coords[..., 0], coords[..., 1]]
    bot_vertices = get_shifted_vertices(vertices, coords, [1, 0])
    v_edges = torch.stack([top_vertices, bot_vertices], dim=-1).reshape(-1, 2)

    # Diagonal edges
    i = torch.arange(h - 1)
    j = torch.arange(w - 1)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    coords = torch.stack([i_grid, j_grid], dim=-1)  # shape (h-1, w-1, 2)
    tl_vertices = vertices[coords[..., 0], coords[..., 1]]
    br_vertices = get_shifted_vertices(vertices, coords, [1, 1])
    d_edges = torch.stack([tl_vertices, br_vertices], dim=-1).reshape(-1, 2)

    return torch.cat([h_edges, v_edges, d_edges], dim=0)


def _freudenthal_triangles(vertices, h, w):
    """
    Constructs triangles by splitting each grid cell into two triangles.
    """
    i = torch.arange(h - 1)
    j = torch.arange(w - 1)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    coords = torch.stack([i_grid, j_grid], dim=-1)  # shape (h-1, w-1, 2)

    tl_vertices = vertices[coords[..., 0], coords[..., 1]]
    br_vertices = get_shifted_vertices(vertices, coords, [1, 1])

    # Upper triangle: top-left, top-right, bottom-right
    tr_vertices = get_shifted_vertices(vertices, coords, [0, 1])
    u_triangles = torch.stack([tl_vertices, tr_vertices, br_vertices], dim=-1).reshape(
        -1, 3
    )

    # Lower triangle: top-left, bottom-left, bottom-right
    bl_vertices = get_shifted_vertices(vertices, coords, [1, 0])
    l_triangles = torch.stack([tl_vertices, bl_vertices, br_vertices], dim=-1).reshape(
        -1, 3
    )

    return torch.cat([u_triangles, l_triangles], dim=0)


def _full_freudenthal(img_shape: Tuple[int, int], device: torch.device):
    """
    Generate edges and triangles for Freudenthal triangulation based on given image dimensions.

    Args:
        img_shape (tuple): A tuple (h, w) representing the grid dimensions.

    Returns:
        edges, triangles (tuple): A tuple containing:
            - edges: An (N, 2) tensor of edges.
            - triangles: An (M, 3) tensor of triangles.
    """
    h, w = img_shape
    # Create a grid of vertex indices on the device
    vertices = torch.arange(h * w, device=device).reshape(h, w)

    edges = _freudenthal_edges(vertices, h, w)
    triangles = _freudenthal_triangles(vertices, h, w)

    return edges, triangles


def _cubical_edges(h: int, w: int, device: Optional[torch.device] = None):
    """
    Constructs horizontal and vertical edges for a grid of size (h, w).

    Args:
        h (int): Number of rows.
        w (int): Number of columns.
        device (torch.device, optional): The device to create tensors on.

    Returns:
        torch.Tensor: A tensor of shape (num_edges, 2) where each row contains two vertex indices forming an edge.
    """
    if device is None:
        device = torch.device("cpu")
    grid = torch.arange(h * w, device=device).reshape(h, w)

    # Horizontal edges: from grid[i,j] to grid[i, j+1] for j in 0..w-2.
    i = torch.arange(h, device=device)
    j = torch.arange(w - 1, device=device)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    left = grid[i_grid, j_grid]
    right = grid[i_grid, j_grid + 1]
    horizontal_edges = torch.stack([left, right], dim=-1).reshape(-1, 2)

    # Vertical edges: from grid[i,j] to grid[i+1, j] for i in 0..h-2.
    i = torch.arange(h - 1, device=device)
    j = torch.arange(w, device=device)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    top = grid[i_grid, j_grid]
    bottom = grid[i_grid + 1, j_grid]
    vertical_edges = torch.stack([top, bottom], dim=-1).reshape(-1, 2)

    return torch.cat([horizontal_edges, vertical_edges], dim=0)


def _cubical_squares(h: int, w: int, device: Optional[torch.device] = None):
    """
    Constructs squares (2-cells) for a grid of size (h, w).

    Args:
        h (int): Number of rows.
        w (int): Number of columns.
        device (torch.device, optional): The device to create tensors on.

    Returns:
        torch.Tensor: A tensor of shape (num_squares, 4) where each row contains the vertex indices
                      of a square ordered as (top-left, top-right, bottom-right, bottom-left).
    """
    if device is None:
        device = torch.device("cpu")
    grid = torch.arange(h * w, device=device).reshape(h, w)

    # Each square is defined on a cell: for i in 0..h-2 and j in 0..w-2.
    i = torch.arange(h - 1, device=device)
    j = torch.arange(w - 1, device=device)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")
    tl = grid[i_grid, j_grid]
    tr = grid[i_grid, j_grid + 1]
    br = grid[i_grid + 1, j_grid + 1]
    bl = grid[i_grid + 1, j_grid]
    squares = torch.stack([tl, tr, br, bl], dim=-1).reshape(-1, 4)
    return squares
