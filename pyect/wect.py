import torch

# Automatically select the best available device
device = torch.device("cuda" if torch.cuda.is_available() else
                      "cpu")

def compute_wect(vertices, higher_simplices, directions, num_heights):
    """
    Calculates a discretization of the differentiated WECT of a simplicial complex embedded in (n+1)-dimensional space.

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

    d = directions.size(dim=0)
    v_coords, v_weights = vertices
    expanded_v_weights = v_weights.unsqueeze(0).expand(d, -1) # Expand to shape (d, k_0)

    # Initialize the differentiated WECT
    diff_wect = torch.zeros((d, num_heights), dtype=v_weights.dtype, device=v_weights.device)

    # Compute the height index of each vertex
    v_indices = vertex_indices(v_coords, directions, num_heights)

    # Add the contribution of the vertices to the differentiated WECT
    diff_wect.scatter_add_(1, v_indices.t(), expanded_v_weights)

    for i, simplices in enumerate(higher_simplices):
        simp_verts, simp_weights = simplices
        expanded_simp_weights = (-1)**(i+1) * simp_weights.unsqueeze(0).expand(d, -1)

        simp_indices = v_indices[simp_verts.to(v_indices.device)]
        max_simp_indices = torch.amax(simp_indices, dim=1)

        diff_wect.scatter_add_(1, max_simp_indices.t(), expanded_simp_weights)

    return torch.cumsum(diff_wect, dim=1)

def vertex_indices(vertex_coords, directions, num_heights):
    """
    Calculates the height values of each vertex and converts them to an index in range(num_heights).
    
    Assumes that directions are unit vectors.

    Args:
        vertex_coords (torch.Tensor): A tensor of shape (k_0, n+1) with rows representing the coordinates of the vertices.
        directions (torch.Tensor): A tensor of shape (d, n+1) with rows representing the sampled direction vectors.
        num_heights (int): The number of height values to sample.

    Returns:
        torch.Tensor: A tensor of shape (k_0, d) with the height indices of each vertex in each direction.
    """
    # Compute the maximum possible height value
    v_norms = torch.linalg.norm(vertex_coords, dim=1)
    max_height = v_norms.max()

    # Compute dot products (heights in each direction)
    v_heights = torch.matmul(vertex_coords, directions.t())

    # Scale and convert to indices in the range [0, num_heights - 1]
    v_indices = torch.ceil(((num_heights - 1) * (max_height + v_heights)) / (2.0 * max_height)).to(torch.int64)

    return v_indices