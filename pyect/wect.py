import torch

# Automatically select the best available device
device = torch.device("cuda" if torch.cuda.is_available() else
                      "cpu")

def scatter_wect(vertices, higher_simplices, directions, num_heights):
    """
    Calculates a discretization of the WECT of a simplicial complex embedded in (n+1)-dimensional space.

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
        wect (torch.Tensor): A tensor of shape (d, num_heights) containing a discretization of the WECT.
    """

    diff_wect = scatter_diff_wect(vertices, higher_simplices, directions, num_heights)
    return torch.cumsum(diff_wect, dim=1)

def scatter_diff_wect(vertices, higher_simplices, directions, num_heights):
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

    return diff_wect


def compute_wect(vertices, higher_simplices, directions, num_heights):
    """
    Calculates a discretization of the WECT of a simplicial complex embedded in (n+1)-dimensional space.

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
        wect (torch.Tensor): A tensor of shape (d, num_heights) containing a discretization of the WECT.
    """

    d_wect = compute_differentiated_wect(vertices, higher_simplices, directions, num_heights)
    d_wect = d_wect.to_dense()
    return torch.cumsum(d_wect, dim=1)

def compute_differentiated_wect(vertices, higher_simplices, directions, num_heights):
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
        d_wect (torch.Tensor): A tensor of shape (d, num_heights) containing a discretization of the differentiated WECT.
    """

    v_coords, v_weights = vertices

    v_indices = vertex_indices(v_coords, directions, num_heights)
    d_wect = vertex_sum(v_indices, v_weights, num_heights)

    for i, simplices in enumerate(higher_simplices):
        simp_sum = simplex_sum(v_indices, simplices, num_heights)
        d_wect = d_wect + (-1)**(i+1)*simp_sum

    return d_wect

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

def sparse_one_hot_2d(labels, num_classes):
    """
    Create a 3D sparse one-hot encoding of a 2D label array.

    Args:
        labels (torch.Tensor): A 2D tensor of shape (n, m) with integer labels in [0, num_classes - 1].
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor (sparse COO): A sparse tensor of shape (n, m, num_classes).
    """
    # labels.shape is (n, m)
    n, m = labels.shape

    # row_indices: [0, 0, ..., 1, 1, ..., 2, 2, ...]
    row_indices = torch.arange(n).unsqueeze(1).expand(n, m).reshape(-1)

    # col_indices: [0, 1, ..., 0, 1, ...] across rows
    col_indices = torch.arange(m).unsqueeze(0).expand(n, m).reshape(-1)

    # Flatten labels to get class indices for each (row, col).
    class_indices = labels.reshape(-1)  # shape [n*m]

    # Stack them into shape (3, n*m):
    indices = torch.stack([row_indices, col_indices, class_indices], dim=0)

    # Values are all 1's for a one-hot encoding
    values = torch.ones(indices.size(1), dtype=torch.float32)

    # The one-hot encoding has shape (n, m, num_classes)
    one_hot_shape = (n, m, num_classes)

    # Build the sparse COO tensor
    sparse_one_hot = torch.sparse_coo_tensor(indices, values, one_hot_shape)

    return sparse_one_hot

def vertex_sum(v_indices, v_weights, num_heights):
    """
    Calculates the contribution of the vertices to the WECT.

    Args:
        v_indices (torch.Tensor): A tensor of shape (k_0, d) with rows the height indices of each vertex in each direction.
        v_weights (torch.Tensor): A tensor of shape (k_0) containing the vertex weights.
        num_heights (Int): The number of height values to sample.

    Returns:
        v_sum (torch.Tensor): A tensor of shape (d, num_heights) representing the contribution of the vertices to the differentiated WECT.
    """

    v_weights = v_weights.reshape(-1, 1, 1)

    v_graphs = sparse_one_hot_2d(v_indices, num_heights)
    weighted_v_graphs = v_weights * v_graphs
    v_sum = torch.sparse.sum(weighted_v_graphs, dim=0)

    return v_sum

def simplex_sum(v_indices, simplices, num_heights):
    """
    Calculates the contribution of the i-simplices to the WECT.

    Args:
        v_indices (torch.Tensor): A tensor of shape (k_0, d) with rows the height indices of each vertex in each direction.
        simplices (torch.Tensor, torch.Tensor): A tuple containing two tensors:
            simplices[0]: A tensor of shape (k_i, i+1) with rows the vertices of each i-simplex.
            simplices[1]: A tensor of shape (k_i) containing the weights of each i-simplex.
        num_heights (Int): The number of height values to sample.

    Returns:
        simp_sum (torch.Tensor): A tensor of shape (d, num_heights) representing the contribution of the i-simplices to the differentiated WECT.
    """

    simp_vertices, simp_weights = simplices
    simp_weights = simp_weights.reshape(-1, 1, 1)

    simp_indices = v_indices[simp_vertices.to(v_indices.device)]
    max_simp_indices = torch.amax(simp_indices, dim=1)
    simp_graphs = sparse_one_hot_2d(max_simp_indices, num_heights)
    weighted_simp_graphs = simp_weights * simp_graphs
    simp_sum = torch.sparse.sum(weighted_simp_graphs, dim=0)

    return simp_sum

