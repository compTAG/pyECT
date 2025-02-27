import torch

# Automatically select the best available device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                      "cpu")

def vertex_indices(vertex_coords, directions, num_heights):
    """
    Calculates the height values of each vertex and 

    Args:
        labels (torch.Tensor): A 2D tensor of shape (n, m) with integer labels in [0, num_classes - 1].
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor (sparse COO): A sparse tensor of shape (n, m, num_classes).
    """

    # Compute the maximum possible height value
    v_norms = torch.norm(vertex_coords, dim=1)
    max_height = v_norms.max()

    # Compute dot products
    v_heights = torch.matmul(vertex_coords, directions.t())

    # Scale and convert to indices
    indices = torch.ceil(((num_heights - 1) * (max_height + v_heights)) / (2.0 * max_height))

    return indices

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

    # Build row and column indices for every element in the 2D labels tensor.
    # row_indices: [0, 0, ..., 1, 1, ..., 2, 2, ...]
    row_indices = torch.arange(n).unsqueeze(1).expand(n, m).reshape(-1)

    # col_indices: [0, 1, ..., 0, 1, ...] across rows
    col_indices = torch.arange(m).unsqueeze(0).expand(n, m).reshape(-1)

    # Flatten labels to get class indices for each (row, col).
    class_indices = labels.reshape(-1)  # shape [n*m]

    # Stack them into shape (3, n*m):
    #   1st row -> row index
    #   2nd row -> column index
    #   3rd row -> class index
    indices = torch.stack([row_indices, col_indices, class_indices], dim=0)

    # Values are all 1's for a one-hot encoding
    values = torch.ones(indices.size(1), dtype=torch.float32)

    # The one-hot encoding has shape (n, m, num_classes)
    one_hot_shape = (n, m, num_classes)

    # Build the sparse COO tensor
    sparse_one_hot = torch.sparse_coo_tensor(indices, values, one_hot_shape)

    return sparse_one_hot

def vertex_sum(v_indices, v_weights, num_heights):

    v_weights = v_weights.reshape(-1, 1, 1)

    v_graphs = sparse_one_hot_2d(v_indices, num_heights)
    weighted_v_graphs = v_weights * v_graphs
    v_sum = torch.sparse.sum(weighted_v_graphs, dim=0)

    return v_sum

def simplex_sum(v_indices, simplices, num_heights):

    simp_vertices, simp_weights = simplices
    simp_weights = simp_weights.reshape(-1, 1, 1)

    simp_indices = v_indices[simp_vertices]
    max_simp_indices = torch.amax(simp_indices, dim=1)
    simp_graphs = sparse_one_hot_2d(max_simp_indices, num_heights)
    weighted_simp_graphs = simp_weights * simp_graphs
    simp_sum = torch.sparse.sum(weighted_simp_graphs, dim=0)

    return simp_sum

def compute_differentiated_wect(vertices, higher_simplices, directions, num_heights):

    v_coords, v_weights = vertices

    v_indices = vertex_indices(v_coords, directions, num_heights)
    d_wect = vertex_sum(v_indices, v_weights, num_heights)

    for i, simplices in enumerate(higher_simplices):
        simp_sum = simplex_sum(v_indices, simplices, num_heights)
        d_wect = d_wect + (-1)**(i+1)*simp_sum

    return d_wect

def compute_wect(vertices, higher_simplices, directions, num_heights):
    d_wect = compute_differentiated_wect(vertices, higher_simplices, directions, num_heights)
    return torch.cumsum(d_wect, 1)
