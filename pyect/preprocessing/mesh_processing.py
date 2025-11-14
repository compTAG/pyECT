"""For converting mesh files into our Complex class"""

import torch
import trimesh
from pyect import Complex

def mesh_to_complex(mesh_path: str, device: torch.device, centering=False) -> Complex:
    """
    Converts a mesh file (OBJ, STL, etc.) to the Complex class.

    Args:
        mesh_path (string): The file path of the mesh file.
        device (torch.device): The device the complex will be stored on.
        centering (bool): If True, the mesh will be recentered about the origin.

    Returns:
        Complex: The simplicial complex of the mesh file.
    """
    
    mesh = trimesh.load_mesh(mesh_path)

    vertex_coords = mesh.vertices
    if centering:
        vertex_coords = vertex_coords - mesh.centroid
    vertex_coords = torch.tensor(vertex_coords).to(device=device, dtype=torch.float32)

    vertex_weights = torch.ones(vertex_coords.size(dim=0), device=device, dtype=torch.float32)
    vertices = (vertex_coords, vertex_weights)

    edge_indices = mesh.edges_unique
    edge_indices = torch.tensor(edge_indices).to(device=device, dtype=torch.int64)
    edge_weights = torch.ones(edge_indices.size(dim=0), device=device, dtype=torch.float32)
    edges = (edge_indices, edge_weights)

    triangle_indices = mesh.faces
    triangle_indices = torch.tensor(triangle_indices).to(device=device, dtype=torch.int64)
    triangle_weights = torch.ones(triangle_indices.size(dim=0), device=device, dtype=torch.float32)
    triangles = (triangle_indices, triangle_weights)

    return Complex(vertices, edges, triangles, device=device)