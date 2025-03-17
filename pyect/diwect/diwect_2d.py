import torch
from .utils import sgn_star, signed_angle, simplex_normals_2d, vertex_balls

def compute_simplex_measures(v_coords, v_balls, simplices):

    simp_verts, simp_weights = simplices
    simp_weights = simp_weights.unsqueeze(-1)
    v_centers, v_radii = v_balls

    simp_normals = simplex_normals_2d(v_coords, simp_verts)

    simp_alphas = signed_angle(simp_normals, v_centers[simp_verts])
    simp_alphas = simp_alphas.unsqueeze(dim=-1)

    simp_betas = signed_angle(v_centers[simp_verts], simp_normals.roll(1, dims=1))
    simp_betas = simp_betas.unsqueeze(dim=-1)

    simp_vert_measures = (
        sgn_star(simp_alphas) * torch.min(v_radii[simp_verts], simp_alphas.abs()) +
        sgn_star(simp_betas) * torch.min(v_radii[simp_verts], simp_betas.abs()) +
        .5 * v_radii[simp_verts] * (1 - sgn_star(simp_alphas)) * (1 - sgn_star(simp_betas))
    )
    simp_measures = torch.sum(simp_vert_measures, dim=1)
    weighted_simp_measures = simp_weights * simp_measures

    return torch.sum(weighted_simp_measures, dim=0)

def compute_diwect_2d(vertices, edges, triangles, num_heights):

    v_coords, v_weights = vertices
    v_weights = v_weights.unsqueeze(1)

    v_balls = vertex_balls(v_coords, num_heights)
    v_radii = v_balls[1]

    diwect = (
        torch.sum(2 * v_weights * v_radii, dim=0) + # contribution of the vertices
        compute_simplex_measures(v_coords, v_balls, edges) + # contribution of the edges 
        compute_simplex_measures(v_coords, v_balls, triangles) # contribution of the triangles
    )

    return diwect