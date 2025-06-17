import torch
import miniball
from pyect import Complex
import numpy as np



def center_complex(Complex):

    '''
    Compute the smallest enclosing ball around the vertices
    of a simplicial complex
    '''

    (coords, coord_weights), (edges, edge_weights) = Complex

    # Compute smallest enclosing ball
    coords_np = coords.detach().cpu().numpy()
    center, _ = miniball.get_bounding_ball(coords_np)
    center = torch.tensor(center, dtype=coords.dtype, device=coords.device)
    
    centered_coords = coords - center

    return type(Complex)((centered_coords, coord_weights), (edges, edge_weights))

'''
def main():

    ## Test with square complex in quadrant 1, with one vertice at origin and side lengths 1.

    coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    coord_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    edges = torch.tensor([[0, 1], [1, 3], [3, 2], [2, 0]])
    edge_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

    complex = Complex(tuple([coordinates, coord_weights]), tuple([edges, edge_weights]))

    directions = torch.tensor([[1.0,0.0], [0.0,1.0], [-1.0,0.0], [0.0,-1.0]])
    num_heights = [4.0]

    new_complex = center(complex)
'''

def test_center_complex(verbose=True):
    # Define a simple square complex
    coords = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    coord_weights = torch.ones(4)
    edges = torch.tensor([[0, 1], [1, 3], [3, 2], [2, 0]])
    edge_weights = torch.ones(4)

    complex = Complex((coords, coord_weights), (edges, edge_weights))

    # Get the original smallest enclosing ball center
    original_center, _ = miniball.get_bounding_ball(coords.numpy())

    # Apply centering
    centered = center_complex(complex)
    new_coords = centered[0][0]
    new_center, _ = miniball.get_bounding_ball(new_coords.numpy())

    # Check closeness to zero
    assert np.allclose(new_center, 0, atol=1e-6), f"New center is not at origin: {new_center}"
    assert torch.equal(centered[1][0], complex[1][0]), "Edges changed"
    assert torch.equal(centered[1][1], complex[1][1]), "Edge weights changed"

    if verbose:
        print("Original center of bounding ball:", original_center)
        print("New center after centering:", new_center)
        print("✅ Passed: Coordinates are centered. Topology is preserved.")

def main():
    test_center_complex()

if __name__ == "__main__":
    main()
