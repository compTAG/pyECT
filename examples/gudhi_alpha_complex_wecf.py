"""Example WECF computation from a Gudhi alpha complex."""

import torch

from pyect import compute_wecfs_general
from pyect.integrations.gudhi import alpha_complex_to_filtration_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points = [
        [1.0, 1.0],
        [7.0, 0.0],
        [4.0, 6.0],
        [9.0, 6.0],
        [0.0, 14.0],
        [2.0, 19.0],
        [9.0, 17.0],
    ]

    point_weights = [0.0, 0.2, 0.1, 0.3, 0.0, 0.1, 0.2]

    filtration_data, simplex_tree = alpha_complex_to_filtration_data(
        points,
        point_weights=point_weights,
        device=device,
    )

    wecf = compute_wecfs_general(filtration_data, num_vals=200)

    print(simplex_tree.num_simplices())
    print(wecf.shape)


if __name__ == "__main__":
    main()
