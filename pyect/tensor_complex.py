"""Tools for working with simplicial complexes.

The Complex class is a collection of simplices, each of which is represented by a
tensor of coordinates and a tensor of weights.
"""

from typing import Tuple, Optional

import torch
import warnings
import numpy.typing as npt

from .dtypes import COORDS_DTYPE, INDICES_DTYPE, WEIGHTS_DTYPE


class Complex:
    """A simplicial complex of arbitrary dimension.

    The representation is as a collection of simplices (or cubical cells) using tensors.
    """

    def __init__(
        self,
        *args: Tuple[torch.Tensor, torch.Tensor],
        vertex_dtype: torch.dtype = COORDS_DTYPE,
        index_dtype: torch.dtype = INDICES_DTYPE,
        weights_dtype: torch.dtype = WEIGHTS_DTYPE,
        device: Optional[torch.device] = None,
        n_type: str = "simplicial",
    ) -> None:
        """Initializes a complex.

        All tensors are cast to the given types.

        Args:
            *args: A variable number of tuples, each containing the simplices of a given
                dimension. Each tuple should contain two tensors.
                The first tensor contains the coordinates of the simplices
                The second tensor contains the weights of the simplices.

                The first tuple should contain the vertices of the complex, and
                therefore must be a tensor of shape [num_vertices, d].

                Any following tuples should contain indices into the vertices tensor,
                and therefore must be a tensor of shape [num_simplices, k], where k+1 is the
                dimension of the simplex.

            vertex_dtype: The data type to use for the vertex coordinates.
            index_dtype: The data type to use for the simplex indices.
            weights_dtype: The data type to use for the simplex weights.
            device: The device to use for the tensors.
            n_type: The type of complex. Currently only "simplicial" and "cubical"
                are supported.
        """
        # Verify the dimensions of the simplices, and raise a UserError if
        # there is a mismatch.
        self._validate_dimensions(*args, n_type=n_type)

        # Call .to on each tensor to cast to the given type and device.
        types = [vertex_dtype] + [index_dtype] * (len(args) - 1)
        self.dimensions = tuple(
            (
                (
                    coords.to(dtype=types[dim], device=device),
                    weights.to(dtype=weights_dtype, device=device),
                )
            )
            for dim, (coords, weights) in enumerate(args)
        )
        self.n_type = n_type

    @staticmethod
    def from_numpy(
        *args: Tuple[npt.NDArray, npt.NDArray],
        vertex_dtype: torch.dtype = COORDS_DTYPE,
        index_dtype: torch.dtype = INDICES_DTYPE,
        weights_dtype: torch.dtype = WEIGHTS_DTYPE,
        device: Optional[torch.device] = None,
        n_type: str = "simplicial",
    ) -> "Complex":
        """Initializes a simplicial complex from numpy arrays.

        Args:
            *args: A variable number of tuples, each containing the simplices of a given
                dimension. Each tuple should contain two numpy arrays.
                The first array contains the coordinates of the simplices
                The second array contains the weights of the simplices.

                The first tuple should contain the vertices of the complex, and
                therefore must be a tensor of shape [num_vertices, d].

                Any following tuples should contain indices into the vertices tensor,
                and therefore must be a tensor of shape [num_simplices, k], where k+1 is the
                dimension of the simplex.

            vertex_dtype: The data type to use for the vertex coordinates.
            index_dtype: The data type to use for the simplex indices.
            weights_dtype: The data type to use for the simplex weights.
            device:
                The device to use for the tensors.
            n_type: The type of the simplicial complex. Currently only "simplicial" and "cubical"
                are supported.

        """
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        typematch = [vertex_dtype] + [index_dtype] * (len(args) - 1)
        dimensions = tuple(
            (
                torch.as_tensor(coords, device=device, dtype=typematch[i]),
                torch.as_tensor(weights, device=device, dtype=weights_dtype),
            )
            for i, (coords, weights) in enumerate(args)
        )
        return Complex(*dimensions, device=device, n_type=n_type)

    def to(self, device: torch.device) -> "Complex":
        """Moves the complex to the given device."""
        return Complex(*self.dimensions, device=device, n_type=self.n_type)

    def __getitem__(self, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the simplices of the given dimension."""
        return self.dimensions[dim]

    def get_coords(self, dim: int) -> torch.Tensor:
        """Returns the coordinates of the simplices of the given dimension."""
        return self.dimensions[dim][0]

    def get_weights(self, dim: int) -> torch.Tensor:
        """Returns the weights of the simplices of the given dimension."""
        return self.dimensions[dim][1]

    def top_dim(self) -> int:
        """Returns the top dimension of the complex."""
        return len(self) - 1

    def __len__(self) -> int:
        """Returns the number of dimensions in the complex."""
        return len(self.dimensions)

    def space_dim(self) -> int:
        """Returns the dimension of the space the complex is embedded in."""
        return self.dimensions[0][0].shape[1]
    
    def center_(self) -> "Complex":
        """
        Re-center the complex in-place so that the average vertex coordinate is at the origin.
        """
        if len(self.dimensions) == 0:
            return self

        v_coords, v_weights = self.dimensions[0]
        if v_coords.numel() == 0:
            return self

        center = v_coords.mean(dim=0)
        new_v_coords = (v_coords - center).contiguous()

        dims: list[Tuple[torch.Tensor, torch.Tensor]] = list(self.dimensions)
        dims[0] = (new_v_coords, v_weights)
        self.dimensions = tuple(dims)
        return self

    @staticmethod
    def _validate_dimensions(
        *args: Tuple[torch.Tensor, torch.Tensor], n_type: str
    ) -> None:
        for dim, simplex_list in enumerate(args):
            if simplex_list[0].dim() != 2:
                raise ValueError(
                    f"Dimension {dim} simplices must be a 2d tensor."
                    + f" Got {simplex_list[0].dim()} dimensions."
                )
            if simplex_list[1].dim() != 1:
                raise ValueError(
                    f"Dimension {dim} weights must be a 1d tensor."
                    + f" Got {simplex_list[1].dim()} dimensions."
                )
            if simplex_list[0].shape[0] != simplex_list[1].shape[0]:
                raise ValueError(
                    f"Dimension {dim} coordinates and weights must have the same number of simplices."
                    + f" Got {simplex_list[0].shape[0]} simplices and {simplex_list[1].shape[0]} weights."
                )

            if dim > 0:  # simplices, k > 0
                if n_type == "simplicial":
                    if simplex_list[0].shape[1] != dim + 1:
                        raise ValueError(
                            f"Dimension {dim} simplices must have {dim + 1} columns."
                            + f" Got {simplex_list[0].shape[1]} columns."
                        )
                elif n_type == "cubical":
                    if simplex_list[0].shape[1] != 2 ** dim:
                        raise ValueError(
                            f"Dimension {dim} simplices must have {2 ** dim} columns."
                            + f" Got {simplex_list[0].shape[1]} columns."
                        )
                else:  # warn that validation not implementod for n_type, but no error
                    warnings.warn(f"Validation not implemented for n_type {n_type}. Proceed with caution.")
