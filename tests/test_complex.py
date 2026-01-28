"""Tests for the Complex class in tensor_complex.py"""

import torch
import pytest
import numpy as np

from pyect import Complex


class TestComplexConstruction:
    """Tests for Complex construction and initialization."""

    def test_simple_2d_triangle(self):
        """Test creating a simple 2D triangle complex."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        vweights = torch.ones(3)

        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]])
        eweights = torch.ones(3)

        fcoords = torch.tensor([[0, 1, 2]])
        fweights = torch.ones(1)

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
        )

        assert len(c) == 3
        assert c.top_dim() == 2
        assert c.space_dim() == 2

    def test_simple_3d_tetrahedron(self):
        """Test creating a 3D tetrahedron complex."""
        vcoords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0]
        ])
        vweights = torch.ones(4)

        ecoords = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        eweights = torch.ones(6)

        fcoords = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        fweights = torch.ones(4)

        tcoords = torch.tensor([[0, 1, 2, 3]])
        tweights = torch.ones(1)

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
            (tcoords, tweights),
        )

        assert len(c) == 4
        assert c.top_dim() == 3
        assert c.space_dim() == 3

    def test_vertices_only(self):
        """Test creating a complex with only vertices."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        vweights = torch.tensor([1.0, 2.0, 3.0])

        c = Complex((vcoords, vweights))

        assert len(c) == 1
        assert c.top_dim() == 0
        assert c.space_dim() == 2

    def test_custom_weights(self):
        """Test creating a complex with custom weights."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        vweights = torch.tensor([0.5, 1.5])

        ecoords = torch.tensor([[0, 1]])
        eweights = torch.tensor([2.0])

        c = Complex((vcoords, vweights), (ecoords, eweights))

        assert torch.allclose(c.get_weights(0), vweights)
        assert torch.allclose(c.get_weights(1), eweights)


class TestComplexCubical:
    """Tests for cubical complex type."""

    def test_cubical_square(self):
        """Test creating a cubical complex (square)."""
        vcoords = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ])
        vweights = torch.ones(4)

        # Edges have 2 vertices
        ecoords = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
        eweights = torch.ones(4)

        # Squares have 4 vertices in cubical complex
        scoords = torch.tensor([[0, 1, 2, 3]])
        sweights = torch.ones(1)

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (scoords, sweights),
            n_type="cubical"
        )

        assert c.n_type == "cubical"
        assert len(c) == 3


class TestComplexValidation:
    """Tests for Complex validation logic."""

    def test_invalid_coords_dimension(self):
        """Test that 1D coords tensor raises error."""
        vcoords = torch.tensor([0.0, 1.0, 2.0])  # 1D instead of 2D
        vweights = torch.ones(3)

        with pytest.raises(ValueError, match="must be a 2d tensor"):
            Complex((vcoords, vweights))

    def test_invalid_weights_dimension(self):
        """Test that 2D weights tensor raises error."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        vweights = torch.tensor([[1.0], [1.0]])  # 2D instead of 1D

        with pytest.raises(ValueError, match="must be a 1d tensor"):
            Complex((vcoords, vweights))

    def test_mismatched_coords_weights_count(self):
        """Test that mismatched counts raise error."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        vweights = torch.ones(3)  # 3 weights for 2 vertices

        with pytest.raises(ValueError, match="same number of simplices"):
            Complex((vcoords, vweights))

    def test_invalid_simplicial_edge_columns(self):
        """Test that edges with wrong column count raise error."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        vweights = torch.ones(3)

        ecoords = torch.tensor([[0, 1, 2]])  # 3 columns for dim-1 simplex
        eweights = torch.ones(1)

        with pytest.raises(ValueError, match="must have 2 columns"):
            Complex((vcoords, vweights), (ecoords, eweights))

    def test_invalid_simplicial_face_columns(self):
        """Test that faces with wrong column count raise error."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        vweights = torch.ones(3)

        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]])
        eweights = torch.ones(3)

        fcoords = torch.tensor([[0, 1]])  # 2 columns for dim-2 simplex
        fweights = torch.ones(1)

        with pytest.raises(ValueError, match="must have 3 columns"):
            Complex((vcoords, vweights), (ecoords, eweights), (fcoords, fweights))

    def test_invalid_cubical_square_columns(self):
        """Test that cubical squares with wrong column count raise error."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        vweights = torch.ones(4)

        ecoords = torch.tensor([[0, 1], [2, 3]])
        eweights = torch.ones(2)

        scoords = torch.tensor([[0, 1, 2]])  # 3 columns instead of 4 for cubical dim-2
        sweights = torch.ones(1)

        with pytest.raises(ValueError, match="must have 4 columns"):
            Complex(
                (vcoords, vweights),
                (ecoords, eweights),
                (scoords, sweights),
                n_type="cubical"
            )


class TestComplexAccess:
    """Tests for Complex accessor methods."""

    def test_getitem(self):
        """Test __getitem__ returns correct tuple."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        vweights = torch.tensor([1.0, 2.0])

        ecoords = torch.tensor([[0, 1]])
        eweights = torch.tensor([3.0])

        c = Complex((vcoords, vweights), (ecoords, eweights))

        v_coords, v_weights = c[0]
        assert torch.allclose(v_coords, vcoords)
        assert torch.allclose(v_weights, vweights)

        e_coords, e_weights = c[1]
        assert torch.allclose(e_coords, ecoords.to(torch.int64))
        assert torch.allclose(e_weights, eweights)

    def test_get_coords(self):
        """Test get_coords method."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        vweights = torch.ones(2)

        c = Complex((vcoords, vweights))

        assert torch.allclose(c.get_coords(0), vcoords)

    def test_get_weights(self):
        """Test get_weights method."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        vweights = torch.tensor([1.5, 2.5])

        c = Complex((vcoords, vweights))

        assert torch.allclose(c.get_weights(0), vweights)


class TestComplexCenter:
    """Tests for Complex centering functionality."""

    def test_center_moves_centroid_to_origin(self):
        """Test that center_() moves centroid to origin."""
        vcoords = torch.tensor([[1.0, 1.0], [3.0, 1.0], [2.0, 3.0]])
        vweights = torch.ones(3)

        c = Complex((vcoords, vweights))
        c.center_()

        new_coords = c.get_coords(0)
        centroid = new_coords.mean(dim=0)

        assert torch.allclose(centroid, torch.zeros(2), atol=1e-6)

    def test_center_preserves_relative_positions(self):
        """Test that center_() preserves relative positions."""
        vcoords = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        vweights = torch.ones(3)

        c = Complex((vcoords, vweights))

        # Compute original pairwise distances
        orig_dists = torch.cdist(vcoords, vcoords)

        c.center_()
        new_coords = c.get_coords(0)

        # Compute new pairwise distances
        new_dists = torch.cdist(new_coords, new_coords)

        assert torch.allclose(orig_dists, new_dists, atol=1e-6)

    def test_center_returns_self(self):
        """Test that center_() returns self for chaining."""
        vcoords = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        vweights = torch.ones(2)

        c = Complex((vcoords, vweights))
        result = c.center_()

        assert result is c


class TestComplexDevice:
    """Tests for Complex device handling."""

    def test_to_device(self):
        """Test moving complex to a device."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        vweights = torch.ones(2)

        c = Complex((vcoords, vweights))
        c_cpu = c.to(torch.device("cpu"))

        assert c_cpu.get_coords(0).device.type == "cpu"
        assert c_cpu.get_weights(0).device.type == "cpu"

    def test_device_parameter(self):
        """Test specifying device at construction."""
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        vweights = torch.ones(2)

        c = Complex((vcoords, vweights), device=torch.device("cpu"))

        assert c.get_coords(0).device.type == "cpu"


class TestComplexFromNumpy:
    """Tests for Complex.from_numpy constructor."""

    def test_from_numpy_basic(self):
        """Test creating Complex from numpy arrays."""
        vcoords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        vweights = np.ones(3)

        ecoords = np.array([[0, 1], [1, 2], [2, 0]])
        eweights = np.ones(3)

        c = Complex.from_numpy(
            (vcoords, vweights),
            (ecoords, eweights),
            device=torch.device("cpu")
        )

        assert len(c) == 2
        assert c.get_coords(0).dtype == torch.float32
        assert isinstance(c.get_coords(0), torch.Tensor)

    def test_from_numpy_preserves_values(self):
        """Test that from_numpy preserves array values."""
        vcoords = np.array([[1.5, 2.5], [3.5, 4.5]])
        vweights = np.array([0.1, 0.9])

        c = Complex.from_numpy((vcoords, vweights), device=torch.device("cpu"))

        assert torch.allclose(
            c.get_coords(0),
            torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
            atol=1e-6
        )
        assert torch.allclose(
            c.get_weights(0),
            torch.tensor([0.1, 0.9]),
            atol=1e-6
        )
