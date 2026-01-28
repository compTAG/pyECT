"""Tests for image preprocessing functions in preprocessing/image_processing.py"""

import torch
import pytest

from pyect import weighted_freudenthal, weighted_cubical, Complex


class TestWeightedFreudenthal:
    """Tests for weighted_freudenthal function."""

    def test_output_is_complex(self):
        """Test that output is a Complex object."""
        arr = torch.rand(5, 5)
        result = weighted_freudenthal(arr)

        assert isinstance(result, Complex)

    def test_output_dimensions(self):
        """Test that output complex has correct dimensions."""
        arr = torch.rand(5, 5)
        result = weighted_freudenthal(arr)

        # Should have vertices (0), edges (1), and triangles (2)
        assert len(result) == 3

    def test_vertices_are_2d(self):
        """Test that vertices are 2D coordinates."""
        arr = torch.rand(3, 4)
        result = weighted_freudenthal(arr)

        v_coords = result.get_coords(0)
        assert v_coords.shape[1] == 2

    def test_no_zero_weight_simplices(self):
        """Test that zero-weight pixels don't create edges/triangles."""
        arr = torch.zeros(3, 3)
        arr[1, 1] = 1.0  # Only center pixel is nonzero

        result = weighted_freudenthal(arr)

        # Should have 1 vertex (center pixel)
        v_coords = result.get_coords(0)
        assert v_coords.shape[0] == 1

        # Should have no edges (isolated vertex)
        e_coords = result.get_coords(1)
        assert e_coords.shape[0] == 0

    def test_full_image_edge_count(self):
        """Test edge count for fully nonzero image."""
        # For a 2x2 image, Freudenthal creates:
        # 4 vertices, horizontal+vertical+diagonal edges
        arr = torch.ones(2, 2)
        result = weighted_freudenthal(arr)

        v_coords = result.get_coords(0)
        assert v_coords.shape[0] == 4

        e_coords = result.get_coords(1)
        # 2 horizontal + 2 vertical + 1 diagonal = 5 edges
        assert e_coords.shape[0] == 5

    def test_full_image_triangle_count(self):
        """Test triangle count for fully nonzero image."""
        arr = torch.ones(2, 2)
        result = weighted_freudenthal(arr)

        f_coords = result.get_coords(2)
        # 2x2 image creates 2 triangles (upper and lower)
        assert f_coords.shape[0] == 2

    def test_weights_are_max_function(self):
        """Test that edge/triangle weights use max function."""
        arr = torch.tensor([[0.1, 0.5], [0.5, 0.9]])
        result = weighted_freudenthal(arr)

        # Check vertex weights match image values
        v_weights = result.get_weights(0)
        expected_weights = arr[arr != 0].flatten()
        assert torch.allclose(v_weights.sort()[0], expected_weights.sort()[0])

    def test_centering(self):
        """Test that vertices are centered around origin."""
        arr = torch.ones(3, 3)
        result = weighted_freudenthal(arr)

        v_coords = result.get_coords(0)
        centroid = v_coords.mean(dim=0)

        # Should be close to origin
        assert torch.allclose(centroid, torch.zeros(2), atol=1e-6)

    def test_device_parameter(self):
        """Test that device parameter works."""
        arr = torch.ones(3, 3)
        result = weighted_freudenthal(arr, device=torch.device("cpu"))

        assert result.get_coords(0).device.type == "cpu"

    def test_device_inheritance(self):
        """Test that device is inherited from input tensor."""
        arr = torch.ones(3, 3, device=torch.device("cpu"))
        result = weighted_freudenthal(arr)

        assert result.get_coords(0).device.type == "cpu"

    def test_sparse_image(self):
        """Test Freudenthal on sparse image."""
        arr = torch.zeros(5, 5)
        arr[0, 0] = 0.5
        arr[4, 4] = 0.5

        result = weighted_freudenthal(arr)

        v_coords = result.get_coords(0)
        # Only 2 nonzero pixels
        assert v_coords.shape[0] == 2

        e_coords = result.get_coords(1)
        # No edges (pixels are not adjacent)
        assert e_coords.shape[0] == 0

    def test_diagonal_neighbors(self):
        """Test that diagonal neighbors create edges and triangles."""
        arr = torch.zeros(3, 3)
        arr[0, 0] = 0.5
        arr[1, 1] = 0.5
        arr[0, 1] = 0.5  # Additional to form triangle

        result = weighted_freudenthal(arr)

        # Should have 3 vertices
        v_coords = result.get_coords(0)
        assert v_coords.shape[0] == 3


class TestWeightedCubical:
    """Tests for weighted_cubical function."""

    def test_output_is_complex(self):
        """Test that output is a Complex object."""
        arr = torch.rand(5, 5)
        result = weighted_cubical(arr)

        assert isinstance(result, Complex)

    def test_output_type_is_cubical(self):
        """Test that output complex type is cubical."""
        arr = torch.rand(3, 3)
        result = weighted_cubical(arr)

        assert result.n_type == "cubical"

    def test_output_dimensions(self):
        """Test that output complex has correct dimensions."""
        arr = torch.rand(5, 5)
        result = weighted_cubical(arr)

        # Should have vertices (0), edges (1), and squares (2)
        assert len(result) == 3

    def test_square_has_4_vertices(self):
        """Test that squares have 4 vertices (cubical complex)."""
        arr = torch.ones(2, 2)
        result = weighted_cubical(arr)

        s_coords = result.get_coords(2)
        # Each square should reference 4 vertices
        assert s_coords.shape[1] == 4

    def test_no_zero_weight_simplices(self):
        """Test that zero-weight pixels don't create edges/squares."""
        arr = torch.zeros(3, 3)
        arr[1, 1] = 1.0  # Only center pixel is nonzero

        result = weighted_cubical(arr)

        # Should have 1 vertex
        v_coords = result.get_coords(0)
        assert v_coords.shape[0] == 1

        # Should have no edges
        e_coords = result.get_coords(1)
        assert e_coords.shape[0] == 0

    def test_full_image_edge_count(self):
        """Test edge count for fully nonzero image."""
        arr = torch.ones(2, 2)
        result = weighted_cubical(arr)

        e_coords = result.get_coords(1)
        # 2 horizontal + 2 vertical = 4 edges (no diagonals in cubical)
        assert e_coords.shape[0] == 4

    def test_full_image_square_count(self):
        """Test square count for fully nonzero image."""
        arr = torch.ones(2, 2)
        result = weighted_cubical(arr)

        s_coords = result.get_coords(2)
        # 2x2 image creates 1 square
        assert s_coords.shape[0] == 1

    def test_centering(self):
        """Test that vertices are centered around origin."""
        arr = torch.ones(3, 3)
        result = weighted_cubical(arr)

        v_coords = result.get_coords(0)
        centroid = v_coords.mean(dim=0)

        # Should be close to origin
        assert torch.allclose(centroid, torch.zeros(2), atol=1e-6)

    def test_device_parameter(self):
        """Test that device parameter works."""
        arr = torch.ones(3, 3)
        result = weighted_cubical(arr, device=torch.device("cpu"))

        assert result.get_coords(0).device.type == "cpu"

    def test_larger_image(self):
        """Test cubical complex for larger image."""
        arr = torch.ones(5, 5)
        result = weighted_cubical(arr)

        v_coords = result.get_coords(0)
        assert v_coords.shape[0] == 25

        s_coords = result.get_coords(2)
        # 4x4 = 16 squares
        assert s_coords.shape[0] == 16


class TestFreudenthalVsCubical:
    """Comparison tests between Freudenthal and cubical complexes."""

    def test_same_vertex_count(self):
        """Test that both produce same vertex count."""
        arr = torch.ones(3, 3)

        freud = weighted_freudenthal(arr)
        cubic = weighted_cubical(arr)

        assert freud.get_coords(0).shape[0] == cubic.get_coords(0).shape[0]

    def test_different_edge_count(self):
        """Test that edge counts differ (Freudenthal has diagonals)."""
        arr = torch.ones(3, 3)

        freud = weighted_freudenthal(arr)
        cubic = weighted_cubical(arr)

        # Freudenthal should have more edges (includes diagonals)
        assert freud.get_coords(1).shape[0] > cubic.get_coords(1).shape[0]

    def test_different_top_dim_structure(self):
        """Test structural difference at top dimension."""
        arr = torch.ones(2, 2)

        freud = weighted_freudenthal(arr)
        cubic = weighted_cubical(arr)

        # Freudenthal: triangles (3 vertices each)
        assert freud.get_coords(2).shape[1] == 3

        # Cubical: squares (4 vertices each)
        assert cubic.get_coords(2).shape[1] == 4


class TestIntegrationWithWECT:
    """Integration tests with WECT."""

    def test_freudenthal_with_wect(self):
        """Test that Freudenthal complex works with WECT."""
        from pyect import WECT

        arr = torch.ones(5, 5) * 0.5
        c = weighted_freudenthal(arr)

        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        wect = WECT(dirs, num_heights=10)

        result = wect(c)

        assert result.shape == (2, 10)
        assert torch.isfinite(result).all()

    def test_cubical_with_wect(self):
        """Test that cubical complex works with WECT."""
        from pyect import WECT

        arr = torch.ones(5, 5) * 0.5
        c = weighted_cubical(arr)

        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        wect = WECT(dirs, num_heights=10)

        result = wect(c)

        assert result.shape == (2, 10)
        assert torch.isfinite(result).all()
