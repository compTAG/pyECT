"""Tests for direction sampling functions in directions.py"""

import torch
import pytest
import math

from pyect import sample_directions_2d, sample_directions_3d


class TestSampleDirections2D:
    """Tests for 2D direction sampling."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        for n in [1, 4, 8, 16, 100]:
            dirs = sample_directions_2d(n)
            assert dirs.shape == (n, 2)

    def test_unit_vectors(self):
        """Test that all directions are unit vectors."""
        dirs = sample_directions_2d(100)
        norms = torch.norm(dirs, dim=1)

        assert torch.allclose(norms, torch.ones(100), atol=1e-6)

    def test_evenly_spaced(self):
        """Test that directions are evenly spaced on circle."""
        n = 8
        dirs = sample_directions_2d(n)

        # Convert to angles
        angles = torch.atan2(dirs[:, 1], dirs[:, 0])

        # Sort angles
        angles_sorted, _ = torch.sort(angles)

        # Compute differences (accounting for wrap-around)
        diffs = torch.diff(angles_sorted)
        expected_diff = 2 * math.pi / n

        assert torch.allclose(diffs, torch.full_like(diffs, expected_diff), atol=1e-5)

    def test_first_direction(self):
        """Test that first direction is along x-axis."""
        dirs = sample_directions_2d(4)

        # First direction should be (1, 0)
        assert torch.allclose(dirs[0], torch.tensor([1.0, 0.0]), atol=1e-6)

    def test_contiguous(self):
        """Test that output is contiguous."""
        dirs = sample_directions_2d(10)
        assert dirs.is_contiguous()

    def test_single_direction(self):
        """Test sampling a single direction."""
        dirs = sample_directions_2d(1)

        assert dirs.shape == (1, 2)
        assert torch.allclose(torch.norm(dirs[0]), torch.tensor(1.0), atol=1e-6)

    def test_device_cpu(self):
        """Test sampling on CPU device."""
        dirs = sample_directions_2d(10, device=torch.device("cpu"))

        assert dirs.device.type == "cpu"

    def test_dtype(self):
        """Test output dtype is float32."""
        dirs = sample_directions_2d(10)

        assert dirs.dtype == torch.float32


class TestSampleDirections3D:
    """Tests for 3D direction sampling (Fibonacci spiral)."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        for n in [1, 4, 8, 16, 100]:
            dirs = sample_directions_3d(n)
            assert dirs.shape == (n, 3)

    def test_unit_vectors(self):
        """Test that all directions are unit vectors."""
        dirs = sample_directions_3d(100)
        norms = torch.norm(dirs, dim=1)

        assert torch.allclose(norms, torch.ones(100), atol=1e-6)

    def test_covers_sphere(self):
        """Test that directions cover the sphere reasonably."""
        dirs = sample_directions_3d(100)

        # Check that y values span from near -1 to near 1
        y_vals = dirs[:, 1]
        assert y_vals.min() < -0.9
        assert y_vals.max() > 0.9

    def test_hemisphere_coverage(self):
        """Test that directions cover both hemispheres."""
        dirs = sample_directions_3d(50)

        # Count directions in each hemisphere (z > 0 and z < 0)
        upper_count = (dirs[:, 2] > 0).sum().item()
        lower_count = (dirs[:, 2] < 0).sum().item()

        # Should be roughly balanced
        assert upper_count > 10
        assert lower_count > 10

    def test_contiguous(self):
        """Test that output is contiguous."""
        dirs = sample_directions_3d(10)
        assert dirs.is_contiguous()

    def test_single_direction(self):
        """Test sampling a single direction."""
        dirs = sample_directions_3d(1)

        assert dirs.shape == (1, 3)
        assert torch.allclose(torch.norm(dirs[0]), torch.tensor(1.0), atol=1e-6)

    def test_device_cpu(self):
        """Test sampling on CPU device."""
        dirs = sample_directions_3d(10, device=torch.device("cpu"))

        assert dirs.device.type == "cpu"

    def test_dtype(self):
        """Test output dtype is float32."""
        dirs = sample_directions_3d(10)

        assert dirs.dtype == torch.float32

    def test_unique_directions(self):
        """Test that all sampled directions are unique."""
        dirs = sample_directions_3d(50)

        # Check that no two directions are identical
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                assert not torch.allclose(dirs[i], dirs[j], atol=1e-4)

    def test_no_clustering(self):
        """Test that directions don't cluster excessively."""
        dirs = sample_directions_3d(100)

        # Compute pairwise distances
        dists = torch.cdist(dirs, dirs)

        # Set diagonal to large value to ignore self-distances
        dists = dists + torch.eye(100) * 100

        # Minimum distance should not be too small
        min_dist = dists.min()
        assert min_dist > 0.1


class TestDirectionsIntegration:
    """Integration tests using sampled directions with WECT."""

    def test_2d_directions_with_wect(self):
        """Test that sampled 2D directions work with WECT."""
        from pyect import WECT, Complex

        dirs = sample_directions_2d(4)
        wect = WECT(dirs, num_heights=10)

        # Create a simple complex
        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        vweights = torch.ones(3)
        c = Complex((vcoords, vweights))

        result = wect(c)
        assert result.shape == (4, 10)
        assert torch.isfinite(result).all()

    def test_3d_directions_with_wect(self):
        """Test that sampled 3D directions work with WECT."""
        from pyect import WECT, Complex

        dirs = sample_directions_3d(6)
        wect = WECT(dirs, num_heights=10)

        # Create a simple 3D complex
        vcoords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0]
        ])
        vweights = torch.ones(4)
        c = Complex((vcoords, vweights))

        result = wect(c)
        assert result.shape == (6, 10)
        assert torch.isfinite(result).all()
