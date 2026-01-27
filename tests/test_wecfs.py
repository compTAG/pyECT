"""Tests for compute_wecfs function in wecfs.py"""

import torch
import pytest

from pyect.wecfs import compute_wecfs
from pyect import Complex


def build_triangle_complex_with_filters(device="cpu"):
    """Build a triangle complex with filter functions for testing."""
    # 3 vertices with 2 filter functions
    filters = torch.tensor([
        [-1.0, 0.0],   # vertex 0: filter values
        [0.0, 1.0],    # vertex 1: filter values
        [1.0, 0.0],    # vertex 2: filter values
    ], device=device)
    vweights = torch.ones(3, device=device)

    ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
    eweights = torch.ones(3, device=device)

    fcoords = torch.tensor([[0, 1, 2]], device=device)
    fweights = torch.ones(1, device=device)

    # Return as list of tuples (same format as Complex but with filters instead of coords)
    return [(filters, vweights), (ecoords, eweights), (fcoords, fweights)]


class TestComputeWecfsBasic:
    """Basic tests for compute_wecfs function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        complex_data = build_triangle_complex_with_filters()

        result = compute_wecfs(complex_data, num_vals=10)

        # 2 filter functions, 10 values
        assert result.shape == (2, 10)

    def test_output_is_finite(self):
        """Test that output contains no NaN or Inf values."""
        complex_data = build_triangle_complex_with_filters()

        result = compute_wecfs(complex_data, num_vals=10)

        assert torch.isfinite(result).all()

    def test_output_dtype(self):
        """Test that output dtype is float32."""
        complex_data = build_triangle_complex_with_filters()

        result = compute_wecfs(complex_data, num_vals=10)

        assert result.dtype == torch.float32

    def test_various_num_vals(self):
        """Test with various num_vals."""
        complex_data = build_triangle_complex_with_filters()

        for n in [2, 5, 10, 50, 100]:
            result = compute_wecfs(complex_data, num_vals=n)
            assert result.shape == (2, n)


class TestComputeWecfsSingleFilter:
    """Tests with single filter function."""

    def test_single_filter(self):
        """Test with single filter function."""
        device = torch.device("cpu")

        filters = torch.tensor([
            [-1.0],
            [0.0],
            [1.0],
        ], device=device)
        vweights = torch.ones(3, device=device)

        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        eweights = torch.ones(3, device=device)

        fcoords = torch.tensor([[0, 1, 2]], device=device)
        fweights = torch.ones(1, device=device)

        complex_data = [(filters, vweights), (ecoords, eweights), (fcoords, fweights)]

        result = compute_wecfs(complex_data, num_vals=5)

        assert result.shape == (1, 5)

    def test_single_filter_vertices_only(self):
        """Test with single filter, vertices only."""
        device = torch.device("cpu")

        filters = torch.tensor([
            [-1.0],
            [0.0],
            [1.0],
        ], device=device)
        vweights = torch.tensor([1.0, 2.0, 3.0], device=device)

        complex_data = [(filters, vweights)]

        result = compute_wecfs(complex_data, num_vals=5)

        assert result.shape == (1, 5)


class TestComputeWecfsWeighted:
    """Tests with weighted complexes."""

    def test_weighted_vertices(self):
        """Test with non-uniform vertex weights."""
        device = torch.device("cpu")

        filters = torch.tensor([
            [-1.0],
            [0.0],
            [1.0],
        ], device=device)
        vweights = torch.tensor([0.5, 1.0, 1.5], device=device)

        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        eweights = torch.ones(3, device=device)

        complex_data = [(filters, vweights), (ecoords, eweights)]

        result = compute_wecfs(complex_data, num_vals=5)

        assert result.shape == (1, 5)
        assert torch.isfinite(result).all()

    def test_weighted_edges(self):
        """Test with non-uniform edge weights."""
        device = torch.device("cpu")

        filters = torch.tensor([
            [-1.0],
            [0.0],
            [1.0],
        ], device=device)
        vweights = torch.ones(3, device=device)

        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        eweights = torch.tensor([0.5, 1.0, 0.5], device=device)

        complex_data = [(filters, vweights), (ecoords, eweights)]

        result = compute_wecfs(complex_data, num_vals=5)

        assert result.shape == (1, 5)
        assert torch.isfinite(result).all()


class TestComputeWecfsMultipleFilters:
    """Tests with multiple filter functions."""

    def test_three_filters(self):
        """Test with three filter functions."""
        device = torch.device("cpu")

        filters = torch.tensor([
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 0.0],
            [-1.0, -1.0, 1.0],
        ], device=device)
        vweights = torch.ones(3, device=device)

        complex_data = [(filters, vweights)]

        result = compute_wecfs(complex_data, num_vals=10)

        assert result.shape == (3, 10)

    def test_many_filters(self):
        """Test with many filter functions."""
        device = torch.device("cpu")
        num_filters = 10

        filters = torch.randn(5, num_filters, device=device)
        vweights = torch.ones(5, device=device)

        ecoords = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], device=device)
        eweights = torch.ones(4, device=device)

        complex_data = [(filters, vweights), (ecoords, eweights)]

        result = compute_wecfs(complex_data, num_vals=20)

        assert result.shape == (10, 20)


class TestComputeWecfsEdgeCases:
    """Edge case tests for compute_wecfs."""

    def test_single_vertex(self):
        """Test with single vertex."""
        device = torch.device("cpu")

        filters = torch.tensor([[0.5, -0.5]], device=device)
        vweights = torch.ones(1, device=device)

        complex_data = [(filters, vweights)]

        result = compute_wecfs(complex_data, num_vals=5)

        assert result.shape == (2, 5)
        assert torch.isfinite(result).all()

    def test_constant_filter(self):
        """Test with constant filter function."""
        device = torch.device("cpu")

        filters = torch.full((3, 1), 0.5, device=device)
        vweights = torch.ones(3, device=device)

        complex_data = [(filters, vweights)]

        result = compute_wecfs(complex_data, num_vals=5)

        assert result.shape == (1, 5)

    def test_near_zero_filter(self):
        """Test with near-zero filter function."""
        device = torch.device("cpu")

        filters = torch.full((3, 1), 1e-6, device=device)
        vweights = torch.ones(3, device=device)

        complex_data = [(filters, vweights)]

        result = compute_wecfs(complex_data, num_vals=5)

        assert result.shape == (1, 5)
        assert torch.isfinite(result).all()


class TestComputeWecfsHigherDimensions:
    """Tests with higher dimensional complexes."""

    def test_tetrahedron(self):
        """Test with tetrahedron complex."""
        device = torch.device("cpu")

        # 4 vertices with 2 filter functions
        filters = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, -1.0],
            [0.5, 0.5],
        ], device=device)
        vweights = torch.ones(4, device=device)

        ecoords = torch.tensor([
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
        ], device=device)
        eweights = torch.ones(6, device=device)

        fcoords = torch.tensor([
            [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
        ], device=device)
        fweights = torch.ones(4, device=device)

        tcoords = torch.tensor([[0, 1, 2, 3]], device=device)
        tweights = torch.ones(1, device=device)

        complex_data = [
            (filters, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
            (tcoords, tweights),
        ]

        result = compute_wecfs(complex_data, num_vals=10)

        assert result.shape == (2, 10)
        assert torch.isfinite(result).all()


class TestComputeWecfsCumulative:
    """Tests verifying cumulative sum behavior."""

    def test_is_cumulative(self):
        """Test that output uses cumulative sum."""
        device = torch.device("cpu")

        filters = torch.tensor([
            [-1.0],
            [0.0],
            [1.0],
        ], device=device)
        vweights = torch.ones(3, device=device)

        complex_data = [(filters, vweights)]

        result = compute_wecfs(complex_data, num_vals=3)

        # For vertices only, result should be cumulative count
        # All values should be non-decreasing
        diffs = torch.diff(result, dim=1)
        assert (diffs >= -1e-6).all()  # Allow small numerical error


class TestComputeWecfsDevice:
    """Device handling tests."""

    def test_device_cpu(self):
        """Test on CPU device."""
        complex_data = build_triangle_complex_with_filters(device=torch.device("cpu"))

        result = compute_wecfs(complex_data, num_vals=10)

        assert result.device.type == "cpu"

    def test_device_consistency(self):
        """Test that output device matches input device."""
        device = torch.device("cpu")

        filters = torch.randn(3, 2, device=device)
        vweights = torch.ones(3, device=device)

        complex_data = [(filters, vweights)]

        result = compute_wecfs(complex_data, num_vals=10)

        assert result.device == device
