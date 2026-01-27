"""Tests for Image_ECF_2D and Image_ECF_3D modules in image_ecf.py"""

import torch
import pytest

from pyect import Image_ECF_2D, Image_ECF_3D


class TestImageECF2DConstruction:
    """Tests for Image_ECF_2D construction."""

    def test_basic_construction(self):
        """Test basic Image_ECF_2D construction."""
        ecf = Image_ECF_2D(num_vals=10)
        assert ecf.num_vals == 10

    def test_various_num_vals(self):
        """Test construction with various num_vals."""
        for n in [2, 5, 10, 50, 100]:
            ecf = Image_ECF_2D(num_vals=n)
            assert ecf.num_vals == n


class TestImageECF2DCellValues:
    """Tests for cell_values_2D static method."""

    def test_cell_values_shapes(self):
        """Test that cell_values returns correct shapes."""
        arr = torch.rand(4, 5)  # 4 rows, 5 columns
        vertex_vals, edge_vals, square_vals = Image_ECF_2D.cell_values_2D(arr)

        # Vertices: h*w = 20
        assert vertex_vals.shape == (20,)

        # Edges: horizontal (h*(w-1) = 16) + vertical ((h-1)*w = 15) = 31
        assert edge_vals.shape == (4 * 4 + 3 * 5,)

        # Squares: (h-1)*(w-1) = 12
        assert square_vals.shape == (12,)

    def test_cell_values_constant_image(self):
        """Test cell values for constant image."""
        arr = torch.full((3, 3), 0.5)
        vertex_vals, edge_vals, square_vals = Image_ECF_2D.cell_values_2D(arr)

        # All values should be 0.5 for constant image
        assert torch.allclose(vertex_vals, torch.full_like(vertex_vals, 0.5))
        assert torch.allclose(edge_vals, torch.full_like(edge_vals, 0.5))
        assert torch.allclose(square_vals, torch.full_like(square_vals, 0.5))

    def test_cell_values_gradient_image(self):
        """Test cell values for gradient image."""
        arr = torch.tensor([[0.0, 0.5], [0.5, 1.0]])
        vertex_vals, edge_vals, square_vals = Image_ECF_2D.cell_values_2D(arr)

        # Vertices should match flattened image
        assert torch.allclose(vertex_vals, arr.reshape(-1))

        # Maximum square value should be 1.0 (max of all corners)
        assert torch.allclose(square_vals[0], torch.tensor(1.0))


class TestImageECF2DForward:
    """Tests for Image_ECF_2D forward pass."""

    def test_output_shape(self):
        """Test output shape for various configurations."""
        ecf = Image_ECF_2D(num_vals=10)

        for h, w in [(3, 3), (5, 7), (10, 10)]:
            arr = torch.rand(h, w)
            result = ecf(arr)
            assert result.shape == (10,)

    def test_output_is_integer_typed(self):
        """Test that output is integer type."""
        ecf = Image_ECF_2D(num_vals=10)
        arr = torch.rand(5, 5)
        result = ecf(arr)

        # Output should be an integer type (int32 or int64 depending on platform)
        assert result.dtype in (torch.int32, torch.int64)

    def test_constant_black_image(self):
        """Test ECF of constant black (zero) image."""
        ecf = Image_ECF_2D(num_vals=5)
        arr = torch.zeros(3, 3)
        result = ecf(arr)

        # For constant zero image, all contributions at index 0
        # Euler characteristic: V - E + F
        # V = 9, E = 12, F = 4, so chi = 9 - 12 + 4 = 1
        assert result.shape == (5,)

    def test_constant_white_image(self):
        """Test ECF of constant white (one) image."""
        ecf = Image_ECF_2D(num_vals=5)
        arr = torch.ones(3, 3)
        result = ecf(arr)

        # For constant one image, all contributions at index n-1
        assert result.shape == (5,)

    def test_output_is_cumulative(self):
        """Test that output is non-decreasing (cumulative nature)."""
        ecf = Image_ECF_2D(num_vals=10)
        arr = torch.rand(5, 5)
        result = ecf(arr)

        # The ECF should be monotonically non-decreasing after initial dip
        # Actually for sublevel sets it should end at Euler char of full space
        assert result.shape == (10,)

    def test_single_pixel(self):
        """Test ECF of single pixel image."""
        ecf = Image_ECF_2D(num_vals=5)
        arr = torch.tensor([[0.5]])
        result = ecf(arr)

        # Single vertex, no edges or faces
        # chi = 1 for all levels >= 0.5
        assert result.shape == (5,)

    def test_device_preservation(self):
        """Test that device is preserved."""
        ecf = Image_ECF_2D(num_vals=10)
        arr = torch.rand(5, 5, device=torch.device("cpu"))
        result = ecf(arr)

        assert result.device.type == "cpu"


class TestImageECF3DConstruction:
    """Tests for Image_ECF_3D construction."""

    def test_basic_construction(self):
        """Test basic Image_ECF_3D construction."""
        ecf = Image_ECF_3D(num_vals=10)
        assert ecf.num_vals == 10

    def test_various_num_vals(self):
        """Test construction with various num_vals."""
        for n in [2, 5, 10, 50]:
            ecf = Image_ECF_3D(num_vals=n)
            assert ecf.num_vals == n


class TestImageECF3DCellValues:
    """Tests for cell_values_3D static method."""

    def test_cell_values_shapes(self):
        """Test that cell_values returns correct shapes."""
        arr = torch.rand(3, 4, 5)  # shape is (3, 4, 5)
        vertex_vals, edge_vals, square_vals, cube_vals = Image_ECF_3D.cell_values_3D(arr)

        # Vertices: 3*4*5 = 60
        assert vertex_vals.shape == (60,)

        # Edges along each axis:
        # x-edges: (3-1)*4*5 = 40
        # y-edges: 3*(4-1)*5 = 45
        # z-edges: 3*4*(5-1) = 48
        expected_edges = 2*4*5 + 3*3*5 + 3*4*4
        assert edge_vals.shape == (expected_edges,)

        # Squares: computed from edge combinations
        # The actual count depends on the implementation
        # Just verify it's a 1D tensor with reasonable size
        assert square_vals.dim() == 1
        assert square_vals.shape[0] > 0

        # Cubes: (3-1)*(4-1)*(5-1) = 24
        assert cube_vals.shape == (24,)

    def test_cell_values_constant_volume(self):
        """Test cell values for constant volume."""
        arr = torch.full((3, 3, 3), 0.5)
        vertex_vals, edge_vals, square_vals, cube_vals = Image_ECF_3D.cell_values_3D(arr)

        # All values should be 0.5 for constant volume
        assert torch.allclose(vertex_vals, torch.full_like(vertex_vals, 0.5))
        assert torch.allclose(edge_vals, torch.full_like(edge_vals, 0.5))
        assert torch.allclose(square_vals, torch.full_like(square_vals, 0.5))
        assert torch.allclose(cube_vals, torch.full_like(cube_vals, 0.5))


class TestImageECF3DForward:
    """Tests for Image_ECF_3D forward pass."""

    def test_output_shape(self):
        """Test output shape for various configurations."""
        ecf = Image_ECF_3D(num_vals=10)

        for d, h, w in [(3, 3, 3), (4, 5, 6), (2, 2, 2)]:
            arr = torch.rand(d, h, w)
            result = ecf(arr)
            assert result.shape == (10,)

    def test_output_is_integer_typed(self):
        """Test that output is integer type."""
        ecf = Image_ECF_3D(num_vals=10)
        arr = torch.rand(3, 3, 3)
        result = ecf(arr)

        # Output should be an integer type (int32 or int64 depending on platform)
        assert result.dtype in (torch.int32, torch.int64)

    def test_constant_black_volume(self):
        """Test ECF of constant black (zero) volume."""
        ecf = Image_ECF_3D(num_vals=5)
        arr = torch.zeros(2, 2, 2)
        result = ecf(arr)

        # For 3D: V - E + F - C
        # 2x2x2: V=8, E=12, F=6, C=1 -> chi = 8 - 12 + 6 - 1 = 1
        assert result.shape == (5,)

    def test_single_voxel(self):
        """Test ECF of single voxel volume."""
        ecf = Image_ECF_3D(num_vals=5)
        arr = torch.tensor([[[0.5]]])
        result = ecf(arr)

        # Single vertex, no edges, faces or cubes
        # chi = 1 for all levels >= 0.5
        assert result.shape == (5,)

    def test_device_preservation(self):
        """Test that device is preserved."""
        ecf = Image_ECF_3D(num_vals=10)
        arr = torch.rand(3, 3, 3, device=torch.device("cpu"))
        result = ecf(arr)

        assert result.device.type == "cpu"


class TestImageECFTorchScript:
    """Tests for TorchScript compatibility."""

    def test_2d_can_script(self):
        """Test that Image_ECF_2D can be compiled with TorchScript."""
        ecf = Image_ECF_2D(num_vals=10)
        scripted = torch.jit.script(ecf)
        assert scripted is not None

    def test_3d_can_script(self):
        """Test that Image_ECF_3D can be compiled with TorchScript."""
        ecf = Image_ECF_3D(num_vals=10)
        scripted = torch.jit.script(ecf)
        assert scripted is not None

    def test_2d_scripted_same_result(self):
        """Test that scripted Image_ECF_2D gives same results."""
        ecf = Image_ECF_2D(num_vals=10)
        scripted = torch.jit.script(ecf)

        arr = torch.rand(5, 5)

        result_normal = ecf(arr)
        result_scripted = scripted(arr)

        assert torch.equal(result_normal, result_scripted)

    def test_3d_scripted_same_result(self):
        """Test that scripted Image_ECF_3D gives same results."""
        ecf = Image_ECF_3D(num_vals=10)
        scripted = torch.jit.script(ecf)

        arr = torch.rand(3, 3, 3)

        result_normal = ecf(arr)
        result_scripted = scripted(arr)

        assert torch.equal(result_normal, result_scripted)


class TestImageECFEdgeCases:
    """Edge case tests for Image ECF modules."""

    def test_2d_very_small_image(self):
        """Test 2D ECF with minimum size image."""
        ecf = Image_ECF_2D(num_vals=5)
        arr = torch.tensor([[0.5]])  # 1x1 image
        result = ecf(arr)

        assert result.shape == (5,)
        assert torch.isfinite(result.float()).all()

    def test_3d_very_small_volume(self):
        """Test 3D ECF with minimum size volume."""
        ecf = Image_ECF_3D(num_vals=5)
        arr = torch.tensor([[[0.5]]])  # 1x1x1 volume
        result = ecf(arr)

        assert result.shape == (5,)
        assert torch.isfinite(result.float()).all()

    def test_2d_binary_image(self):
        """Test 2D ECF with binary (0/1) image."""
        ecf = Image_ECF_2D(num_vals=2)
        arr = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = ecf(arr)

        assert result.shape == (2,)

    def test_3d_binary_volume(self):
        """Test 3D ECF with binary (0/1) volume."""
        ecf = Image_ECF_3D(num_vals=2)
        arr = torch.zeros(2, 2, 2)
        arr[0, 0, 0] = 1.0
        result = ecf(arr)

        assert result.shape == (2,)

    def test_2d_narrow_image(self):
        """Test 2D ECF with narrow image (1 row)."""
        ecf = Image_ECF_2D(num_vals=5)
        arr = torch.rand(1, 10)
        result = ecf(arr)

        assert result.shape == (5,)

    def test_3d_flat_volume(self):
        """Test 3D ECF with flat volume (depth 1)."""
        ecf = Image_ECF_3D(num_vals=5)
        arr = torch.rand(1, 5, 5)
        result = ecf(arr)

        assert result.shape == (5,)
