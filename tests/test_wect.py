"""Tests for the WECT module in wect.py"""

import torch
import pytest

from pyect import WECT, Complex


def build_triangle_complex(device="cpu"):
    """Build a simple triangle complex for testing."""
    vcoords = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device=device
    )
    vweights = torch.ones(3, device=device)

    ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
    eweights = torch.ones(3, device=device)

    fcoords = torch.tensor([[0, 1, 2]], device=device)
    fweights = torch.ones(1, device=device)

    return Complex(
        (vcoords, vweights),
        (ecoords, eweights),
        (fcoords, fweights),
    )


def build_line_segment_complex(device="cpu"):
    """Build a simple line segment (1D) complex."""
    vcoords = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], device=device)
    vweights = torch.ones(2, device=device)

    ecoords = torch.tensor([[0, 1]], device=device)
    eweights = torch.ones(1, device=device)

    return Complex((vcoords, vweights), (ecoords, eweights))


class TestWECTConstruction:
    """Tests for WECT module construction."""

    def test_basic_construction(self):
        """Test basic WECT construction."""
        dirs = torch.tensor([[1.0, 0.0]])
        wect = WECT(dirs, num_heights=10)

        assert wect.num_heights == 10
        assert wect.dirs.shape == (1, 2)

    def test_multiple_directions(self):
        """Test WECT with multiple directions."""
        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        wect = WECT(dirs, num_heights=5)

        assert wect.dirs.shape == (3, 2)

    def test_direction_normalization(self):
        """Test that directions are normalized."""
        dirs = torch.tensor([[3.0, 4.0]])  # norm = 5
        wect = WECT(dirs, num_heights=5)

        norms = torch.norm(wect.dirs, dim=1)
        assert torch.allclose(norms, torch.ones(1), atol=1e-6)

    def test_invalid_num_heights(self):
        """Test that non-positive num_heights raises error."""
        dirs = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError, match="num_heights must be positive"):
            WECT(dirs, num_heights=0)

        with pytest.raises(ValueError, match="num_heights must be positive"):
            WECT(dirs, num_heights=-5)


class TestWECTForward:
    """Tests for WECT forward pass."""

    def test_output_shape_single_direction(self):
        """Test output shape with single direction."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=10).to(device)

        result = wect(c)

        assert result.shape == (1, 10)

    def test_output_shape_multiple_directions(self):
        """Test output shape with multiple directions."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device)
        wect = WECT(dirs, num_heights=8).to(device)

        result = wect(c)

        assert result.shape == (3, 8)

    def test_output_is_finite(self):
        """Test that output contains no NaN or Inf values."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
        wect = WECT(dirs, num_heights=10).to(device)

        result = wect(c)

        assert torch.isfinite(result).all()

    def test_exact_triangle_horizontal_direction(self):
        """Test exact WECT values for triangle with horizontal direction."""
        device = torch.device("cpu")

        vcoords = torch.tensor(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device=device
        )
        vweights = torch.ones(3, device=device)
        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        eweights = torch.ones(3, device=device)
        fcoords = torch.tensor([[0, 1, 2]], device=device)
        fweights = torch.ones(1, device=device)

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
        )

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=3).to(device)

        result = wect(c)

        # From the existing test, expected result for this configuration
        expected = torch.tensor([1.0, 1.0, 1.0], device=device)
        assert torch.allclose(result[0], expected, atol=1e-6)

    def test_weighted_complex(self):
        """Test WECT with weighted complex."""
        device = torch.device("cpu")

        vcoords = torch.tensor(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device=device
        )
        vweights = torch.tensor([0.5, 1.0, 1.5], device=device)
        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        eweights = torch.tensor([0.5, 1.0, 0.5], device=device)
        fcoords = torch.tensor([[0, 1, 2]], device=device)
        fweights = torch.tensor([0.5], device=device)

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
        )

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=3).to(device)

        result = wect(c)

        expected = torch.tensor([0.5, 1.0, 1.5], device=device)
        assert torch.allclose(result[0], expected, atol=1e-6)


class TestWECTEdgeCases:
    """Tests for WECT edge cases."""

    def test_empty_complex(self):
        """Test WECT with empty complex."""
        device = torch.device("cpu")

        vcoords = torch.zeros((0, 2), device=device)
        vweights = torch.zeros(0, device=device)

        c = Complex((vcoords, vweights))

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=5).to(device)

        result = wect(c)

        assert result.shape == (1, 5)
        assert torch.allclose(result, torch.zeros((1, 5), device=device))

    def test_single_vertex(self):
        """Test WECT with single vertex."""
        device = torch.device("cpu")

        vcoords = torch.tensor([[0.0, 0.0]], device=device)
        vweights = torch.ones(1, device=device)

        c = Complex((vcoords, vweights))

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=5).to(device)

        result = wect(c)

        assert result.shape == (1, 5)
        assert torch.isfinite(result).all()

    def test_vertices_at_origin(self):
        """Test WECT when all vertices are at origin."""
        device = torch.device("cpu")

        vcoords = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device=device)
        vweights = torch.ones(2, device=device)

        c = Complex((vcoords, vweights))

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=5).to(device)

        result = wect(c)

        assert result.shape == (1, 5)
        assert torch.isfinite(result).all()

    def test_vertices_only_no_edges(self):
        """Test WECT with vertices but no edges/faces."""
        device = torch.device("cpu")

        vcoords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], device=device)
        vweights = torch.ones(3, device=device)

        c = Complex((vcoords, vweights))

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=3).to(device)

        result = wect(c)

        # Just vertices, Euler characteristic = number of vertices at each level
        assert result.shape == (1, 3)
        assert torch.isfinite(result).all()


class TestWECT3D:
    """Tests for WECT in 3D."""

    def test_3d_tetrahedron(self):
        """Test WECT on a 3D tetrahedron."""
        device = torch.device("cpu")

        vcoords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0]
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

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
            (tcoords, tweights),
        )

        dirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device)
        wect = WECT(dirs, num_heights=10).to(device)

        result = wect(c)

        assert result.shape == (3, 10)
        assert torch.isfinite(result).all()

    def test_3d_multiple_directions(self):
        """Test WECT with multiple 3D directions."""
        device = torch.device("cpu")

        vcoords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], device=device)
        vweights = torch.ones(3, device=device)

        ecoords = torch.tensor([[0, 1], [0, 2], [1, 2]], device=device)
        eweights = torch.ones(3, device=device)

        fcoords = torch.tensor([[0, 1, 2]], device=device)
        fweights = torch.ones(1, device=device)

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
        )

        dirs = torch.randn(5, 3, device=device)
        wect = WECT(dirs, num_heights=8).to(device)

        result = wect(c)

        assert result.shape == (5, 8)


class TestWECTTorchScript:
    """Tests for TorchScript compatibility."""

    def test_can_script(self):
        """Test that WECT can be compiled with TorchScript."""
        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        wect = WECT(dirs, num_heights=10)

        scripted = torch.jit.script(wect)
        assert scripted is not None

    def test_scripted_gives_same_result(self):
        """Test that scripted WECT gives same results."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=5).to(device)
        scripted = torch.jit.script(wect)

        result_normal = wect(c)
        result_scripted = scripted(c)

        assert torch.allclose(result_normal, result_scripted, atol=1e-6)


class TestWECTEvalMode:
    """Tests for WECT in evaluation mode."""

    def test_eval_mode(self):
        """Test WECT works in eval mode."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        wect = WECT(dirs, num_heights=5).to(device).eval()

        result = wect(c)

        assert result.shape == (1, 5)
        assert torch.isfinite(result).all()
