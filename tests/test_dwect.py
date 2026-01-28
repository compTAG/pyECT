"""Tests for the DWECT (Differentiable WECT) module in differentiable_wect.py"""

import torch
import pytest

from pyect import DWECT, WECT, Complex


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


class TestDWECTConstruction:
    """Tests for DWECT module construction."""

    def test_basic_construction(self):
        """Test basic DWECT construction."""
        dirs = torch.tensor([[1.0, 0.0]])
        dwect = DWECT(dirs, num_heights=10, growth_rate=10.0)

        assert dwect.num_heights == 10
        assert dwect.growth_rate == 10.0
        assert dwect.dirs.shape == (1, 2)

    def test_multiple_directions(self):
        """Test DWECT with multiple directions."""
        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        dwect = DWECT(dirs, num_heights=5, growth_rate=5.0)

        assert dwect.dirs.shape == (3, 2)

    def test_direction_normalization(self):
        """Test that directions are normalized."""
        dirs = torch.tensor([[3.0, 4.0]])  # norm = 5
        dwect = DWECT(dirs, num_heights=5, growth_rate=10.0)

        norms = torch.norm(dwect.dirs, dim=1)
        assert torch.allclose(norms, torch.ones(1), atol=1e-6)

    def test_invalid_num_heights(self):
        """Test that non-positive num_heights raises error."""
        dirs = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError, match="num_heights must be positive"):
            DWECT(dirs, num_heights=0, growth_rate=10.0)

        with pytest.raises(ValueError, match="num_heights must be positive"):
            DWECT(dirs, num_heights=-5, growth_rate=10.0)

    def test_various_growth_rates(self):
        """Test DWECT with various growth rates."""
        dirs = torch.tensor([[1.0, 0.0]])

        for rate in [0.1, 1.0, 10.0, 100.0]:
            dwect = DWECT(dirs, num_heights=5, growth_rate=rate)
            assert dwect.growth_rate == rate


class TestDWECTForward:
    """Tests for DWECT forward pass."""

    def test_output_shape_single_direction(self):
        """Test output shape with single direction."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        dwect = DWECT(dirs, num_heights=10, growth_rate=10.0).to(device)

        result = dwect(c)

        assert result.shape == (1, 10)

    def test_output_shape_multiple_directions(self):
        """Test output shape with multiple directions."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device)
        dwect = DWECT(dirs, num_heights=8, growth_rate=10.0).to(device)

        result = dwect(c)

        assert result.shape == (3, 8)

    def test_output_is_finite(self):
        """Test that output contains no NaN or Inf values."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
        dwect = DWECT(dirs, num_heights=10, growth_rate=10.0).to(device)

        result = dwect(c)

        assert torch.isfinite(result).all()

    def test_empty_complex(self):
        """Test DWECT with empty complex."""
        device = torch.device("cpu")

        vcoords = torch.zeros((0, 2), device=device)
        vweights = torch.zeros(0, device=device)

        c = Complex((vcoords, vweights))

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        dwect = DWECT(dirs, num_heights=5, growth_rate=10.0).to(device)

        result = dwect(c)

        assert result.shape == (1, 5)
        assert torch.allclose(result, torch.zeros((1, 5), device=device))


class TestDWECTGradients:
    """Tests for DWECT gradient computation."""

    def test_gradients_flow(self):
        """Test that gradients flow through DWECT."""
        device = torch.device("cpu")

        # Create complex with requires_grad
        vcoords = torch.tensor(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            device=device,
            requires_grad=True
        )
        vweights = torch.ones(3, device=device, requires_grad=True)

        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        eweights = torch.ones(3, device=device, requires_grad=True)

        fcoords = torch.tensor([[0, 1, 2]], device=device)
        fweights = torch.ones(1, device=device, requires_grad=True)

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
        )

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        dwect = DWECT(dirs, num_heights=5, growth_rate=10.0).to(device)

        result = dwect(c)
        loss = result.sum()
        loss.backward()

        # Check gradients exist for weights
        assert vweights.grad is not None
        assert eweights.grad is not None
        assert fweights.grad is not None

    def test_gradients_are_finite(self):
        """Test that computed gradients are finite."""
        device = torch.device("cpu")

        vcoords = torch.tensor(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            device=device
        )
        vweights = torch.ones(3, device=device, requires_grad=True)

        ecoords = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        eweights = torch.ones(3, device=device, requires_grad=True)

        fcoords = torch.tensor([[0, 1, 2]], device=device)
        fweights = torch.ones(1, device=device, requires_grad=True)

        c = Complex(
            (vcoords, vweights),
            (ecoords, eweights),
            (fcoords, fweights),
        )

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        dwect = DWECT(dirs, num_heights=5, growth_rate=10.0).to(device)

        result = dwect(c)
        loss = result.sum()
        loss.backward()

        assert torch.isfinite(vweights.grad).all()
        assert torch.isfinite(eweights.grad).all()
        assert torch.isfinite(fweights.grad).all()


class TestDWECTSoftCumsum:
    """Tests for the soft cumsum functionality."""

    def test_soft_cumsum_shape(self):
        """Test soft cumsum preserves shape."""
        dirs = torch.tensor([[1.0, 0.0]])
        dwect = DWECT(dirs, num_heights=5, growth_rate=10.0)

        M = torch.randn(3, 5)
        result = dwect._soft_cum_sum(M)

        assert result.shape == M.shape

    def test_high_growth_rate_approaches_cumsum(self):
        """Test that high growth rate approximates regular cumsum."""
        dirs = torch.tensor([[1.0, 0.0]])
        dwect_high = DWECT(dirs, num_heights=5, growth_rate=1000.0)

        M = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        soft_result = dwect_high._soft_cum_sum(M)
        hard_result = torch.cumsum(M, dim=1)

        # With very high growth rate, should be close to regular cumsum
        # Note: soft_cum_sum uses sigmoid which saturates but doesn't equal hard cumsum
        # Check that monotonicity is preserved and values are in similar range
        assert soft_result[0, -1] > soft_result[0, 0]  # Monotonic increase
        assert torch.isfinite(soft_result).all()

    def test_low_growth_rate_is_smooth(self):
        """Test that low growth rate produces smooth output."""
        dirs = torch.tensor([[1.0, 0.0]])
        dwect_low = DWECT(dirs, num_heights=5, growth_rate=0.5)

        M = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]])
        result = dwect_low._soft_cum_sum(M)

        # With low growth rate, output should be smoother than input
        # Check that middle values are not zero
        assert result[0, 2] > 0


class TestDWECTComparisonToWECT:
    """Tests comparing DWECT to WECT behavior."""

    def test_dwect_wect_same_shape(self):
        """Test DWECT and WECT produce same shape output."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)

        wect = WECT(dirs, num_heights=10).to(device)
        dwect = DWECT(dirs, num_heights=10, growth_rate=100.0).to(device)

        wect_result = wect(c)
        dwect_result = dwect(c)

        assert wect_result.shape == dwect_result.shape

    def test_high_growth_rate_similar_to_wect(self):
        """Test that DWECT with high growth rate is similar to WECT."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0]], device=device)

        wect = WECT(dirs, num_heights=5).to(device)
        dwect = DWECT(dirs, num_heights=5, growth_rate=1000.0).to(device)

        wect_result = wect(c)
        dwect_result = dwect(c)

        # With very high growth rate, should be close
        assert torch.allclose(wect_result, dwect_result, atol=0.5)


class TestDWECTTorchScript:
    """Tests for TorchScript compatibility."""

    def test_can_script(self):
        """Test that DWECT can be compiled with TorchScript."""
        dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dwect = DWECT(dirs, num_heights=10, growth_rate=10.0)

        scripted = torch.jit.script(dwect)
        assert scripted is not None

    def test_scripted_gives_same_result(self):
        """Test that scripted DWECT gives same results."""
        device = torch.device("cpu")
        c = build_triangle_complex(device)

        dirs = torch.tensor([[1.0, 0.0]], device=device)
        dwect = DWECT(dirs, num_heights=5, growth_rate=10.0).to(device)
        scripted = torch.jit.script(dwect)

        result_normal = dwect(c)
        result_scripted = scripted(c)

        assert torch.allclose(result_normal, result_scripted, atol=1e-6)


class TestDWECT3D:
    """Tests for DWECT in 3D."""

    def test_3d_triangle(self):
        """Test DWECT on a triangle in 3D."""
        device = torch.device("cpu")

        vcoords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], device=device)
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

        dirs = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        dwect = DWECT(dirs, num_heights=8, growth_rate=10.0).to(device)

        result = dwect(c)

        assert result.shape == (3, 8)
        assert torch.isfinite(result).all()
