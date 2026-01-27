"""Pytest configuration and shared fixtures for pyECT tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cpu")


@pytest.fixture
def triangle_vertices():
    """Return vertices for a simple 2D triangle."""
    return torch.tensor([
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ])


@pytest.fixture
def triangle_edges():
    """Return edge indices for a simple triangle."""
    return torch.tensor([
        [0, 1],
        [1, 2],
        [2, 0]
    ])


@pytest.fixture
def triangle_faces():
    """Return face indices for a simple triangle."""
    return torch.tensor([[0, 1, 2]])


@pytest.fixture
def tetrahedron_vertices():
    """Return vertices for a simple 3D tetrahedron."""
    return torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ])


@pytest.fixture
def tetrahedron_edges():
    """Return edge indices for a tetrahedron."""
    return torch.tensor([
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3]
    ])


@pytest.fixture
def tetrahedron_faces():
    """Return face indices for a tetrahedron."""
    return torch.tensor([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])


@pytest.fixture
def tetrahedron_tets():
    """Return tetrahedron indices."""
    return torch.tensor([[0, 1, 2, 3]])


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require CUDA"
    )


def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def has_mps():
    """Check if MPS (Apple Silicon) is available."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
