"""Tests for optional Gudhi integration helpers."""

import sys
import types

import pytest
import torch

from pyect import compute_wecfs_general
from pyect.integrations.gudhi import (
    alpha_complex_to_filtration_data,
    gudhi_simplex_tree_to_filtration_data,
)


class FakeSimplexTree:
    """Small stand-in for Gudhi's SimplexTree API."""

    def __init__(self, filtration):
        self.filtration = filtration

    def get_filtration(self):
        return iter(self.filtration)


def test_simplex_tree_conversion_uses_unit_weights_by_default():
    simplex_tree = FakeSimplexTree([
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([0, 1], 1.0),
        ([0, 1, 2], 2.0),
    ])

    filtration_data = gudhi_simplex_tree_to_filtration_data(simplex_tree)

    assert len(filtration_data) == 3
    assert filtration_data[0][0].shape == (3, 1)
    assert filtration_data[1][0].shape == (1, 1)
    assert filtration_data[2][0].shape == (1, 1)

    for _, simplex_weights in filtration_data:
        assert torch.equal(simplex_weights, torch.ones_like(simplex_weights))


def test_simplex_weight_callback_can_use_max_vertex_weight():
    vertex_weights = [0.5, 2.0, 1.25]
    simplex_tree = FakeSimplexTree([
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([0, 2], 1.0),
        ([0, 1, 2], 2.0),
    ])

    def max_vertex_weight(simplex, dim, filtration_value):
        return max(vertex_weights[vertex] for vertex in simplex)

    filtration_data = gudhi_simplex_tree_to_filtration_data(
        simplex_tree,
        simplex_weight_fn=max_vertex_weight,
    )

    assert torch.allclose(
        filtration_data[0][1],
        torch.tensor([0.5, 2.0, 1.25]),
    )
    assert torch.allclose(filtration_data[1][1], torch.tensor([1.25]))
    assert torch.allclose(filtration_data[2][1], torch.tensor([2.0]))


def test_simplex_tree_conversion_respects_dtype_and_device():
    device = torch.device("cpu")
    simplex_tree = FakeSimplexTree([
        ([0], 0.0),
        ([0, 1], 4.0),
    ])

    filtration_data = gudhi_simplex_tree_to_filtration_data(
        simplex_tree,
        dtype=torch.float64,
        device=device,
    )

    for simplex_filters, simplex_weights in filtration_data:
        assert simplex_filters.dtype == torch.float64
        assert simplex_weights.dtype == torch.float64
        assert simplex_filters.device == device
        assert simplex_weights.device == device


def test_simplex_tree_conversion_can_use_alpha_instead_of_alpha_square():
    simplex_tree = FakeSimplexTree([
        ([0], 0.0),
        ([0, 1], 4.0),
    ])

    filtration_data = gudhi_simplex_tree_to_filtration_data(
        simplex_tree,
        use_alpha_instead_of_alpha_square=True,
    )

    assert torch.allclose(filtration_data[0][0], torch.tensor([[0.0]]))
    assert torch.allclose(filtration_data[1][0], torch.tensor([[2.0]]))


def test_simplex_tree_conversion_rejects_non_finite_filtration():
    simplex_tree = FakeSimplexTree([([0], float("inf"))])

    with pytest.raises(ValueError, match="non-finite"):
        gudhi_simplex_tree_to_filtration_data(simplex_tree)


def test_simplex_tree_conversion_rejects_negative_sqrt():
    simplex_tree = FakeSimplexTree([([0], -1.0)])

    with pytest.raises(ValueError, match="negative"):
        gudhi_simplex_tree_to_filtration_data(
            simplex_tree,
            use_alpha_instead_of_alpha_square=True,
        )


def test_alpha_complex_passes_point_weights_to_gudhi(monkeypatch):
    calls = {}
    simplex_tree = FakeSimplexTree([([0], 0.0)])

    class FakeAlphaComplex:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def create_simplex_tree(self, max_alpha_square):
            calls["max_alpha_square"] = max_alpha_square
            return simplex_tree

    monkeypatch.setitem(
        sys.modules,
        "gudhi",
        types.SimpleNamespace(AlphaComplex=FakeAlphaComplex),
    )

    points = [[0.0, 0.0]]
    point_weights = [0.25]

    filtration_data, returned_simplex_tree = alpha_complex_to_filtration_data(
        points,
        point_weights=point_weights,
        max_alpha_square=2.0,
    )

    assert calls["kwargs"] == {"points": points, "weights": point_weights}
    assert calls["max_alpha_square"] == 2.0
    assert returned_simplex_tree is simplex_tree
    assert torch.equal(filtration_data[0][1], torch.ones(1))


def test_gudhi_alpha_complex_wecf_smoke():
    pytest.importorskip("gudhi")

    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    filtration_data, simplex_tree = alpha_complex_to_filtration_data(points)
    wecf = compute_wecfs_general(filtration_data, num_vals=10)

    assert simplex_tree.num_simplices() > 0
    assert isinstance(wecf, torch.Tensor)
    assert wecf.shape == (1, 10)
    assert torch.isfinite(wecf).all()
