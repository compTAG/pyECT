"""Optional Gudhi integration helpers."""

from collections import defaultdict
from math import isfinite, sqrt
from typing import Callable, Optional, Sequence

import torch


def gudhi_simplex_tree_to_filtration_data(
    simplex_tree,
    *,
    max_dim: Optional[int] = None,
    simplex_weight_fn: Optional[Callable[[Sequence[int], int, float], float]] = None,
    use_alpha_instead_of_alpha_square: bool = False,
    dtype=torch.float32,
    device=None,
):
    """Convert a Gudhi SimplexTree to ``compute_wecfs_general`` input data.

    Args:
        simplex_tree: A Gudhi SimplexTree with filtration values assigned to
            simplices.
        max_dim: Optional maximum simplex dimension to include.
        simplex_weight_fn: Optional function
            ``simplex_weight_fn(simplex, dim, filtration_value) -> float``.
            Do not include the Euler sign; ``compute_wecfs_general`` handles it.
        use_alpha_instead_of_alpha_square: Gudhi alpha complexes return squared
            alpha values by default. Set this to True to use alpha values.
        dtype: Torch dtype for the output tensors.
        device: Torch device for the output tensors.

    Returns:
        A list where ``filtration_data[d] = (simplex_filters, simplex_weights)``.
        ``simplex_filters`` has shape ``(num_d_simplices, 1)`` and
        ``simplex_weights`` has shape ``(num_d_simplices,)``.
    """

    filters_by_dim = defaultdict(list)
    weights_by_dim = defaultdict(list)

    for simplex, filt in simplex_tree.get_filtration():
        dim = len(simplex) - 1

        if max_dim is not None and dim > max_dim:
            continue

        filt = float(filt)

        if not isfinite(filt):
            raise ValueError("Encountered non-finite Gudhi filtration value.")

        if use_alpha_instead_of_alpha_square:
            if filt < 0:
                raise ValueError(
                    "Cannot take sqrt of a negative filtration value. "
                    "This can happen for weighted alpha complexes."
                )
            filt = sqrt(filt)

        weight = 1.0
        if simplex_weight_fn is not None:
            weight = float(simplex_weight_fn(simplex, dim, filt))

        filters_by_dim[dim].append(filt)
        weights_by_dim[dim].append(weight)

    if not filters_by_dim:
        raise ValueError("The Gudhi SimplexTree contains no simplices.")

    filtration_data = []

    for dim in range(max(filters_by_dim) + 1):
        simplex_filters = torch.tensor(
            filters_by_dim.get(dim, []),
            dtype=dtype,
            device=device,
        ).reshape(-1, 1)

        simplex_weights = torch.tensor(
            weights_by_dim.get(dim, []),
            dtype=dtype,
            device=device,
        )

        filtration_data.append((simplex_filters, simplex_weights))

    return filtration_data


def alpha_complex_to_filtration_data(
    points,
    *,
    point_weights=None,
    max_alpha_square=float("inf"),
    max_dim=None,
    simplex_weight_fn=None,
    use_alpha_instead_of_alpha_square=False,
    dtype=torch.float32,
    device=None,
):
    """Build a Gudhi AlphaComplex and convert it to pyECT filtration data.

    ``point_weights`` are passed to Gudhi to construct a weighted alpha
    filtration. They do not change the pyECT simplex weights, which are one by
    default unless ``simplex_weight_fn`` is provided.
    """

    try:
        import gudhi
    except ImportError as exc:
        raise ImportError(
            "Gudhi support requires the optional dependency 'gudhi'. "
            "Install it with: pip install pyect[gudhi]"
        ) from exc

    kwargs = {"points": points}
    if point_weights is not None:
        kwargs["weights"] = point_weights

    alpha_complex = gudhi.AlphaComplex(**kwargs)
    simplex_tree = alpha_complex.create_simplex_tree(
        max_alpha_square=max_alpha_square
    )

    filtration_data = gudhi_simplex_tree_to_filtration_data(
        simplex_tree,
        max_dim=max_dim,
        simplex_weight_fn=simplex_weight_fn,
        use_alpha_instead_of_alpha_square=use_alpha_instead_of_alpha_square,
        dtype=dtype,
        device=device,
    )

    return filtration_data, simplex_tree
