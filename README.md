# pyECT

The Weighted Euler Characteristic Transform (WECT) is a mathematical tool
used to analyze and summarize geometric and topological features of data.
This package provides an efficient and simple implementation of the WECT using
PyTorch.

This codebase accompanies [this preprint](https://arxiv.org/abs/2511.03909).
If you use this package, please include the following citation in your work:
```
@misc{cisewskikehe2025vectorizedcomputationeulercharacteristic,
      title={Vectorized Computation of Euler Characteristic Functions and Transforms}, 
      author={Jessi Cisewski-Kehe and Brittany Terese Fasy and Alexander McCleary and Eli Quist and Jack Ruder},
      year={2025},
      eprint={2511.03909},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={https://arxiv.org/abs/2511.03909}, 
}
```

## Installation

To install `pyECT`, use pip:

```bash
pip install pyect 
```

Gudhi alpha-complex support is optional:

```bash
pip install pyect[gudhi]
```

The Gudhi integration lives at `pyect.integrations.gudhi` so Gudhi is not
imported by the core package. In `alpha_complex_to_filtration_data`,
`point_weights` are passed to Gudhi to construct the alpha filtration. The
pyECT simplex weights are `1.0` by default; pass `simplex_weight_fn` to use a
custom weighting rule, such as the max of the simplex vertex weights.

## Usage

Here's a simple example of how to use `pyECT`:

```python
from pyect import WECT

# Example data and weight function
data = [...]  # Replace with your data
weight_function = lambda x: x**2  # Replace with your weight function

# Compute the WECT
wect = WECT(data, weight_function)
result = wect.compute()

print("WECT result:", result)
```

For more detailed examples, please see the `/examples` directory.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the
repository and submit a pull request. For major changes, please open an issue
first to discuss what you'd like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE)
file for details.
