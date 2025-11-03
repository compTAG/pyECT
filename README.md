# pyECT

The Weighted Euler Characteristic Transform (WECT) is a mathematical tool
used to analyze and summarize geometric and topological features of data.
This package provides an efficient and simple implementation of the WECT using
PyTorch.

This codebase accompanies the following paper (and should be cited if you use
this package):

```
TODO: Add Citation
```

## Installation

To install `pyECT`, use pip:

```bash
pip install pyect 
```

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
