# Visualization examples

These examples show to use the visualization capabilities of ODL to view data.

## Basic usage examples

Example | Purpose | Complexity
------- | ------- | ----------
[`show_vector.py`](show_vector.py) | Using `Tensor.show` | low
[`show_1d.py`](show_1d.py) | Using `DiscreteLpElement.show` in 1D | low
[`show_2d.py`](show_2d.py) | Using `DiscreteLpElement.show` in 2D | low
[`show_2d.py`](show_2d_complex.py) | Using `DiscreteLpElement.show` in 2D with complex data | low
[`show_productspace.py`](show_productspace.py) | Using `ProductSpaceElement.show` | low
[`visualize_vector_examples.py`](visualize_vector_examples.py) | Show all example vectors in `DiscreteLp.examples` | low

## Real time updating

Example | Purpose | Complexity
------- | ------- | ----------
[`show_update_1d.py`](show_update_1d.py) | Using show and updating the figure in real time in 1d | low
[`show_update_2d.py`](show_update_2d.py) | Using show and updating the figure in real time in 2d | low
[`show_callback.py`](show_callback.py) | Using `odl.solvers.CallbackShow` | low
