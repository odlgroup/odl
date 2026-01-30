# pytorch

This package functionality for integrating ODL with [pytorch](http://pytorch.org/).

All functionality in this subpackage assumes that the ``torch`` module in the ``pytorch`` package has been installed. See the [pytorch installation guide](https://github.com/pytorch/pytorch#installation) for further instructions.

## Content

- `OperatorModule` in [`operator.py`](operator.py) wraps an ODL Operator into a pytorch ``Module``.
  This allows using arbitrary ODL operators in pytorch computational graphs and fully supports both forward evaluation and backward (gradient) evaluation.

## Example usage

The [examples](examples) folder contains examples on how to use the above functionality.
Specifically:

- [`operator.py`](examples/operator.py) shows how an ODL `MatrixOperator` can be converted to a pytorch module and how to compute gradients using backpropagation.

There are also some rudimentary tests in the [`test`](test) folder.
