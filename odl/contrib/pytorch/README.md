# pytorch

This package functionality for integrating ODL with [pytorch](http://pytorch.org/).

## Content

* `TorchOperator` in [layer.py](layer.py) wraps an ODL Operator into a pytorch ``Function``.
  This allows using arbitrary ODL operators in pytorch computational graphs and fully supports both forward evaluation and backward (gradient) evaluation.

## Example usage

The [examples](examples) folder contains examples on how to use the above functionality.
Specifically:

* [basic.py](examples/basic.py) shows how an ODL `MatrixOperator` can be converted to a pytorch layer and how to compute gradients using backwards computations.

There are also some rudimentary tests in the [test](test) folder.
