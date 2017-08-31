# Theano

This package contains ODL functionality related to [Theano](http://www.deeplearning.net/software/theano/).

## Content

* `TheanoOperator` in [layer.py](layer.py) wraps an ODL Operator into a Theano operator.
  This allows using arbitrary ODL operators inside Theano computational graphs and fully supports both forward evaluation and backward (gradient) evaluation.

## Example usage

The [examples](examples) folder contains examples on how to use the above functionality.
Specifically:

* [theano_layer_matrix.py](examples/theano_layer_matrix.py) shows how an ODL `MatrixOperator` can be converted to a Theano layer.

There are also some rudimentary tests in the [test](test) folder.