# Theano

This package contains ODL functionality related to [Theano](http://www.deeplearning.net/software/theano/).

## Content

* `TheanoOperator` in [layer.py](layer.py) wraps an ODL operator into a Theano operator.
  This allows using arbitrary ODL operators inside theano computational graphs.

## Example usage

The [examples](examples) folder contains examples on how to use the above functionality.
Specifically:

* [theano_layer_matrix.py](examples/theano_layer_matrix.py) shows how an ODL `MatrixOperator` can be converted to a theano layer.

There are also some rudimentary tests in the [test](test) folder.