# Tensorflow

This package contains ODL functionality related to [TensorFlow](https://www.tensorflow.org/).

## Content

* `TensorflowOperator` in [operator.py](operator.py) wraps a tensorflow expression into an ODL operator.
  This allows using tensorflow neural networks as operators in ODL.
* `as_tensorflow_layer` in [layer.py](layer.py) wraps an ODL operator into a tensorflow layer.
  This allows using arbitrary ODL operators inside tensorflow neural networks.
* `TensorflowSpace` in [space.py](space.py) is a `TensorSpace` which uses tensorflow as a backend.

## Example usage

The [examples](examples) folder contains examples on how to use the above functionality.
Specifically:

* [tensorflow_layer_matrix.py](examples/tensorflow_layer_matrix.py) shows how an ODL `MatrixOperator` can be converted to a tensorflow layer.
* [tensorflow_layer_productspace.py](examples/tensorflow_layer_productspace.py) shows how an ODL operator acting on `ProductSpace`s can be converted to a tensorflow layer.
* [tensorflow_layer_ray_transform.py](examples/tensorflow_layer_ray_transform.py) shows how a `RayTransform` can be converted to a tensorflow layer.
* [tensorflow_operator_matrix.py](examples/tensorflow_operator_matrix.py) shows how `tf.matmul` can be used as an ODL operator.
* [tensorflow_tomography.py](examples/tensorflow_tomography.py) shows how tensorflow optimizers can be used with ODL operators to solve inverse problems.

There are also some rudimentary tests in the [test](test) folder.