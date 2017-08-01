"""Example of how to convert an ODL operator to a tensorflow layer."""

import tensorflow as tf
import numpy as np
import odl

sess = tf.InteractiveSession()

# Define ODL operator
matrix = np.array([[1, 2],
                   [0, 0],
                   [0, 1]], dtype='float32')
odl_op = odl.MatrixOperator(matrix)

# Define evaluation points
x = [1., 2.]
z = [1., 2., 3.]

# Add empty axes for batch and channel
x_tf = tf.constant(x)[None, ..., None]
z_tf = tf.constant(z)[None, ..., None]

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(
        odl_op, 'MatrixOperator')
y_tf = odl_op_layer(x_tf)

# Evaluate using tensorflow
print(y_tf.eval().ravel())

# Compare result with pure ODL
print(odl_op(x))

# Evaluate the adjoint of the derivative, called gradient in tensorflow
print(tf.gradients(y_tf, [x_tf], z_tf)[0].eval().ravel())

# Compare result with pure ODL
print(odl_op.derivative(x).adjoint(z))
