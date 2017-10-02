"""Example of how to convert an ODL operator to a tensorflow layer.

In this example we take an ODL operator given by a `MatrixOperator` and
convert it into a tensorflow layer that can be used inside any tensorflow
computational graph.

We also demonstrate that we can compute the "gradients" properly using the
adjoint of the derivative.
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np

import odl
import odl.contrib.tensorflow

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
print('Tensorflow eval:                  ',
      y_tf.eval().ravel())

# Compare result with pure ODL
print('ODL eval:                         ',
      odl_op(x))

# Evaluate the adjoint of the derivative, called gradient in tensorflow
print('Tensorflow adjoint of derivative: ',
      tf.gradients(y_tf, [x_tf], z_tf)[0].eval().ravel())

# Compare result with pure ODL
print('ODL adjoint of derivative:        ',
      odl_op.derivative(x).adjoint(z))
