"""Example of how to convert an ODL operator to a tensorflow layer."""

import tensorflow as tf
import numpy as np
import odl

sess = tf.InteractiveSession()

# Define ODL operator
space = odl.uniform_discr([0, 0], [1, 1], [10, 10], dtype='float32')

odl_op = odl.Gradient(space)

# Define evaluation points
x = odl_op.domain.one()
z = odl_op.range.one()

# Add empty axes for batch and channel
x_tf = tf.ones([1, 10, 10, 1])
z_tf = tf.ones([1, 2, 10, 10, 1])

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(
        odl_op, 'Gradient')
y_tf = odl_op_layer(x_tf)

# Evaluate using tensorflow
print(y_tf.eval().squeeze())

# Compare result with pure ODL
print(odl_op(x))

# Evaluate the adjoint of the derivative, called gradient in tensorflow
scale = 1 / space.cell_volume
print(tf.gradients(y_tf, [x_tf], z_tf)[0].eval().squeeze() * scale)

# Compare result with pure ODL
print(odl_op.derivative(x).adjoint(z))
