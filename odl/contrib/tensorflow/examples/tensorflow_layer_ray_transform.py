"""Example of how to convert a RayTransform operator to a tensorflow layer.

This example is similar to ``tensorflow_layer_matrix``, but demonstrates how
more advanced operators, such as a ray transform, can be handled.
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

space = odl.uniform_discr([-64, -64], [64, 64], [128, 128],
                          dtype='float32')
geometry = odl.tomo.parallel_beam_geometry(space)
ray_transform = odl.tomo.RayTransform(space, geometry)

x = tf.constant(np.asarray(ray_transform.domain.one()))
z = tf.constant(np.asarray(ray_transform.range.one()))

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(
    ray_transform, 'RayTransform')


# Add empty axes for batch and channel
x_reshaped = x[None, ..., None]
z_reshaped = z[None, ..., None]

# Lazily apply operator in tensorflow
y = odl_op_layer(x_reshaped)

# Evaluate using tensorflow
print(y.eval())

# Compare result with pure ODL
print(ray_transform(x.eval()))

# Evaluate the adjoint of the derivative, called gradient in tensorflow
# We need to scale by cell size to get correct value since the derivative
# in tensorflow uses unweighted spaces.
scale = ray_transform.range.cell_volume / ray_transform.domain.cell_volume
print(tf.gradients(y, [x_reshaped], z_reshaped)[0].eval() * scale)

# Compare result with pure ODL
print(ray_transform.derivative(x.eval()).adjoint(z.eval()))
