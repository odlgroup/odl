"""Example of how to solve a simple tomography problem using tensorflow.

In this example, we solve the TV regularized tomography problem::

    min_x ||A(x) - b||_2^2 + 50 * ||grad(x)||_1

using a gradient descent method (ADAM).
"""

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow

sess = tf.InteractiveSession()

# Create ODL data structures
space = odl.uniform_discr([-64, -64], [64, 64], [128, 128],
                          dtype='float32')
geometry = odl.tomo.parallel_beam_geometry(space)
ray_transform = odl.tomo.RayTransform(space, geometry)
grad = odl.Gradient(space)

# Create data
phantom = odl.phantom.shepp_logan(space, True)
data = ray_transform(phantom)
noisy_data = data + odl.phantom.white_noise(data.space)

# Create tensorflow layers from odl operators
ray_transform_layer = odl.contrib.tensorflow.as_tensorflow_layer(
    ray_transform, name='RayTransform')
grad_layer = odl.contrib.tensorflow.as_tensorflow_layer(
    grad, name='Gradient')
x = tf.Variable(tf.zeros(shape=space.shape), name="x")

# Create constant right hand side
y = tf.constant(np.asarray(noisy_data))

# Add empty axes for batch and channel
x_reshaped = x[None, ..., None]
y_reshaped = y[None, ..., None]

# Define loss function
loss = (tf.reduce_sum((ray_transform_layer(x_reshaped) - y_reshaped) ** 2) +
        50 * tf.reduce_sum(tf.abs(grad_layer(x_reshaped))))

# Train using the ADAM optimizer
optimizer = tf.train.AdamOptimizer(1e-1).minimize(loss)

# Initialize all TF variables
tf.global_variables_initializer().run()

# Solve with an ODL callback to see what happens
callback = odl.solvers.CallbackShow()

for i in range(200):
    sess.run(optimizer)

    callback(space.element(x.eval()))
