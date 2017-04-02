"""Example of how to solve a simple tomography problem using tensorflow."""

import tensorflow as tf
import numpy as np
import odl


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


with tf.Session() as sess:
    # Create ODL data structures
    space = odl.uniform_discr([-64, -64], [64, 64], [512, 512],
                              dtype='float32')
    geometry = odl.tomo.parallel_beam_geometry(space, angles=20)
    ray_transform = odl.tomo.RayTransform(space, geometry)
    phantom = odl.phantom.shepp_logan(space, True)
    data = ray_transform(phantom)
    fbp = odl.tomo.fbp_op(ray_transform)(data)

    # Create tensorflow layer from odl operator
    odl_op_layer = odl.as_tensorflow_layer(ray_transform,
                                           'RayTransform')
    x = tf.constant(np.asarray(fbp), name="x")
    x = x[None, :, :, None]

    # Create constant right hand side
    y = tf.constant(np.asarray(data))

    init = np.random.randn(*[3, 3, 1, 32]) * 0.1
    W = tf.Variable(tf.constant(init, dtype='float32'))
    x = tf.nn.relu(conv2d(x, W))

    for i in range(2):
        init = np.random.randn(*[3, 3, 32, 32]) * 0.1
        W = tf.Variable(tf.constant(init, dtype='float32'))
        x = tf.nn.relu(conv2d(x, W))

    init = np.random.randn(*[3, 3, 32, 1]) * 0.1
    W = tf.Variable(tf.constant(init, dtype='float32'))
    x = tf.nn.relu(conv2d(x, W))

    # Reshape for ODL
    x = x[0, :, :, 0]

    # Define l2 loss function
    loss = tf.nn.l2_loss(odl_op_layer(x) - y)

    # Train using the adam optimizer
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # Initialize all TF variables
    tf.global_variables_initializer().run()

    # Solve with an ODL callback to see what happens
    callback = odl.solvers.CallbackShow(clim=[0, 1])

    for i in range(1000):
        sess.run(optimizer)
        callback(space.element(x.eval()))
