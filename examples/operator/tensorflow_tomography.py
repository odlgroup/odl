"""Example of how to solve a simple tomography problem using tensorflow."""

import tensorflow as tf
import numpy as np
import odl


with tf.Session() as sess:
    # Create ODL data structures
    space = odl.uniform_discr([-64, -64], [64, 64], [128, 128],
                              dtype='float32')
    geometry = odl.tomo.parallel_beam_geometry(space)
    ray_transform = odl.tomo.RayTransform(space, geometry)
    phantom = odl.phantom.shepp_logan(space, True)
    data = ray_transform(phantom)

    # Create tensorflow layer from odl operator
    odl_op_layer = odl.as_tensorflow_layer(ray_transform, 'RayTransform')
    x = tf.Variable(tf.constant(0.0, shape=space.shape), name="x")

    # Create constant right hand side
    y = tf.constant(np.asarray(data))

    # Define l2 loss function
    loss = tf.nn.l2_loss(odl_op_layer(x) - y)

    # Train using the adam optimizer
    optimizer = tf.train.AdamOptimizer(1e-1).minimize(loss)

    # Initialize all TF variables
    tf.global_variables_initializer().run()

    # Solve with an ODL callback to see what happens
    callback = odl.solvers.CallbackShow()

    for i in range(100):
        sess.run(optimizer)

        callback(space.element(x.eval()))
