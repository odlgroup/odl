"""Example of how to solve a simple tomography problem using tensorflow."""

import tensorflow as tf
import numpy as np
import odl


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def var(x):
    return tf.Variable(tf.constant(x, dtype='float32'))


with tf.Session() as sess:
    # Create ODL data structures
    space = odl.uniform_discr([-64, -64], [64, 64], [256, 256],
                              dtype='float32')
    geometry = odl.tomo.parallel_beam_geometry(space)
    ray_transform = odl.tomo.RayTransform(space, geometry)
    phantom = odl.phantom.shepp_logan(space, True)
    data = ray_transform(phantom)
    noisy_data = data + odl.phantom.white_noise(ray_transform.range) * np.mean(data) * 0.1
    fbp = odl.tomo.fbp_op(ray_transform)(noisy_data)

    # Create tensorflow layer from odl operator
    odl_op_layer = odl.as_tensorflow_layer(ray_transform,
                                           'RayTransform')
    odl_op_layer_adjoint = odl.as_tensorflow_layer(ray_transform.adjoint,
                                                   'RayTransformAdjoint')
    x = tf.constant(np.asarray(fbp)[None, :, :, None], name="x")

    x_true = tf.constant(np.asarray(phantom)[None, :, :, None], name="x_true")

    s = tf.Variable(tf.constant(np.zeros([1, 256, 256, 5]), dtype='float32'))

    # Create constant right hand side
    y = tf.constant(np.asarray(noisy_data)[None, :, :, None])

    w1 = var(np.random.randn(*[3, 3, 7, 32]) * 0.01)
    b1 = var(np.random.randn(*[1, 256, 256, 32]) * 0.001)

    w2 = var(np.random.randn(*[3, 3, 32, 32]) * 0.01)
    b2 = var(np.random.randn(*[1, 256, 256, 32]) * 0.001)

    w3 = var(np.random.randn(*[3, 3, 32, 6]) * 0.01)
    b3 = var(np.random.randn(*[1, 256, 256, 6]) * 0.001)

    n_iter = 5
    x_vals = []

    loss = 0
    for i in range(n_iter):
        # Reshape for ODL
        residual = odl_op_layer(x) - y
        dx = odl_op_layer_adjoint(residual)

        dx = tf.concat([x, dx, s], axis=3)

        dx = tf.nn.relu(conv2d(dx, w1) + b1)
        dx = tf.nn.relu(conv2d(dx, w2) + b2)
        dx = tf.nn.dropout(dx, 0.95)
        dx = tf.nn.tanh(conv2d(dx, w3) + b3)

        s = dx[..., 1:]
        dx = dx[..., 0][..., None]

        x = x + dx

        x_vals.append(x)

        loss = loss + tf.nn.l2_loss(x - x_true) * (2. ** (i - n_iter))

    # Train using the adam optimizer
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Initialize all TF variables
    tf.global_variables_initializer().run()

    # Solve with an ODL callback to see what happens
    callback = odl.solvers.CallbackShow(clim=[0, 1])

    for i in range(1000):
        if i < 100:
            sess.run(optimizer, feed_dict={learning_rate: 0.001})
        else:
            sess.run(optimizer, feed_dict={learning_rate: 0.0001})
        callback((space ** n_iter).element([xi.eval() for xi in x_vals]))
        print(loss.eval())
