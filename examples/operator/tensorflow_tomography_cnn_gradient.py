"""Example of how to solve a simple tomography problem using tensorflow."""

import tensorflow as tf
import numpy as np
import odl


def random_ellipse():
    return (np.random.rand() - 0.3,
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            np.random.rand() - 0.5, np.random.rand() - 0.5,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc):
    n = np.random.poisson(10)
    ellipses = [random_ellipse() for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def var(x):
    return tf.Variable(tf.constant(x, dtype='float32'))


def create_variable(name, shape, stddev=0.01):
    variable = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
    return variable


with tf.Session() as sess:

    # Create ODL data structures
    size = 128
    space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                              dtype='float32')
    geometry = odl.tomo.parallel_beam_geometry(space)
    ray_transform = odl.tomo.RayTransform(space, geometry)


    # Create tensorflow layer from odl operator
    odl_op_layer = odl.as_tensorflow_layer(ray_transform,
                                           'RayTransform')
    odl_op_layer_adjoint = odl.as_tensorflow_layer(ray_transform.adjoint,
                                                   'RayTransformAdjoint')

    n_data = 50
    x_arr = np.empty((n_data, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((n_data, ray_transform.range.shape[0], ray_transform.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_data, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_data):
        if i == 0:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = random_phantom(space)
        data = ray_transform(phantom)
        noisy_data = data + odl.phantom.white_noise(ray_transform.range) * np.mean(data) * 0.1
        fbp = odl.tomo.fbp_op(ray_transform)(noisy_data)

        x_arr[i] = np.asarray(fbp)[..., None]
        x_true_arr[i] = np.asarray(phantom)[..., None]
        y_arr[i] = np.asarray(noisy_data)[..., None]

    x_0 = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_0")
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    y = tf.placeholder(tf.float32, shape=[None, ray_transform.range.shape[0], ray_transform.range.shape[1], 1], name="y")

    s = tf.fill([tf.shape(x_0)[0], size, size, 5], np.float32(0.0), name="s")

    # Create constant right hand side

    w1 = tf.Variable(tf.truncated_normal([3, 3, 7, 32], stddev=0.01))
    b1 = var(np.ones(32) * 0.1)

    w2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.01))
    b2 = var(np.ones(32) * 0.1)

    w3 = tf.Variable(tf.truncated_normal([3, 3, 32, 6], stddev=0.01))
    b3 = var(np.random.randn(6) * 0.001)

    n_iter = 5

    x = x_0
    loss = 0
    for i in range(n_iter):
        gradx = odl_op_layer_adjoint(odl_op_layer(x) - y)

        update = tf.concat([x, gradx, s], axis=3)

        update = tf.nn.relu(conv2d(update, w1) + b1)
        update = tf.nn.relu(conv2d(update, w2) + b2)
        update = tf.nn.dropout(update, 0.8)
        update = tf.nn.tanh(conv2d(update, w3) + b3)

        ds = update[..., 1:]
        dx = update[..., 0][..., None]

        s = s + ds
        x = x + dx

        loss = loss + tf.nn.l2_loss(x - x_true) * (2. ** (i - n_iter)) / n_data

    # Train using the adam optimizer
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Initialize all TF variables
    tf.global_variables_initializer().run()

    # Solve with an ODL callback to see what happens
    callback = odl.solvers.CallbackShow()

    for i in range(10000):
        _, loss_training = sess.run([optimizer, loss],
                                  feed_dict={learning_rate: 0.001,
                                             x_0: x_arr[1:],
                                             x_true: x_true_arr[1:],
                                             y: y_arr[1:]})

        x_value, loss_value = sess.run([x, loss],
                       feed_dict={x_0: x_arr[0:1],
                                  x_true: x_true_arr[0:1],
                                  y: y_arr[0:1]})

        print('iter={}, training loss={}, validation loss={}'.format(i, loss_training, loss_value))
        callback(space.element(x_value[0]))
