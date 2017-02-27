"""Example of how to convert an ODL ray transform to a tensorflow layer."""

import tensorflow as tf
import numpy as np
import odl


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    space = odl.uniform_discr([-64, -64], [64, 64], [128, 128],
                              dtype='float32')
    geometry = odl.tomo.parallel_beam_geometry(space)
    ray_transform = odl.tomo.RayTransform(space, geometry)

    x = tf.constant(np.asarray(ray_transform.domain.one()))
    z = tf.constant(np.asarray(ray_transform.range.one()))

    odl_op_layer = odl.as_tensorflow_layer(ray_transform, 'RayTransform')
    y = odl_op_layer(x)

    # Evaluate using tensorflow
    print(y.eval())

    # Compare result with pure ODL
    print(ray_transform(x.eval()))

    # Evaluate the adjoint of the derivative, called gradient in tensorflow
    print(tf.gradients(y, [x], z)[0].eval())

    # Compare result with pure ODL
    print(ray_transform.derivative(x.eval()).adjoint(z.eval()))
