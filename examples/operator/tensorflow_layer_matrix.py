"""Example of how to convert an ODL operator to a tensorflow layer."""

import tensorflow as tf
import numpy as np
import odl


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    matrix = np.array([[1, 2],
                       [0, 0],
                       [0, 1]], dtype='float32')
    odl_op = odl.MatrixOperator(matrix)

    x = tf.constant([1., 2.])
    z = tf.constant([1., 2., 3.])

    odl_op_layer = odl.as_tensorflow_layer(odl_op, 'MatrixOperator')
    y = odl_op_layer(x)

    # Evaluate using tensorflow
    print(y.eval())

    # Compare result with pure ODL
    print(odl_op(x.eval()))

    # Evaluate the adjoint of the derivative, called gradient in tensorflow
    print(tf.gradients(y, [x], z)[0].eval())

    # Compare result with pure ODL
    print(odl_op.derivative(x.eval()).adjoint(z.eval()))
