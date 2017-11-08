"""Example of how to convert a tensorflow layer to an ODL operator.

In this example we take a tensorflow layer (given by matrix multiplication)
and convert it into an ODL operator.
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow

sess = tf.InteractiveSession()

matrix = np.array([[1, 2],
                   [0, 0],
                   [0, 1]], dtype='float32')

# Define ODL operator
odl_op_pure = odl.MatrixOperator(matrix)

# Define ODL operator using tensorflow
input_tensor = tf.placeholder(tf.float32, shape=[2])
output_tensor = tf.reshape(tf.matmul(matrix,
                                     tf.expand_dims(input_tensor, 1)),
                           [3])

odl_op_tensorflow = odl.contrib.tensorflow.TensorflowOperator(input_tensor,
                                                              output_tensor)

# Define evaluation points
x = [1., 2.]
dx = [3., 4.]
dy = [1., 2., 3.]

# Evaluate
print('Tensorflow eval:                  ',
      odl_op_tensorflow(x))
print('ODL eval:                         ',
      odl_op_pure(x))

# Evaluate the derivative
print('Tensorflow derivative:            ',
      odl_op_tensorflow.derivative(x)(dx))
print('ODL derivative:                   ',
      odl_op_pure.derivative(x)(dx))

# Evaluate the adjoint of the derivative
print('Tensorflow adjoint of derivative: ',
      odl_op_tensorflow.derivative(x).adjoint(dy))
print('ODL adjoint of derivative:        ',
      odl_op_pure.derivative(x).adjoint(dy))
