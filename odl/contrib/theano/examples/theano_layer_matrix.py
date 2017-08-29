"""Example of how to convert an ODL operator to a Theano operator layer.

In this example we take an ODL operator given by a `MatrixOperator` and
convert it into a theano operator that can be used inside any theano
computational graph.

We also demonstrate that we can compute the "gradients" properly using the
adjoint of the derivative.
"""

import theano
import theano.tensor as T
import numpy as np
import odl
import odl.contrib.theano

# Define ODL operator
matrix = np.array([[1., 2.],
                   [0., 0.],
                   [0., 1.]])
odl_op = odl.MatrixOperator(matrix)

# Define evaluation points
x = [1., 2.]
dy = [1., 2., 3.]

# Create theano placeholders
x_theano = T.fvector('x')
dy_theano = T.fvector('dy')

# Create theano layer from odl operator
odl_op_layer = odl.contrib.theano.TheanoOperator(odl_op)
y_theano = odl_op_layer(x_theano)
y_theano_func = theano.function([x_theano], y_theano)

# Evaluate using theano
print('Theano eval:                  ',
      y_theano_func(x))

# Compare result with pure ODL
print('ODL eval:                     ',
      odl_op(x))

# Compute adjoint of derivative "gradients"/"Lop" using theano.
# Note that the reuslt is indpendent of x, since the operator is linear.
# We need to explicitly tell theano to ignore this.
dy_theano_func = theano.function([x_theano, dy_theano],
                                 T.Lop(y_theano,
                                       x_theano,
                                       dy_theano),
                                 on_unused_input='ignore')

# Evaluate the adjoint of the derivative, called gradient in theano
print('Theano adjoint of derivative: ',
      dy_theano_func(x, dy))

# Compare result with pure ODL
print('ODL adjoint of derivative:    ',
      odl_op.derivative(x).adjoint(dy))
