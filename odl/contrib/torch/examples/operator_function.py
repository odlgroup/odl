"""Demonstration of ODL operators as pytorch autograd functions.

This example shows how to use an ODL ``Operator`` with the PyTorch
autograd machinery that supports forward evaluation on ``Tensor``
objects and backpropagation using automatic differentiation.
"""

from __future__ import print_function

import numpy as np
import torch

import odl
from odl.contrib.torch import OperatorFunction

# --- Forward --- #

# Define ODL operator
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=float)
op = odl.MatrixOperator(matrix)

# Define evaluation point and wrap into a variable. Mark as
# `requires_gradient`, otherwise `backward()` doesn't do anything.
x = torch.ones(3, dtype=torch.float64, requires_grad=True)

# Evalueate using ODL
odl_res = op(x.detach().numpy())

# Evaluate using torch function
torch_res = OperatorFunction.apply(op, x)

print('PyTorch result: ', torch_res.detach().numpy())
print('ODL result    : ', np.asarray(odl_res))

# --- Gradient (backward) --- #

# Define ODL loss functional
l2sq = odl.solvers.L2NormSquared(op.range)

# Compute forward pass
z = OperatorFunction.apply(op, x)
loss = OperatorFunction.apply(l2sq, z)

# Populate gradients by backwards pass
loss.backward()

# Same operations using ODL
odl_grad = (l2sq * op).gradient(x.detach().numpy())

print('PyTorch gradient: ', x.grad.detach().numpy())
print('ODL gradient    : ', np.asarray(odl_grad))

# --- Gradients for input batches --- #

# This time without operator
l2sq = odl.solvers.L2NormSquared(odl.rn(3))


# To define a loss, we need to handle two arguments and the final
# reduction over the batch axis
def mse(x, y):
    return OperatorFunction.apply(l2sq, x - y).mean()


x = torch.ones((2, 1, 3), dtype=torch.float64, requires_grad=True)
y = -torch.ones((2, 1, 3), dtype=torch.float64)

loss = mse(x, y)
loss.backward()

print('Multiple gradients:')
print(x.grad.detach().numpy())
