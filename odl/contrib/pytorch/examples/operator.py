"""Demonstration of basic ODL->pytorch integration functionality."""

import numpy as np
import torch
import odl
from odl.contrib.pytorch import TorchOperator

# --- Forward --- #

# Define ODL operator
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=float)
odl_op = odl.MatrixOperator(matrix)

# Wrap as torch operator
torch_op = TorchOperator(odl_op)

# Define evaluation point and wrap into a variable. Mark as
# `requires_gradient`, otherwise `backward()` doesn't do anything.
# This is supported by the ODL wrapper.
x = torch.DoubleTensor([1, 1, 1])
x_var = torch.autograd.Variable(x, requires_grad=True)

# Evaluate torch operator
res_var = torch_op(x_var)

# ODL result
odl_res = odl_op(x.numpy())

print('pytorch result: ', res_var.data.numpy())
print('ODL result    : ', odl_res.asarray())

# --- Gradient (backward) --- #

# Define ODL cost functional
odl_cost = odl.solvers.L2NormSquared(odl_op.range)

# Wrap with torch
torch_cost = TorchOperator(odl_cost)

# Compute forward pass
res_var = torch_cost(torch_op(x_var))

# Populate gradients by backwards pass
res_var.backward()

# ODL result
odl_grad = (odl_cost * odl_op).gradient(x.numpy())

print('pytorch gradient: ', x_var.grad.data.numpy())
print('ODL gradient    : ', odl_grad.asarray())

# --- Gradient in general spaces --- #

# Same steps as above

# ODL part
dom = odl.uniform_discr(0, 6, 3)  # weight 2.0
odl_op = odl.MatrixOperator(matrix, domain=dom)
odl_cost = odl.solvers.L2NormSquared(odl_op.range)
odl_functional = odl_cost * odl_op

# Torch part
torch_op = TorchOperator(odl_op)
torch_cost = TorchOperator(odl_cost)

x = torch.ones((3,)).type(torch.DoubleTensor)
x_var = torch.autograd.Variable(x, requires_grad=True)
y_var = torch_op(x_var)
res_var = torch_cost(y_var)
res_var.backward()

# These may be different, see https://github.com/odlgroup/odl/issues/1068
print('pytorch gradient (weighted): ', x_var.grad.data.numpy())
print('ODL gradient (weighted)    : ',
      odl_functional.gradient(x.numpy()).asarray())
