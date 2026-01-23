"""Demonstration of ODL operators as pytorch modules.

This example shows how to wrap an ODL ``Operator`` as a
``torch.nn.Module`` that can be used as a layer in a Neural Network.
It supports backpropagation as well as inputs that contain extra dimensions
for, e.g., batches and channels.
"""

import numpy as np
import torch
from torch import nn

import odl
from odl.contrib import torch as odl_torch

# --- Forward --- #


# Define ODL operator
matrix = np.array([[1, 0, 0],
                   [0, 1, 1]], dtype='float32')
odl_op = odl.MatrixOperator(matrix)

# Wrap ODL operator as `Module`
op_layer = odl_torch.OperatorModule(odl_op)

# Test with some inputs. We need to add at least one batch axis
inp = torch.ones((1, 3))
print('Operator layer evaluated on a 1x3 tensor:')
print(op_layer(inp))
print()

inp = torch.ones((1, 1, 3))
print('Operator layer evaluated on a 1x1x3 tensor:')
print(op_layer(inp))
print()

# We combine the module with some builtin pytorch layers of matching shapes
layer_before = nn.Linear(3, 3)
layer_after = nn.Linear(2, 2)
model1 = nn.Sequential(layer_before, op_layer, layer_after)

print('Composed model 1:')
print(model1)
print()

inp = torch.ones((1, 3))
print('Model 1 evaluated on a 1x3 tensor:')
print(model1(inp))

# We can also use convolutional layers with extra channel axes. Since
# convolutions without padding reduce the size of the input by
# `kernel_size - 1`, the input has to have size 4 here, and the output
# will have size 1.
layer_before = nn.Conv1d(1, 2, 2)
layer_after = nn.Conv1d(2, 1, 2)
model2 = nn.Sequential(layer_before, op_layer, layer_after)

print('Composed model 2:')
print(model2)
print()

# Add extra batch and channel axes
inp = torch.ones((1, 1, 4))
print('Model 2 evaluated on a 1x1x4 tensor:')
print(model2(inp))


# --- Backward --- #


# Define a loss function and targets to compare against
loss_func = nn.MSELoss()
model = model1
inp = torch.ones((1, 3), requires_grad=True)
target = torch.zeros((1, 2))

# Compute the loss and backpropagate
loss = loss_func(model(inp), target)
print('Loss function value:', loss.item())
print()

loss.backward()
print('All parameter gradients should be populated now:')
for p in model.named_parameters():
    name, value = p
    print('{}:'.format(name))
    print(value)
    print('gradient')
    print(value.grad)
