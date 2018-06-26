# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the ODL-pytorch integration."""

import numpy as np
import torch
from torch import autograd, nn

import odl
from odl.contrib import torch as odl_torch
from odl.util.testutils import all_almost_equal, simple_fixture


dtype = simple_fixture('dtype', ['float32', 'float64'])
use_cuda_params = [False]
if torch.cuda.is_available():
    use_cuda_params.append(True)
use_cuda = simple_fixture('use_cuda', use_cuda_params)
shape = simple_fixture('shape', [(3,), (2, 3), (2, 2, 3)])


def test_autograd_function_forward(dtype, use_cuda):
    """Test forward evaluation with operators as autograd functions."""
    # Define ODL operator
    matrix = np.random.rand(2, 3).astype(dtype)
    odl_op = odl.MatrixOperator(matrix)

    # Wrap as torch autograd function
    torch_op = odl_torch.OperatorAsAutogradFunction(odl_op)

    # Define evaluation point and wrap into a variable
    x = torch.from_numpy(np.ones(3, dtype=dtype))
    if use_cuda:
        x = x.cuda()
    x_var = autograd.Variable(x)

    # Evaluate torch operator
    res_var = torch_op(x_var)

    # ODL result
    odl_res = odl_op(x.cpu().numpy())

    assert res_var.data.cpu().numpy().dtype == dtype
    assert all_almost_equal(res_var.data.cpu().numpy(), odl_res)

    # Make sure data stays on the GPU
    if use_cuda:
        assert res_var.is_cuda


def test_autograd_function_backward(dtype, use_cuda):
    """Test backprop with operators/functionals as autograd functions."""
    # Define ODL operator and cost functional
    matrix = np.random.rand(2, 3).astype(dtype)
    odl_op = odl.MatrixOperator(matrix)
    odl_cost = odl.solvers.L2NormSquared(odl_op.range)
    odl_functional = odl_cost * odl_op

    # Wrap operator and cost with pytorch
    torch_op = odl_torch.OperatorAsAutogradFunction(odl_op)
    torch_cost = odl_torch.OperatorAsAutogradFunction(odl_cost)

    # Define evaluation point and wrap into a variable. Mark as
    # `requires_gradient`, otherwise `backward()` doesn't do anything.
    # This is supported by the ODL wrapper.
    x = torch.from_numpy(np.ones(3, dtype=dtype))
    if use_cuda:
        x = x.cuda()
    x_var = autograd.Variable(x, requires_grad=True)

    # Compute forward pass
    y_var = torch_op(x_var)
    res_var = torch_cost(y_var)

    # Populate gradients by backwards pass
    res_var.backward()
    torch_grad = x_var.grad

    # ODL result
    odl_grad = odl_functional.gradient(x.cpu().numpy())

    assert torch_grad.data.cpu().numpy().dtype == dtype
    assert all_almost_equal(torch_grad.data.cpu().numpy(), odl_grad)

    # Make sure data stays on the GPU
    if use_cuda:
        assert torch_grad.is_cuda


def test_module_forward(shape, use_cuda):
    """Test forward evaluation with operators as modules."""
    ndim = len(shape)
    space = odl.uniform_discr([0] * ndim, shape, shape)
    odl_op = odl.ScalingOperator(space, 2)
    op_mod = odl_torch.OperatorAsModule(odl_op)

    x = torch.from_numpy(np.ones(shape))
    if use_cuda:
        x = x.cuda()

    # Test with 1 extra dim (minimum)
    x_var = autograd.Variable(x, requires_grad=True)[None, ...]
    y_var = op_mod(x_var)

    assert y_var.data.shape == (1,) + odl_op.range.shape
    assert all_almost_equal(y_var.data.cpu().numpy(),
                            2 * np.ones((1,) + shape))

    # Test with 2 extra dims
    x_var = autograd.Variable(x, requires_grad=True)[None, None, ...]
    y_var = op_mod(x_var)

    assert y_var.data.shape == (1, 1) + odl_op.range.shape
    assert all_almost_equal(y_var.data.cpu().numpy(),
                            2 * np.ones((1, 1) + shape))

    # Make sure data stays on the GPU
    if use_cuda:
        assert y_var.is_cuda


def test_module_forward_diff_shapes(use_cuda):
    """Test operator module with different shapes of input and output."""
    matrix = np.random.rand(2, 3)
    odl_op = odl.MatrixOperator(matrix)
    op_mod = odl_torch.OperatorAsModule(odl_op)

    x = torch.from_numpy(np.ones(3))
    if use_cuda:
        x = x.cuda()

    # Test with 1 extra dim (minimum)
    x_var = autograd.Variable(x, requires_grad=True)[None, ...]
    y_var = op_mod(x_var)
    assert y_var.data.shape == (1,) + odl_op.range.shape
    assert all_almost_equal(y_var.data.cpu().numpy(),
                            odl_op(np.ones(3)).asarray().reshape((1, 2)))

    # Test with 2 extra dims
    x_var = autograd.Variable(x, requires_grad=True)[None, None, ...]
    y_var = op_mod(x_var)
    assert y_var.data.shape == (1, 1) + odl_op.range.shape
    assert all_almost_equal(y_var.data.cpu().numpy(),
                            odl_op(np.ones(3)).asarray().reshape((1, 1, 2)))


def test_module_backward(use_cuda):
    """Test backpropagation with operators as modules."""
    matrix = np.random.rand(2, 3).astype('float32')
    odl_op = odl.MatrixOperator(matrix)
    op_mod = odl_torch.OperatorAsModule(odl_op)
    loss_fun = nn.MSELoss()

    # Test with linear layers (1 extra dim)
    layer_before = nn.Linear(3, 3)
    layer_after = nn.Linear(2, 2)
    model = nn.Sequential(layer_before, op_mod, layer_after)
    x = torch.from_numpy(np.ones(3, dtype='float32'))
    target = torch.from_numpy(np.zeros(2, dtype='float32'))

    if use_cuda:
        x = x.cuda()
        target = target.cuda()
        model = model.cuda()

    x_var = autograd.Variable(x, requires_grad=True)[None, ...]
    target_var = autograd.Variable(target)[None, ...]

    loss = loss_fun(model(x_var), target_var)
    loss.backward()
    assert all(p is not None for p in model.parameters())

    # Test with conv layers (2 extra dims)
    layer_before = nn.Conv1d(1, 2, 2)  # 1->2 channels
    layer_after = nn.Conv1d(2, 1, 2)  # 2->1 channels
    model = nn.Sequential(layer_before, op_mod, layer_after)
    # Input size 4 since initial convolution reduces by 1
    x = torch.from_numpy(np.ones(4, dtype='float32'))
    # Output size 1 since final convolution reduces by 1
    target = torch.from_numpy(np.zeros(1, dtype='float32'))

    if use_cuda:
        x = x.cuda()
        target = target.cuda()
        model = model.cuda()

    x_var = autograd.Variable(x, requires_grad=True)[None, None, ...]
    target_var = autograd.Variable(target)[None, None, ...]

    loss = loss_fun(model(x_var), target_var)
    loss.backward()
    assert all(p is not None for p in model.parameters())

    # Make sure data stays on the GPU
    if use_cuda:
        assert x_var.is_cuda


if __name__ == '__main__':
    odl.util.test_file(__file__)
