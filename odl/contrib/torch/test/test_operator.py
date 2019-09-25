# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the ODL-PyTorch integration."""

import numpy as np
import torch
from torch import nn

import odl
from odl.contrib import torch as odl_torch
from odl.util.testutils import all_almost_equal, simple_fixture


dtype = simple_fixture('dtype', ['float32', 'float64'])
device_params = ['cpu']
if torch.cuda.is_available():
    device_params.append('cuda')
device = simple_fixture('device', device_params)
shape = simple_fixture('shape', [(3,), (2, 3), (2, 2, 3)])


def test_autograd_function_forward(dtype, device):
    """Test forward evaluation with operators as autograd functions."""
    # Define ODL operator
    matrix = np.random.rand(2, 3).astype(dtype)
    odl_op = odl.MatrixOperator(matrix)

    # Compute forward pass with both ODL and PyTorch
    x_arr = np.ones(3, dtype=dtype)
    x = torch.from_numpy(x_arr).to(device)
    res = odl_torch.OperatorFunction.apply(odl_op, x)
    res_arr = res.detach().cpu().numpy()
    odl_res = odl_op(x_arr)

    assert res_arr.dtype == dtype
    assert all_almost_equal(res_arr, odl_res)
    assert x.device.type == res.device.type == device


def test_autograd_function_backward(dtype, device):
    """Test backprop with operators/functionals as autograd functions."""
    # Define ODL operator and cost functional
    matrix = np.random.rand(2, 3).astype(dtype)
    odl_op = odl.MatrixOperator(matrix)
    odl_cost = odl.solvers.L2NormSquared(odl_op.range)
    odl_functional = odl_cost * odl_op

    # Define evaluation point and mark as `requires_grad` to enable
    # backpropagation
    x_arr = np.ones(3, dtype=dtype)
    x = torch.from_numpy(x_arr).to(device)
    x.requires_grad_(True)

    # Compute forward pass
    y = odl_torch.OperatorFunction.apply(odl_op, x)
    res = odl_torch.OperatorFunction.apply(odl_cost, y)

    # Populate gradients by backwards pass
    res.backward()
    grad = x.grad
    grad_arr = grad.detach().cpu().numpy()

    # Compute gradient with ODL
    odl_grad = odl_functional.gradient(x_arr)

    assert grad_arr.dtype == dtype
    assert all_almost_equal(grad_arr, odl_grad)
    assert x.device.type == grad.device.type == device


def test_module_forward(shape, device):
    """Test forward evaluation with operators as modules."""
    # Define ODL operator and wrap as module
    ndim = len(shape)
    space = odl.uniform_discr([0] * ndim, shape, shape, dtype='float32')
    odl_op = odl.ScalingOperator(space, 2)
    op_mod = odl_torch.OperatorModule(odl_op)

    # Input data
    x_arr = np.ones(shape, dtype='float32')

    # Test with 1 extra dim (minimum)
    x = torch.from_numpy(x_arr).to(device)[None, ...]
    x.requires_grad_(True)
    res = op_mod(x)
    res_arr = res.detach().cpu().numpy()
    assert res_arr.shape == (1,) + odl_op.range.shape
    assert all_almost_equal(
        res_arr, np.asarray(odl_op(x_arr))[None, ...]
    )
    assert x.device.type == res.device.type == device

    # Test with 2 extra dims
    x = torch.from_numpy(x_arr).to(device)[None, None, ...]
    x.requires_grad_(True)
    res = op_mod(x)
    res_arr = res.detach().cpu().numpy()
    assert res_arr.shape == (1, 1) + odl_op.range.shape
    assert all_almost_equal(
        res_arr, np.asarray(odl_op(x_arr))[None, None, ...]
    )
    assert x.device.type == res.device.type == device


def test_module_forward_diff_shapes(device):
    """Test operator module with different shapes of input and output."""
    # Define ODL operator and wrap as module
    matrix = np.random.rand(2, 3).astype('float32')
    odl_op = odl.MatrixOperator(matrix)
    op_mod = odl_torch.OperatorModule(odl_op)

    # Input data
    x_arr = np.ones(3, dtype='float32')

    # Test with 1 extra dim (minimum)
    x = torch.from_numpy(x_arr).to(device)[None, ...]
    x.requires_grad_(True)
    res = op_mod(x)
    res_arr = res.detach().cpu().numpy()
    assert res_arr.shape == (1,) + odl_op.range.shape
    assert all_almost_equal(
        res_arr, np.asarray(odl_op(x_arr))[None, ...]
    )
    assert x.device.type == res.device.type == device

    # Test with 2 extra dims
    x = torch.from_numpy(x_arr).to(device)[None, None, ...]
    x.requires_grad_(True)
    res = op_mod(x)
    res_arr = res.detach().cpu().numpy()
    assert res_arr.shape == (1, 1) + odl_op.range.shape
    assert all_almost_equal(
        res_arr, np.asarray(odl_op(x_arr))[None, None, ...]
    )
    assert x.device.type == res.device.type == device


def test_module_backward(device):
    """Test backpropagation with operators as modules."""
    # Define ODL operator and wrap as module
    matrix = np.random.rand(2, 3).astype('float32')
    odl_op = odl.MatrixOperator(matrix)
    op_mod = odl_torch.OperatorModule(odl_op)
    loss_fn = nn.MSELoss()

    # Test with linear layers (1 extra dim)
    layer_before = nn.Linear(3, 3)
    layer_after = nn.Linear(2, 2)
    model = nn.Sequential(layer_before, op_mod, layer_after).to(device)
    x = torch.from_numpy(
        np.ones(3, dtype='float32')
    )[None, ...].to(device)
    x.requires_grad_(True)
    target = torch.from_numpy(
        np.zeros(2, dtype='float32')
    )[None, ...].to(device)
    loss = loss_fn(model(x), target)
    loss.backward()
    assert all(p is not None for p in model.parameters())
    assert x.grad.detach().cpu().abs().sum() != 0
    assert x.device.type == loss.device.type == device

    # Test with conv layers (2 extra dims)
    layer_before = nn.Conv1d(1, 2, 2)  # 1->2 channels
    layer_after = nn.Conv1d(2, 1, 2)  # 2->1 channels
    model = nn.Sequential(layer_before, op_mod, layer_after).to(device)
    # Input size 4 since initial convolution reduces by 1
    x = torch.from_numpy(
        np.ones(4, dtype='float32')
    )[None, None, ...].to(device)
    x.requires_grad_(True)
    # Output size 1 since final convolution reduces by 1
    target = torch.from_numpy(
        np.zeros(1, dtype='float32')
    )[None, None, ...].to(device)

    loss = loss_fn(model(x), target)
    loss.backward()
    assert all(p is not None for p in model.parameters())
    assert x.grad.detach().cpu().abs().sum() != 0
    assert x.device.type == loss.device.type == device


if __name__ == '__main__':
    odl.util.test_file(__file__)
