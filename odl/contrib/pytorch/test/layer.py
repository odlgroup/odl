# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the ODL-pytorch integration."""

import numpy as np
import pytest
import torch

import odl
from odl.contrib.pytorch import TorchOperator
from odl.util.testutils import all_almost_equal, simple_fixture


dtype = simple_fixture('dtype', ['float32', 'float64'])


def test_forward(dtype):
    """Test forward evaluation with pytorch-wrapped operators."""
    # Define ODL operator
    matrix = np.random.rand(2, 3).astype(dtype)
    odl_op = odl.MatrixOperator(matrix)

    # Wrap as torch operator
    torch_op = TorchOperator(odl_op)

    # Define evaluation point and wrap into a variable
    x = torch.from_numpy(np.ones(3, dtype=dtype))
    x_var = torch.autograd.Variable(x)

    # Evaluate torch operator
    res_var = torch_op(x_var)

    # ODL result
    odl_res = odl_op(x.numpy())

    assert res_var.data.numpy().dtype == dtype
    assert all_almost_equal(res_var.data.numpy(), odl_res)


def test_backward(dtype):
    """Test gradient evaluation with pytorch-wrapped operators/functionals."""
    # Define ODL operator and cost functional
    matrix = np.random.rand(2, 3).astype(dtype)
    odl_op = odl.MatrixOperator(matrix)
    odl_cost = odl.solvers.L2NormSquared(odl_op.range)
    odl_functional = odl_cost * odl_op

    # Wrap operator and cost with pytorch
    torch_op = TorchOperator(odl_op)
    torch_cost = TorchOperator(odl_cost)

    # Define evaluation point and wrap into a variable. Mark as
    # `requires_gradient`, otherwise `backward()` doesn't do anything.
    # This is supported by the ODL wrapper.
    x = torch.from_numpy(np.ones(3, dtype=dtype))
    x_var = torch.autograd.Variable(x, requires_grad=True)

    # Compute forward pass
    y_var = torch_op(x_var)
    res_var = torch_cost(y_var)

    # Populate gradients by backwards pass
    res_var.backward()
    torch_grad = x_var.grad

    # ODL result
    odl_grad = odl_functional.gradient(x.numpy())

    assert torch_grad.data.numpy().dtype == dtype
    assert all_almost_equal(torch_grad.data.numpy(), odl_grad)


if __name__ == '__main__':
    odl.util.test_file(__file__)
