# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Theano."""

from __future__ import division
import pytest
import numpy as np
import theano
import theano.tensor as T

import odl
import odl.contrib.theano
from odl.util import all_almost_equal


def test_theano_operator():
    """Test the ODL->Theano operator wrapper."""
    # Define ODL operator
    matrix = np.random.rand(3, 2)
    odl_op = odl.MatrixOperator(matrix)

    # Define evaluation points
    x = [1., 2.]
    dy = [1., 2., 3.]

    # Create Theano placeholders
    x_theano = T.dvector()
    dy_theano = T.dvector()

    # Create Theano layer from odl operator
    odl_op_layer = odl.contrib.theano.TheanoOperator(odl_op)

    # Build computation graphs
    y_theano = odl_op_layer(x_theano)
    y_theano_func = theano.function([x_theano], y_theano)
    dy_theano_func = theano.function([x_theano, dy_theano],
                                     T.Rop(y_theano, x_theano, dy_theano))

    # Evaluate using Theano
    result = y_theano_func(x)
    expected = odl_op(x)

    assert all_almost_equal(result, expected)

    # Evaluate the adjoint of the derivative, called gradient in Theano
    result = dy_theano_func(x, dy)
    expected = odl_op.derivative(x).adjoint(dy)

    assert all_almost_equal(result, expected)


def test_theano_gradient():
    """Test the gradient of ODL functionals wrapped as Theano Ops."""
    # Define ODL operator
    matrix = np.random.rand(3, 2)
    odl_op = odl.MatrixOperator(matrix)

    # Define evaluation point
    x = [1., 2.]

    # Define ODL cost and the composed functional
    odl_cost = odl.solvers.L2NormSquared(odl_op.range)
    odl_functional = odl_cost * odl_op

    # Create Theano placeholder
    x_theano = T.dvector()

    # Create Theano layers from odl operators
    odl_op_layer = odl.contrib.theano.TheanoOperator(odl_op)
    odl_cost_layer = odl.contrib.theano.TheanoOperator(odl_cost)

    # Build computation graph
    y_theano = odl_op_layer(x_theano)
    cost_theano = odl_cost_layer(y_theano)
    cost_theano_func = theano.function([x_theano], cost_theano)
    cost_grad_theano = T.grad(cost_theano, x_theano)
    cost_grad_theano_func = theano.function([x_theano], cost_grad_theano)

    # Evaluate using Theano
    result = cost_theano_func(x)
    expected = odl_functional(x)
    assert result == pytest.approx(expected)

    # Evaluate the gradient of the cost, should be 2 * matrix^T.dot(x)
    result = cost_grad_theano_func(x)
    expected = odl_functional.gradient(x)
    assert all_almost_equal(result, expected)


if __name__ == '__main__':
    odl.util.test_file(__file__)
