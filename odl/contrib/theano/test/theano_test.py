# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for theano."""

from __future__ import division
import pytest
import numpy as np
import theano
import theano.tensor as T

import odl
import odl.contrib.theano
from odl.util import all_almost_equal


def test_theano_operator():
    # Define ODL operator
    matrix = np.random.rand(3, 2)
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
    dy_theano_func = theano.function([x_theano, dy_theano],
                                     T.Lop(y_theano,
                                           x_theano,
                                           dy_theano),
                                     on_unused_input='ignore')

    # Evaluate using theano
    result = y_theano_func(x)
    expected = odl_op(x)

    assert all_almost_equal(result, expected)

    # Evaluate the adjoint of the derivative, called gradient in theano
    result = dy_theano_func(x, dy)
    expected = odl_op.derivative(x).adjoint(dy)

    assert all_almost_equal(result, expected)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
