# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for tensorflow."""

from __future__ import division
from itertools import permutations
import pytest
import numpy as np
import tensorflow as tf

import odl
import odl.contrib.tensorflow
from odl.util import all_almost_equal


def test_as_tensorflow_layer():
    # Define ODL operator
    matrix = np.random.rand(3, 2)
    odl_op = odl.MatrixOperator(matrix)

    # Define evaluation points
    x = np.random.rand(2)
    z = np.random.rand(3)

    # Add empty axes for batch and channel
    x_tf = tf.constant(x)[None, ..., None]
    z_tf = tf.constant(z)[None, ..., None]

    # Create tensorflow layer from odl operator
    odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(
        odl_op, 'MatrixOperator')
    y_tf = odl_op_layer(x_tf)

    # Evaluate using tensorflow
    result = y_tf.eval().ravel()
    expected = odl_op(x)

    assert all_almost_equal(result, expected)

    # Evaluate the adjoint of the derivative, called gradient in tensorflow
    result = tf.gradients(y_tf, [x_tf], z_tf)[0].eval().ravel()
    expected = odl_op.derivative(x).adjoint(z)

    assert all_almost_equal(result, expected)


if __name__ == '__main__':
    with tf.Session():
        odl.util.test_file(__file__)
