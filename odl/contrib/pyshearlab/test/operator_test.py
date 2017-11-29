# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the ODL-pyshearlab integration."""

import pytest
import numpy as np
import odl
import odl.contrib.pyshearlab
from odl.util.testutils import all_almost_equal, simple_fixture


dtype = simple_fixture('dtype', ['float32', 'float64'])
shape = simple_fixture('shape', [(64, 64), (128, 128)])


def test_operator(dtype, shape):
    """Test PyShearlabOperator operator."""
    rel = 100 * np.finfo('float32').resolution

    space = odl.uniform_discr([-1, -1], [1, 1], shape, dtype=dtype)

    op = odl.contrib.pyshearlab.PyShearlabOperator(space, num_scales=2)

    phantom = odl.phantom.shepp_logan(space, True)

    # Test evaluation
    y = op(phantom)

    # <Ax, Ax> = <x, AtAx>
    ax = op.adjoint(y)
    assert pytest.approx(y.inner(y), rel=rel) == phantom.inner(ax)

    # A^{-1} A x = x
    rec = op.inverse(y)
    assert all_almost_equal(op.inverse(y), phantom)

    # <A^{-1}y, A^{-1}y> = <y, A^{-*}A^{-1}y>
    recadj = op.inverse.adjoint(rec)
    assert pytest.approx(rec.inner(rec), rel=rel) == y.inner(recadj)

    # A^{-*}A^*y = y
    adjinvadj = op.adjoint.inverse(op.adjoint(y))
    assert all_almost_equal(adjinvadj, y, places=5)

    # A^*A^{-*}x = x
    adjadjinv = op.adjoint(op.adjoint.inverse(phantom))
    assert all_almost_equal(adjadjinv, phantom, places=5)


if __name__ == '__main__':
    odl.util.test_file(__file__)
