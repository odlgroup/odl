# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for the ODL-pyshearlab integration."""

import pytest

import odl
import odl.contrib.pyshearlab
from odl.util.testutils import all_almost_equal, simple_fixture


dtype = simple_fixture('dtype', ['float32', 'float64'])


def test_operator(dtype):
    """Test PyShearlabOperator operator."""

    space = odl.uniform_discr([-1, -1], [1, 1], [128, 128], dtype=dtype)
    
    op = odl.contrib.pyshearlab.PyShearlabOperator(space, scales=2)
    
    phantom = odl.phantom.shepp_logan(space, True)
    
    y = op(phantom)
    
    assert all_almost_equal(op.inverse(y), phantom)
    
    # Compute <Ax, Ax> = <x, AtAx>
    ax = op.adjoint(y)
    assert pytest.approx(y.inner(y)) == phantom.inner(ax)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
