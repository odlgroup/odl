# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the Primal-Dual Hybrid Gradient algorithm."""

from __future__ import division

import numpy as np

import odl
from odl.solvers import pdhg
from odl.util.testutils import all_almost_equal

NDIGITS = 8

# Algorithm parameters
TAU = 0.3
SIGMA = 0.7
THETA = 0.9

# Test data array
DATA = np.arange(6)


def test_pdhg_simple_space():
    """Test for the Primal-Dual Hybrid Gradient algorithm."""
    space = odl.uniform_discr(0, 1, DATA.size)
    op = odl.IdentityOperator(space)
    x = space.element(DATA)

    # Relaxation and dual variables required to resume iteration
    x_relax = space.copy(x)
    y = op.range.zero()

    # Use the same functional for f^* and g
    f = odl.solvers.ZeroFunctional(space)
    g = f.convex_conj

    # Run one iteration of the algorithm and compare against explicit
    # calculation
    pdhg(x, f, g, op, niter=1, tau=TAU, sigma=SIGMA, theta=THETA,
         x_relax=x_relax, y=y)
    x_expected = (1 - TAU * SIGMA) * DATA
    assert all_almost_equal(x, x_expected, NDIGITS)
    x_relax_expected = (1 + THETA) * x_expected - THETA * DATA
    assert all_almost_equal(x_relax, x_relax_expected, NDIGITS)

    # Resume iteration with previous x but without previous relaxation
    pdhg(x, f, g, op, niter=1, tau=TAU, sigma=SIGMA, theta=THETA)
    x_expected *= (1 - SIGMA * TAU)
    assert all_almost_equal(x, x_expected, NDIGITS)

    # Resume iteration with x1 as above and with relaxation parameter
    x[:] = x_expected
    pdhg(x, f, g, op, niter=1, tau=TAU, sigma=SIGMA, theta=THETA,
         x_relax=x_relax, y=y)

    x_expected = x_expected - TAU * SIGMA * (DATA + x_relax_expected)
    assert all_almost_equal(x, x_expected, NDIGITS)

    # Test acceleration parameter: use output argument for the relaxation
    # variable since otherwise two iterations are required for the
    # relaxation to take effect on the input variable
    # Acceleration parameter gamma=0 corresponds to relaxation parameter
    # theta=1 without acceleration

    # Relaxation parameter 1 and no acceleration
    x = op.domain.element(DATA)
    x_relax_no_gamma = op.domain.element(DATA)
    pdhg(x, f, g, op, niter=1, tau=TAU, sigma=SIGMA, theta=1,
         gamma_primal=None, x_relax=x_relax_no_gamma)

    # Acceleration parameter 0, overwrites relaxation parameter
    x = op.domain.element(DATA)
    x_relax_g0 = op.domain.element(DATA)
    pdhg(x, f, g, op, niter=1, tau=TAU, sigma=SIGMA, theta=0,
         gamma_primal=0, x_relax=x_relax_g0)

    assert not all_almost_equal(x, x_relax_no_gamma)
    assert all_almost_equal(x_relax_no_gamma, x_relax_g0)

    # Test callback execution
    pdhg(x, f, g, op, niter=1, tau=TAU, sigma=SIGMA, theta=THETA,
         callback=odl.solvers.CallbackPrintIteration())


def test_pdhg_product_space():
    """Test the PDHG algorithm using a product space operator."""
    space = odl.uniform_discr(0, 1, DATA.size)
    I = odl.IdentityOperator(space)
    op = odl.BroadcastOperator(I, -2 * I)

    # Starting point for explicit computation
    x_0 = space.element(DATA)

    # Copy to be overwritten by the algorithm
    x = space.copy(DATA)

    # Using f and g such that f = g^*
    f = odl.solvers.ZeroFunctional(op.domain)
    g = odl.solvers.ZeroFunctional(op.range).convex_conj

    pdhg(x, f, g, op, niter=1, tau=TAU, sigma=SIGMA, theta=THETA)

    x_expected = x_0 - TAU * SIGMA * op.adjoint(op(x_0))
    assert all_almost_equal(x, x_expected, NDIGITS)


if __name__ == '__main__':
    odl.util.test_file(__file__)
