# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the Chambolle-Pock solver."""

from __future__ import division
import numpy as np
import pytest

import odl
from odl.solvers import chambolle_pock_solver
from odl.util.testutils import all_almost_equal

# Places for the accepted error when comparing results
PLACES = 8

# Algorithm parameters
TAU = 0.3
SIGMA = 0.7
THETA = 0.9

# Test data array
DATA = np.arange(6)


def test_chambolle_pock_solver_simple_space():
    """Test for the Chambolle-Pock algorithm."""

    # Create a discretized image space
    space = odl.uniform_discr(0, 1, DATA.size)

    # Operator
    op = odl.IdentityOperator(space)

    # Starting point (image)
    discr_vec = op.domain.element(DATA)

    # Relaxation variable required to resume iteration
    discr_vec_relax = discr_vec.copy()

    # Dual variable required to resume iteration
    discr_dual = op.range.zero()

    # Functional, use the same functional for F^* and G
    g = odl.solvers.ZeroFunctional(space)
    f = g.convex_conj

    # Run the algorithm
    chambolle_pock_solver(discr_vec, f, g, op, tau=TAU, sigma=SIGMA,
                          theta=THETA, niter=1, callback=None,
                          x_relax=discr_vec_relax, y=discr_dual)

    # Explicit computation
    vec_expl = (1 - TAU * SIGMA) * DATA

    assert all_almost_equal(discr_vec, vec_expl, PLACES)

    # Explicit computation of the value of the relaxation variable
    vec_relax_expl = (1 + THETA) * vec_expl - THETA * DATA

    assert all_almost_equal(discr_vec_relax, vec_relax_expl, PLACES)

    # Resume iteration with previous x but without previous relaxation
    chambolle_pock_solver(discr_vec, f, g, op, tau=TAU, sigma=SIGMA,
                          theta=THETA, niter=1)

    vec_expl *= (1 - SIGMA * TAU)
    assert all_almost_equal(discr_vec, vec_expl, PLACES)

    # Resume iteration with x1 as above and with relaxation parameter
    discr_vec[:] = vec_expl
    chambolle_pock_solver(discr_vec, f, g, op, tau=TAU, sigma=SIGMA,
                          theta=THETA, niter=1, x_relax=discr_vec_relax,
                          y=discr_dual)

    vec_expl = vec_expl - TAU * SIGMA * (DATA + vec_relax_expl)
    assert all_almost_equal(discr_vec, vec_expl, PLACES)

    # Test acceleration parameter: use output argument for the relaxation
    # variable since otherwise two iterations are required for the
    # relaxation to take effect on the input variable
    # Acceleration parameter gamma=0 corresponds to relaxation parameter
    # theta=1 without acceleration

    # Relaxation parameter 1 and no acceleration
    discr_vec = op.domain.element(DATA)
    discr_vec_relax_no_gamma = op.domain.element(DATA)
    chambolle_pock_solver(discr_vec, f, g, op, tau=TAU, sigma=SIGMA,
                          theta=1, gamma=None, niter=1,
                          x_relax=discr_vec_relax_no_gamma)

    # Acceleration parameter 0, overwrites relaxation parameter
    discr_vec = op.domain.element(DATA)
    discr_vec_relax_g0 = op.domain.element(DATA)
    chambolle_pock_solver(discr_vec, f, g, op, tau=TAU, sigma=SIGMA,
                          theta=0, gamma=0, niter=1,
                          x_relax=discr_vec_relax_g0)

    assert discr_vec != discr_vec_relax_no_gamma
    assert all_almost_equal(discr_vec_relax_no_gamma, discr_vec_relax_g0)

    # Test callback execution
    chambolle_pock_solver(discr_vec, f, g, op, tau=TAU, sigma=SIGMA,
                          theta=THETA, niter=1,
                          callback=odl.solvers.CallbackPrintIteration())


def test_chambolle_pock_solver_produce_space():
    """Test the Chambolle-Pock algorithm using a product space operator."""

    # Create a discretized image space
    space = odl.uniform_discr(0, 1, DATA.size)

    # Operator
    identity = odl.IdentityOperator(space)

    # Create broadcasting operator
    prod_op = odl.BroadcastOperator(identity, -2 * identity)

    # Starting point for explicit computation
    discr_vec_0 = prod_op.domain.element(DATA)

    # Copy to be overwritten by the algorithm
    discr_vec = discr_vec_0.copy()

    # Proximal operator using the same factory function for F^* and G
    g = odl.solvers.ZeroFunctional(prod_op.domain)
    f = odl.solvers.ZeroFunctional(prod_op.range).convex_conj

    # Run the algorithm
    chambolle_pock_solver(discr_vec, f, g, prod_op, tau=TAU, sigma=SIGMA,
                          theta=THETA, niter=1)

    vec_expl = discr_vec_0 - TAU * SIGMA * prod_op.adjoint(
        prod_op(discr_vec_0))
    assert all_almost_equal(discr_vec, vec_expl, PLACES)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
