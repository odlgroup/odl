# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the Douglas-Rachford solver."""

from __future__ import division

import pytest

import odl
from odl.solvers import douglas_rachford_pd
from odl.util.testutils import all_almost_equal, noise_element

# Places for the accepted error when comparing results
HIGH_ACCURACY = 8
LOW_ACCURACY = 4


def test_primal_dual_input_handling():
    """Test to see that input is handled correctly."""

    space1 = odl.uniform_discr(0, 1, 10)

    lin_ops = [odl.ZeroOperator(space1), odl.ZeroOperator(space1)]
    g = [odl.solvers.ZeroFunctional(space1),
         odl.solvers.ZeroFunctional(space1)]
    f = odl.solvers.ZeroFunctional(space1)

    # Check that the algorithm runs. With the above operators, the algorithm
    # returns the input.
    x0 = noise_element(space1)
    x = x0.copy()
    niter = 3

    douglas_rachford_pd(x, f, g, lin_ops, tau=1.0,
                        sigma=[1.0, 1.0], niter=niter)

    assert x == x0

    # Testing that sizes needs to agree:
    # Too few sigma_i:s
    with pytest.raises(ValueError):
        douglas_rachford_pd(x, f, g, lin_ops, tau=1.0,
                            sigma=[1.0], niter=niter)

    # Too many operators
    g_too_many = [odl.solvers.ZeroFunctional(space1),
                  odl.solvers.ZeroFunctional(space1),
                  odl.solvers.ZeroFunctional(space1)]
    with pytest.raises(ValueError):
        douglas_rachford_pd(x, f, g_too_many, lin_ops,
                            tau=1.0, sigma=[1.0, 1.0], niter=niter)

    # Test for correct space
    space2 = odl.uniform_discr(1, 2, 10)
    x = noise_element(space2)
    with pytest.raises(ValueError):
        douglas_rachford_pd(x, f, g, lin_ops, tau=1.0,
                            sigma=[1.0, 1.0], niter=niter)


def test_primal_dual_l1():
    """Verify that the correct value is returned for l1 dist optimization.

    Solves the optimization problem

        min_x ||x - data_1||_1 + 0.5 ||x - data_2||_1

    which has optimum value data_1.
    """

    # Define the space
    space = odl.rn(5)

    # Operator
    L = [odl.IdentityOperator(space)]

    # Data
    data_1 = odl.util.testutils.noise_element(space)
    data_2 = odl.util.testutils.noise_element(space)

    # Proximals
    f = odl.solvers.L1Norm(space).translated(data_1)
    g = [0.5 * odl.solvers.L1Norm(space).translated(data_2)]

    # Solve with f term dominating
    x = space.zero()
    douglas_rachford_pd(x, f, g, L, tau=3.0, sigma=[1.0], niter=10)

    assert all_almost_equal(x, data_1, places=2)


def test_primal_dual_no_operator():
    """Verify that the correct value is returned when there is no operator.

    Solves the optimization problem

        min_x ||x - data_1||_1

    which has optimum value data_1.
    """

    # Define the space
    space = odl.rn(5)

    # Operator
    L = []

    # Data
    data_1 = odl.util.testutils.noise_element(space)

    # Proximals
    f = odl.solvers.L1Norm(space).translated(data_1)
    g = []

    # Solve with f term dominating
    x = space.zero()
    douglas_rachford_pd(x, f, g, L, tau=3.0, sigma=[], niter=10)

    assert all_almost_equal(x, data_1, places=2)


def test_primal_dual_with_li():
    """Test for the forward-backward solver with infimal convolution.

    The test is done by minimizing the functional ``(g @ l)(x)``, where

        ``(g @ l)(x) = inf_y { g(y) + l(x - y) }``,

    g is the indicator function on [-3, -1], and l(x) = 1/2||x||_2^2.
    The optimal solution to this problem is given by x in [-3, -1].
    """
    # Parameter values for the box constraint
    upper_lim = -1
    lower_lim = -3

    space = odl.rn(1)

    lin_ops = [odl.IdentityOperator(space)]
    g = [odl.solvers.IndicatorBox(space, lower=lower_lim, upper=upper_lim)]
    f = odl.solvers.ZeroFunctional(space)
    l = [odl.solvers.L2NormSquared(space)]

    # Centering around a point further away from [-3,-1].
    x = space.element(10)

    douglas_rachford_pd(x, f, g, lin_ops, tau=0.5, sigma=[1.0], niter=20, l=l)

    assert lower_lim - 10 ** -LOW_ACCURACY <= float(x)
    assert float(x) <= upper_lim + 10 ** -LOW_ACCURACY


if __name__ == '__main__':
    odl.util.test_file(__file__)
