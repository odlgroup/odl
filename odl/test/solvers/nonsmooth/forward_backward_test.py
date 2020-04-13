# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the forward-backward solver."""

from __future__ import division

import pytest

import odl
from odl.solvers import forward_backward_pd
from odl.util.testutils import all_almost_equal, noise_element

HIGH_ACCURACY = 8
LOW_ACCURACY = 4


def test_forward_backward_input_handling():
    """Test to see that input is handled correctly."""

    space = odl.uniform_discr(0, 1, 10)

    L = [odl.ZeroOperator(space), odl.ZeroOperator(space)]
    g = [odl.solvers.ZeroFunctional(space),
         odl.solvers.ZeroFunctional(space)]
    f = odl.solvers.ZeroFunctional(space)
    h = odl.solvers.ZeroFunctional(space)

    # Check that the algorithm runs. With the above operators, the algorithm
    # returns the input.
    x0 = noise_element(space)
    x = space.copy(x0)
    niter = 3

    forward_backward_pd(x, f, g, L, h, tau=1.0,
                        sigma=[1.0, 1.0], niter=niter)

    assert all_almost_equal(x, x0)

    # Testing that sizes needs to agree:
    # Too few sigmas
    with pytest.raises(ValueError):
        forward_backward_pd(x, f, g, L, h, tau=1.0,
                            sigma=[1.0], niter=niter)

    # Too many operators
    g_too_many = [odl.solvers.ZeroFunctional(space),
                  odl.solvers.ZeroFunctional(space),
                  odl.solvers.ZeroFunctional(space)]
    with pytest.raises(ValueError):
        forward_backward_pd(x, f, g_too_many, L, h,
                            tau=1.0, sigma=[1.0, 1.0], niter=niter)


def test_forward_backward_basic():
    """Test for the forward-backward solver by minimizing ||x||_2^2.

    The general problem is of the form

        ``min_x f(x) + sum_i g_i(L_i x) + h(x)``

    and here we take f(x) = g(x) = 0, h(x) = ||x||_2^2 and L is the
    zero-operator.
    """
    space = odl.rn(10)

    L = [odl.ZeroOperator(space)]
    g = [odl.solvers.ZeroFunctional(space)]
    f = odl.solvers.ZeroFunctional(space)
    h = odl.solvers.L2NormSquared(space)

    x = noise_element(space)
    x_global_min = space.zero()

    forward_backward_pd(x, f, g, L, h, tau=0.5,
                        sigma=[1.0], niter=10)

    assert all_almost_equal(x, x_global_min, ndigits=HIGH_ACCURACY)


def test_forward_backward_with_lin_ops():
    """Test for the forward-backward solver with linear operatros.

    The test is done by minimizing ||x - b||_2^2 + ||alpha * x||_2^2. The
    general problem is of the form

        ``min_x f(x) + sum_i g_i(L_i x) + h(x)``

    and here we take f = 0, g = ||.||_2^2, L = alpha * IndentityOperator,
    and h = ||. - b||_2^2.
    """

    space = odl.rn(10)
    alpha = 0.1
    b = noise_element(space)

    lin_ops = [alpha * odl.IdentityOperator(space)]
    g = [odl.solvers.L2NormSquared(space)]
    f = odl.solvers.ZeroFunctional(space)

    # Gradient of two-norm square
    h = odl.solvers.L2NormSquared(space).translated(b)

    x = noise_element(space)

    # Explicit solution: x_hat = (I^T * I + (alpha*I)^T * (alpha*I))^-1 * (I*b)
    x_global_min = b / (1 + alpha ** 2)

    forward_backward_pd(x, f, g, lin_ops, h, tau=0.5,
                        sigma=[1.0], niter=20)

    assert all_almost_equal(x, x_global_min, ndigits=LOW_ACCURACY)


def test_forward_backward_with_li():
    """Test for the forward-backward solver with infimal convolution.

    The test is done by minimizing the functional ``(g @ l)(x)``, where

        ``(g @ l)(x) = inf_y { g(y) + l(x-y) }``,

    g is the indicator function on [-3, -1], and l(x) = 1/2||x||_2^2.
    The optimal solution to this problem is given by x in [-3, -1].
    """
    # Parameter values for the box constraint
    upper_lim = -1
    lower_lim = -3

    space = odl.rn(1)

    lin_op = odl.IdentityOperator(space)
    lin_ops = [lin_op]
    g = [odl.solvers.IndicatorBox(space, lower=lower_lim, upper=upper_lim)]
    f = odl.solvers.ZeroFunctional(space)
    h = odl.solvers.ZeroFunctional(space)
    l = [0.5 * odl.solvers.L2NormSquared(space)]

    # Creating an element not to far away from [-3,-1], in order to converge in
    # a few number of iterations.
    x = space.element(10)

    forward_backward_pd(x, f, g, lin_ops, h, tau=0.5,
                        sigma=[1.0], niter=20, l=l)

    assert lower_lim - 10 ** (-LOW_ACCURACY) <= x[0]
    assert x[0] <= upper_lim + 10 ** (-LOW_ACCURACY)


def test_forward_backward_with_li_and_h():
    """Test for the forward-backward solver with infimal convolution.

    The test is done by minimizing the functional ``(g @ l)(x) + h(x)``, where

        ``(g @ l)(x) = inf_y { g(y) + l(x-y) }``,

    g is the indicator function on [-3, -1], and l(x) = h(x) = 1/2||x||_2^2.
    The optimal solution to this problem is given by x = -0.5.
    """

    # Parameter values for the box constraint
    upper_lim = -1
    lower_lim = -3

    space = odl.rn(1)

    lin_ops = [odl.IdentityOperator(space)]
    g = [odl.solvers.IndicatorBox(space, lower=lower_lim, upper=upper_lim)]
    f = odl.solvers.ZeroFunctional(space)
    h = 0.5 * odl.solvers.L2NormSquared(space)
    l = [0.5 * odl.solvers.L2NormSquared(space)]

    # Creating an element not to far away from -0.5, in order to converge in
    # a few number of iterations.
    x = space.element(10)

    forward_backward_pd(x, f, g, lin_ops, h, tau=0.5,
                        sigma=[1.0], niter=20, l=l)

    expected_result = -0.5
    assert x[0] == pytest.approx(expected_result, rel=10 ** -LOW_ACCURACY)


if __name__ == '__main__':
    odl.util.test_file(__file__)
