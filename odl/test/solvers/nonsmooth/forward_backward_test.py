# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Test for the forward-backward solver."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import pytest

# Internal
import odl
from odl.solvers import forward_backward_pd
from odl.util.testutils import all_almost_equal, almost_equal, noise_element

# Places for the accepted error when comparing results
HIGH_ACCURACY = 8
LOW_ACCURACY = 4


def test_forward_backward_input_handling():
    """Test to see that input is handled correctly."""

    space1 = odl.uniform_discr(0, 1, 10)

    lin_ops = [odl.ZeroOperator(space1), odl.ZeroOperator(space1)]
    g = [odl.solvers.ZeroFunctional(space1),
         odl.solvers.ZeroFunctional(space1)]
    f = odl.solvers.ZeroFunctional(space1)
    h = odl.solvers.ZeroFunctional(space1)

    # Check that the algorithm runs. With the above operators, the algorithm
    # returns the input.
    x0 = noise_element(space1)
    x = x0.copy()
    niter = 3

    forward_backward_pd(x, f, g, lin_ops, h, tau=1.0,
                        sigma=[1.0, 1.0], niter=niter)

    assert x == x0

    # Testing that sizes needs to agree:
    # Too few sigma_i:s
    with pytest.raises(ValueError):
        forward_backward_pd(x, f, g, lin_ops, h, tau=1.0,
                            sigma=[1.0], niter=niter)

    # Too many operators
    g_too_many = [odl.solvers.ZeroFunctional(space1),
                  odl.solvers.ZeroFunctional(space1),
                  odl.solvers.ZeroFunctional(space1)]
    with pytest.raises(ValueError):
        forward_backward_pd(x, f, g_too_many, lin_ops, h,
                            tau=1.0, sigma=[1.0, 1.0], niter=niter)

    # Test for correct space
    space2 = odl.uniform_discr(1, 2, 10)
    x = noise_element(space2)
    with pytest.raises(ValueError):
        forward_backward_pd(x, f, g, lin_ops, h, tau=1.0,
                            sigma=[1.0, 1.0], niter=niter)


def test_forward_backward_basic():
    """Test for the forward-backward solver by minimizing ||x||_2^2.

    The general problem is of the form

        ``min_x f(x) + sum_i g_i(L_i x) + h(x)``

    and here we take f(x) = g(x) = 0, h(x) = ||x||_2^2 and L is the
    zero-operator.
    """

    space = odl.rn(10)

    lin_ops = [odl.ZeroOperator(space)]
    g = [odl.solvers.ZeroFunctional(space)]
    f = odl.solvers.ZeroFunctional(space)
    h = odl.solvers.L2NormSquared(space)

    x = noise_element(space)
    x_global_min = space.zero()

    forward_backward_pd(x, f, g, lin_ops, h, tau=0.5,
                        sigma=[1.0], niter=10)

    assert all_almost_equal(x, x_global_min, places=HIGH_ACCURACY)


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

    assert all_almost_equal(x, x_global_min, places=LOW_ACCURACY)


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
    assert almost_equal(x[0], expected_result, places=LOW_ACCURACY)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
