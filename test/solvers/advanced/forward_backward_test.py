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
import numpy as np

# Internal
import odl
from odl.solvers import forward_backward_pd
from odl.util.testutils import all_almost_equal, example_element

# Places for the accepted error when comparing results
PLACES = 8


def test_forward_backward_inpit_handeling():
    """Test to see that input is handled correctly.
    """

    # Create spaces
    space1 = odl.uniform_discr(0, 1, 10)

    # Create "correct" set of operators
    lin_ops = [odl.ZeroOperator(space1), odl.ZeroOperator(space1)]
    prox_cc_g = [odl.solvers.proximal_zero(space1),  # Identity operator
                 odl.solvers.proximal_zero(space1)]  # Identity operator
    prox_f = odl.solvers.proximal_zero(space1)  # Identity operator
    grad_h = odl.ZeroOperator(space1)

    # Check that the algorithm runs. With the above operators the algorithm
    # is the identity operator in each iteration
    x0 = example_element(space1)
    x = x0.copy()
    niter = np.random.randint(low=3, high=20)

    forward_backward_pd(x, prox_f, prox_cc_g, lin_ops, grad_h, tau=1.0,
                        sigma=[1.0, 1.0], niter=niter)

    assert x == x0

    # Testing that sizes needs to agree:
    # To few sigma_i:s
    with pytest.raises(ValueError):
        forward_backward_pd(x, prox_f, prox_cc_g, lin_ops, grad_h, tau=1.0,
                            sigma=[1.0], niter=niter)

    # To many sigma_i:s
    with pytest.raises(ValueError):
        forward_backward_pd(x, prox_f, prox_cc_g, lin_ops, grad_h, tau=1.0,
                            sigma=[1.0, 1.0, 1.0], niter=niter)

    # To few operators
    prox_cc_g_to_few = [odl.solvers.proximal_zero(space1)]
    with pytest.raises(ValueError):
        forward_backward_pd(x, prox_f, prox_cc_g_to_few, lin_ops, grad_h,
                            tau=1.0, sigma=[1.0, 1.0], niter=niter)

    # To many operators
    prox_cc_g_to_many = [odl.solvers.proximal_zero(space1),
                         odl.solvers.proximal_zero(space1),
                         odl.solvers.proximal_zero(space1)]
    with pytest.raises(ValueError):
        forward_backward_pd(x, prox_f, prox_cc_g_to_many, lin_ops, grad_h,
                            tau=1.0, sigma=[1.0, 1.0], niter=niter)

    # Test for correct space
    space2 = odl.uniform_discr(1, 2, 10)
    x = example_element(space2)
    with pytest.raises(ValueError):
        forward_backward_pd(x, prox_f, prox_cc_g, lin_ops, grad_h, tau=1.0,
                            sigma=[1.0, 1.0], niter=niter)


def test_forward_backward():
    """Test for the forward-backward solver by minimizing ||x||_2^2. The
    general problem is of the form

        ``min_x f(x) + sum_i g_i(L_i x) + h(x)``

    and here we take f(x) = g(x) = 0, h(x) = ||x||_2^2 and L is the
    zero-operator.
    """

    # Create spaces
    space = odl.rn(10)

    # Create "correct" set of operators
    lin_ops = [odl.ZeroOperator(space)]
    prox_cc_g = [odl.solvers.proximal_cconj(odl.solvers.proximal_zero(space))]
    prox_f = odl.solvers.proximal_zero(space)
    grad_h = 2.0 * odl.IdentityOperator(space)  # Gradient of two-norm square

    x = example_element(space)
    x_global_min = space.zero()

    forward_backward_pd(x, prox_f, prox_cc_g, lin_ops, grad_h, tau=0.5,
                        sigma=[1.0], niter=10)

    assert all_almost_equal(x, x_global_min, places=PLACES)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
