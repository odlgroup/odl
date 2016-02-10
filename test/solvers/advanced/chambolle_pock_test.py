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

"""Test for the Chambolle-Pock solver."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
from odl.solvers import chambolle_pock_solver, proximal_zero
from odl.util.testutils import all_almost_equal


def test_chambolle_pock_solver():
    """Simple execution test for the Chambolle-Pock solver"""

    # Places for the accepted error when comparing results
    places = 8

    # Create a discretized image space
    pts = 5
    discr_space = odl.uniform_discr(0, 1, pts)

    # Operator
    identity = odl.IdentityOperator(discr_space)

    # Starting point (image)
    x0 = np.arange(pts)
    x = identity.domain.element(x0)

    # Relaxation variable needed to resume iteration
    xr = x.copy()

    # Dual variable needed to resume iteration
    y = identity.range.zero()

    # Proximal operator, use same the factory function for F^* and G
    prox = proximal_zero(discr_space)

    # Run the algorithm
    tau = 0.3
    sigma = 0.7
    theta = 0.9
    chambolle_pock_solver(identity, x, tau=tau, sigma=sigma,
                          proximal_primal=prox, proximal_dual=prox,
                          theta=theta, niter=1, partial=None,
                          x_relax=xr, y=y)

    # Explicit computation
    x1 = (1 - tau * sigma) * x0

    assert all_almost_equal(x, x1, places)

    # Explicit computation for the relaxation
    xr1 = (1 + theta) * x1 - theta * x0

    assert all_almost_equal(xr, xr1, places)

    # Resume iteration with previous x but without previous relaxation xr
    chambolle_pock_solver(identity, x, tau=tau, sigma=sigma,
                          proximal_primal=prox, proximal_dual=prox,
                          theta=theta, niter=1, partial=None)

    x2 = (1 - sigma * tau) * x1
    assert all_almost_equal(x, x2, places)

    # Resume iteration with x1 as above and with relaxation parameter xr
    x[:] = x1
    chambolle_pock_solver(identity, x, tau=tau, sigma=sigma,
                          proximal_primal=prox, proximal_dual=prox,
                          theta=theta, niter=1, partial=None,
                          x_relax=xr, y=y)

    x2 = x1 - tau * sigma * (x0 + xr1)
    assert all_almost_equal(x, x2, places)

    # Test with product space operator

    # Create product space operator
    prod_op = odl.ProductSpaceOperator([[identity], [-2 * identity]])

    # Starting point
    x0 = prod_op.domain.element([x0])
    x = x0.copy()

    # Proximal operator using the same factory function for F^* and G
    prox_primal = proximal_zero(prod_op.domain)
    prox_dual = proximal_zero(prod_op.range)

    # Run the algorithm
    chambolle_pock_solver(prod_op, x, tau=tau, sigma=sigma,
                          proximal_primal=prox_primal,
                          proximal_dual=prox_dual,
                          theta=theta, niter=1, partial=None)

    x1 = x0 - tau * sigma * prod_op.adjoint(prod_op(prod_op.domain.element([
        x0])))
    assert all_almost_equal(x, x1, places)

    # Test partial
    chambolle_pock_solver(prod_op, x, tau=tau, sigma=sigma,
                          proximal_primal=prox_primal,
                          proximal_dual=prox_dual, theta=theta, niter=1,
                          partial=odl.solvers.util.PrintIterationPartial())


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
