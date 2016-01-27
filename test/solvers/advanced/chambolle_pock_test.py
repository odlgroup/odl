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
from odl.solvers import (chambolle_pock_solver, proximal_zero,
                         proximal_convexconjugate_l2_l1, proximal_primal)
from odl.util.testutils import all_almost_equal, all_equal


def test_proximal():

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x_data = x_space.element(np.arange(-5, 5))
    out = x_space.element()

    make_prox = proximal_primal(x_space)

    prox_op = make_prox(0)

    assert isinstance(prox_op, odl.IdentityOperator)

    make_prox = proximal_primal(x_space, 'non-negative')

    prox_op = make_prox(0)

    prox_op(x_data, out)

    assert all(out.asarray() >= 0)




# TODO: improve test documentation
def test_proximal_factories():
    """Test of factory functions creating the proximal operator instances."""

    precision = 8

    # Projection
    g = np.arange(20) / 2.
    g_space = odl.uniform_discr(0, 10, g.shape)
    g_data = g_space.element(g)

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x_data = x_space.element(np.arange(-10, 0))

    # Product space for matrix of operators
    operator_range = odl.ProductSpace(g_space, odl.ProductSpace(x_space, 2))

    # Product space element
    p = np.arange(g.size)
    vec0 = np.arange(5, 15)
    vec1 = np.arange(10, 0, -1)
    # dual variable
    y = operator_range.element([p, [vec0, vec1]])

    # Factory function for the prox op of the convex conjugate,F^*, of F
    lam = 12
    make_prox_f = proximal_convexconjugate_l2_l1(operator_range, g_data,
                                                 lam=lam)

    # Proximal operator of F^*
    sigma = 0.5
    prox_op_f_cc = make_prox_f(sigma)
    print(sigma)

    # Optimal point of the auxiliary minimization problem prox_sigma[F^*]
    y_opt = prox_op_f_cc(y)

    # First component of y_opt
    assert all_almost_equal(y_opt[0].asarray(),
                            (p - sigma * g) / (1 + sigma), precision)

    # Second component of y_opt
    tmp = np.sqrt(vec0 ** 2 + vec1 ** 2)
    tmp[tmp < lam] = lam
    tmp = lam * vec0 / tmp

    assert all_almost_equal(y_opt[1][0], tmp, precision)

    # Factory function for the proximal operator of G
    make_prox_g = proximal_zero(x_space)

    # Proximal operator of G
    tau = 3
    prox_op_g = make_prox_g(tau)

    # Optimal point of the auxiliary minimization problem prox_tau[F]
    x_opt = prox_op_g(x_data)

    assert x_data == x_opt


def test_chambolle_pock_solver():
    """Simple execution test for the Chambolle-Pock solver"""

    precision = 8

    # Discretized space
    n = 5
    discr_space = odl.uniform_discr(0, 1, n)

    # Operator
    identity = odl.IdentityOperator(discr_space)

    # Starting point
    x0 = np.arange(n)
    x = identity.domain.element(x0)
    xr = x.copy()
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
                          x_relaxation=xr, y=y)

    x1 = (1 - tau * sigma) * x0
    assert all_almost_equal(x, x1, precision)
    xr1 = (1 + theta) * x1 - theta * x0
    assert all_almost_equal(xr, xr1)

    # Resume iteration with previous x but without previous relaxation xr
    chambolle_pock_solver(identity, x, tau=tau, sigma=sigma,
                          proximal_primal=prox, proximal_dual=prox,
                          theta=theta, niter=1, partial=None)

    x2 = (1 - sigma * tau) * x1
    assert all_almost_equal(x, x2, precision)

    # Resume iteration with x1 as above and with relaxation parameter xr
    x[:] = x1
    chambolle_pock_solver(identity, x, tau=tau, sigma=sigma,
                          proximal_primal=prox, proximal_dual=prox,
                          theta=theta, niter=1, partial=None,
                          x_relaxation=xr, y=y)

    x2 = x1 - tau * sigma * (x0 + xr1)
    assert all_almost_equal(x, x2, precision)

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
    assert all_almost_equal(x, x1, precision)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
