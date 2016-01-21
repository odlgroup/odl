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

"""(Quasi-)Newton schemes to find zeros of functions (gradients)."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
from odl.solvers import chambolle_pock_solver, f_cc_prox_l2_tv, g_prox_none
from odl.util.testutils import all_almost_equal


def test_proximal_factories():
    """Test factory function creating the proximal operator."""

    precision = 8

    # Projection
    g = np.arange(20) / 2.
    g_space = odl.uniform_discr(0, 10, g.shape)
    g_data = g_space.element(g)

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x_data = x_space.element(np.arange(-10, 0))

    # Product space for matrix of operators
    K_range = odl.ProductSpace(g_space, odl.ProductSpace(x_space, x_space))

    # Product space element
    p = np.arange(0, g.size)
    vec0 = np.arange(5, 15)
    vec1 = np.arange(10, 0, -1)
    # dual variable
    y = K_range.element([p, [vec0, vec1]])

    # Factory function for the prox op of the convex conjugate,F^*, of F
    lam = 12
    make_prox_f = f_cc_prox_l2_tv(K_range, g_data, lam=lam)

    # Proximal operator of F^*
    sigma = 0.5
    prox_op_f_cc = make_prox_f(sigma)

    # Optimal point of the auxiliary minimization problem prox_sigma[F^*]
    y_opt = prox_op_f_cc(y)

    # First compoment of y_opt
    assert all_almost_equal(y_opt[0].asarray(),
                            (p - sigma * g) / (1 + sigma), precision)

    # Second compoment of y_opt
    tmp = np.sqrt(vec0 ** 2 + vec1 ** 2)
    tmp[tmp < lam] = lam
    tmp = lam * vec0 / tmp
    assert all_almost_equal(y_opt[1][0], tmp, precision)

    # Factory function for the proximal operator of G
    make_prox_g = g_prox_none(x_space)

    # Proximal operator of G
    tau = 3
    prox_op_g = make_prox_g(tau)

    # Optimal point of the auxiliary minimization problem prox_tau[F]
    x_opt = prox_op_g(x_data)

    assert x_data == x_opt


def test_chambolle_pock_solver():
    """Simple execution test for the Chambolle-Pock solver"""

    # Discretized space
    discr_space = odl.uniform_discr(0, 1, 10)

    # Operator
    op = odl.IdentityOperator(discr_space)

    # Proximal operator, use same the factory function for F^* and G
    prox = g_prox_none(discr_space)

    # Run the algorihtms
    rec = chambolle_pock_solver(op, prox, prox, tau=0.2, sigma=0.5,
                                niter=3, partial=None)
    assert all(rec) == 0

    # Product space operator
    prod_op = odl.ProductSpaceOperator([[op], [op]])

    # Proximal operator, use same the factory function for F^* and G
    prox_range = g_prox_none(prod_op.range)
    prox_domain = g_prox_none(prod_op.domain)

    # Run the algorihtms
    rec = chambolle_pock_solver(prod_op, prox_range, prox_domain,
                                sigma=0.1, tau=0.2, niter=3, partial=None)[0]

    assert all(rec) == 0


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
