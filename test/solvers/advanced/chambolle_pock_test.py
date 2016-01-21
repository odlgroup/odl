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
from odl.solvers import chambolle_pock_solver, f_dual_prox_l2_tv, g_prox_none
from odl.util.testutils import all_almost_equal


def test_proximal_factories():
    """Test factory function creating the proximal operator."""

    # Projection
    g = np.arange(10) / 2.
    g_space = odl.uniform_discr(0, 10, 10)
    g_data = g_space.element(g)

    # Image space
    im_space = odl.uniform_discr(0, 10, 10)
    im = im_space.element(np.arange(-10, 0))

    # Product space: range of matrix of operators
    K_range = odl.ProductSpace(g_space, odl.ProductSpace(im_space, im_space))

    # Product space element
    y = np.arange(0, 10)
    v0 = np.arange(10, 20)
    v1 = np.arange(30, 20, -1)
    K_vec = K_range.element([y, [v0, v1]])

    # Factory function for the prox op of the convex conjugate,F^*, of F
    lam = 10
    make_prox_f = f_dual_prox_l2_tv(K_range, g_data, lam=lam)

    # Proximal operator of F^*
    sigma = 0.5
    prox_op_f_cc = make_prox_f(sigma)

    x_f = prox_op_f_cc(K_vec)
    assert all_almost_equal(x_f[0].asarray(), (y - sigma * g) / (1 + sigma), 8)

    print('\n')

    # print(x_f[1][1])
    tmp = np.sqrt(v0 ** 2 + v1 ** 2)
    print(tmp)
    print(x_f[1][0])
    print(lam * v0/tmp)

    # Factory function for the proximal operator of G
    make_prox_g = g_prox_none(im_space)

    # Proximal operator of G
    prox_op_g = make_prox_g(3)

    x_g = prox_op_g(im)
    # print(im)
    # print(xg)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
