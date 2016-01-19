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
from builtins import super
import numpy as np

# External

# Internal
from odl.operator.operator import Operator


def chambolle_pock(K, f_dual_prox, g_prox, niter=1, partial=None):
    """

    Parameters
    ----------

    niter : `int`, optional
        Number of iterations
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    `None`

    References
    ----------

    """
    sigma = 1
    tau = 1
    theta = 1

    x = K.domain.zero()
    xbar = x.copy()
    y = K.range.zero()

    f_dual_prox_sigma = f_dual_prox(sigma)
    g_prox_tau = g_prox(tau)
    Kadjoint = K.adjoint

    for _ in range(niter):

        xold = x.copy()
        f_dual_prox_sigma(y + sigma * K(xbar), out = y)
        x = g_prox_tau(x - tau * Kadjoint(y))
        xbar = x + theta * (x - xold)


        # TODO: decide on what to send
        # if partial is not None:
        #     partial.send(x)

    return x


def f_dual_prox_l2_tv(K, g, lam):

    def make_prox(sigma):

        class _prox_op(Operator):

            def __init__(self, sigma):
                self.sigma = sigma
                super().__init__(K.domain, K.range)

            def _call(self, x, out):


                y = x[0]
                z = x[1]

                out[0] = (y - self.sigma * g) / (1 + self.sigma)
                out[1] = lam * z /
                return out


        return _prox_op(sigma)

    return make_prox


import odl
from odl import IdentityOperator

g_space = odl.uniform_discr(0, 10, 10)
g1_space = odl.uniform_discr(0, 10, 10)
g2_space = odl.uniform_discr(0, 10, 10)

prod1 = odl.ProductSpace(g1_space, g2_space)
prod2 = odl.ProductSpace(g_space, prod1)
print(prod2)
print(prod2.zero())

K = IdentityOperator(g_space)
g = g_space.one()

tmp = f_dual_prox_l2_tv(K, g)
t = tmp(1)
print(type(tmp))
print(type(tmp(1)))
print(type(t))

out = t(g)
print(type(out))

