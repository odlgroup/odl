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
# along with ODL. If not, see <http://www.gnu.org/licenses/>.

"""Proximal operators.

The Proximal operators of f(x) is denoted by prox_tau[f](x) (or sometimes as
prox_{ tau f}( x) and defined as

    prox_tau[f](x) = arg min_y { f(y) + 1 / (2 * tau) * L2(x - y)^2 }

Separable sum property: if f is separable across two variables, i.e.
f(x, y) = g(x) + h(y), then

    prox_f(x, y) = prox_g(x) + prox_f(y)

Indicator function:

    ind_{S}(x) = {0 if x in S, infty if x not in S}

Special indicator function:

    ind_{box(a)}(x) = {0 if ||x||_infty <= a, infty if ||x||_infty > a}
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External

# Internal
from odl.operator.operator import Operator


class ProximalConvConjL2(Operator):

    """The proximal operator."""

    def __init__(self, sigma, space, g, lam):
        """Initialize the proximal operator.

        Parameters
        ----------
        sigma : positive `float`
        """
        self.sigma = float(sigma)
        self.lam = lam
        self.g = g
        super().__init__(domain=space, range=space, linear=False)

    def _call(self, x, out):
        """Apply the operator to ``x`` and stores the result in ``out``"""

        # (x - sig*g) / (1 + sig/lam)

        sig = self.sigma
        lam = self.lam
        out.lincomb(1 / (1 + sig / lam), x, -sig / (1 + sig / lam), self.g)


class ProximalConvConjL1(Operator):

    """The proximal operator."""

    def __init__(self, sigma, space, lam):
        """Initialize the proximal operator.

        Parameters
        ----------
        sigma : positive `float`
        """
        self.sigma = float(sigma)
        self.lam = lam
        super().__init__(domain=space, range=space, linear=False)

    def _call(self, x, out):
        """Apply the operator to ``x`` and stores the result in ``out``"""

        # lam * x / (max(lam, |x|))

        # Calculate |x| = pointwise 2-norm of x
        tmp = x ** 2
        sq_tmp = x.space.element()
        for xi in x[1:]:
            sq_tmp.multiply(xi, xi)
            tmp += sq_tmp
        tmp.ufunc.sqrt(out=tmp)

        # Pointwise maximum of |x| and lambda
        tmp.ufunc.maximum(self.lam, out=tmp)
        tmp /= self.lam

        for oi, xi in zip(out[1], x):
            oi.divide(xi, tmp)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
