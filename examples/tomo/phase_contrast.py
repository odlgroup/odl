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

"""Phase contrast reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# Internal
import odl
from odl.discr.lp_discr import DiscreteLp
from odl.discr.tensor_ops import PointwiseTensorFieldOperator
from odl.space.pspace import ProductSpace


class IntensityOperator(PointwiseTensorFieldOperator):

    """Intensity mapping of a complex wave function.

    A complex wave function is interpreted as a product space
    element ``x in X^2``, where ``X`` is a discretized function
    space. It maps a pair ``(f1, f2)`` to

        ``I(f1, f2) = |1 + i/2*(f1 + i*f2)|^2 = 1 - f2 + (f1^2 + f2^2)/4``

    where ``k`` is the wave number of the incoming plane wave and
    ``d`` the propagation distance.
    """

    def __init__(self, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : power space of `DiscreteLp`, optional
            The space of elements which the operator acts on. If
            ``range`` is given, ``domain`` must fulfill
            ``domain == ProductSpace(range, 2)``.
            This is required if ``range`` is not given.
        range : `DiscreteLp`, optional
            The space of elements to which the operator maps.
            This is required if ``domain`` is not given.
        """
        if domain is None and range is None:
            raise ValueError('either domain or range must be specified.')

        if domain is None:
            if not isinstance(range, DiscreteLp):
                raise TypeError('range {!r} is not a DiscreteLp instance.'
                                ''.format(range))
            domain = ProductSpace(range, 2)

        if range is None:
            if not isinstance(domain, ProductSpace):
                raise TypeError('domain {!r} is not a `ProductSpace` '
                                'instance.'.format(domain))
            if domain.shape != (2,):
                raise ValueError('domain must be a power space of shape (2,), '
                                 'got {}.'.format(domain.shape))
            range = domain[0]

        super().__init__(domain, range, linear=False)

    def _call(self, x, out):
        """Implement ``self(x, out)``."""
        out[:] = 1.0
        out -= x[1]
        tmp = x[0].copy()
        tmp *= tmp
        tmp /= 4
        out += tmp
        tmp = x[1].copy()
        tmp *= tmp
        tmp /= 4
        out += tmp

    def derivative(self, f):
        """Return the derivative operator in ``f``.

        The derivative of the intensity operator is given by

            ``DI(f1, f2)(h1, h2) = -h2 + 1/2*(f1*h1 + f2*h2)``.

        Its adjoint maps a function ``g`` to the product space element

            ``DI(f1, f2)^*(g) = (f1/2 * g, (-1+f2/2) * g)``.
        """
        intens_op = self

        class Deriv(PointwiseTensorFieldOperator):
            def __init__(self):
                super().__init__(intens_op.domain, intens_op.range,
                                 linear=True)

            def _call(self, h, out):
                tmp = self.range.element()
                out.multiply(f[0], h[0])
                tmp.multiply(f[1], h[1])
                out.lincomb(0.5, out, 0.5, tmp)
                out -= h[1]

            @property
            def adjoint(self):
                return DerivAdjoint()

        class DerivAdjoint(PointwiseTensorFieldOperator):
            def __init__(self):
                super().__init__(intens_op.range, intens_op.domain,
                                 linear=True)

            def _call(self, g, out):
                out[0].multiply(f[0] / 2, g)
                out[1].assign(-g)
                out[1].lincomb(1, out[1], 0.5, f[1])

        return Deriv()
