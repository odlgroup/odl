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
from builtins import super

import numpy as np

from odl.discr.lp_discr import DiscreteLp
from odl.discr.tensor_ops import PointwiseTensorFieldOperator
from odl.operator.default_ops import MultiplyOperator
from odl.operator.pspace_ops import ReductionOperator
from odl.space.pspace import ProductSpace


class IntensityOperator(PointwiseTensorFieldOperator):

    """Intensity mapping of a vectorial function."""

    def __init__(self, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : power space of `DiscreteLp`, optional
            The space of elements which the operator acts on. If
            ``range`` is given, ``domain`` must be a power space
            of ``range``.
        range : `DiscreteLp`, optional
            The space of elements to which the operator maps.
            This is required if ``domain`` is not given.

        Notes
        -----
        This operator maps a real vector field :math:`f = (f_1, \dots, f_d)`
        to its pointwise intensity

            :math:`\mathcal{I}(f) = \\lvert f\\rvert^2 :
            x \mapsto \sum_{j=1}^d f_i(x)^2`.

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
            range = domain[0]

        super().__init__(domain, range, linear=False)

    def _call(self, x, out):
        """Implement ``self(x, out)``."""
        out[:] = x[0]
        out *= out

        tmp = self.base_space.element()
        for xi in x[1:]:
            tmp.assign(xi)
            tmp *= tmp
            out += tmp

    def derivative(self, f):
        """Return the derivative operator in ``f``.

        Parameters
        ----------
        f : domain element
            Point at which the derivative is taken

        Returns
        -------
        deriv : `Operator`
            Derivative operator at the specified point

        Notes
        -----
        The derivative of the intensity operator is given by

            :math:`\partial \mathcal{I}(f_1, f_2)(h_1, h_2) =
            2 (f_1 h_1 + f_2 h_2)`.

        Its adjoint maps a function :math:`g` to the product space
        element

            :math:`\\left[\partial\mathcal{I}(f_1, f_2)\\right]^*(g) =
            2 (f_1 g, f_2 g)`.
        """
        mul_ops = [2 * MultiplyOperator(fi, domain=self.base_space)
                   for fi in f]
        return ReductionOperator(*mul_ops)

