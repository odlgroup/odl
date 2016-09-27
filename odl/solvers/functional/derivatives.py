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

"""Utilities for computing the gradient and Hessian of functionals."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.solvers.functional.functional import Functional
from odl.operator.operator import Operator
from odl.space.base_ntuples import FnBase


__all__ = ('NumericalGradient',)


class NumericalGradient(Operator):

    """The gradient of a `Functional` computed by finite differences."""

    def __init__(self, functional, method='forward', step=None):
        """Initialize a new instance.

        Parameters
        ----------
        functional : `Functional`
            The functional whose gradient should be computed. Its domain must
            be a `FnBase` space.
        method : {'backward', 'forward', 'central'}
            The method to use to compute the gradient.
        step : `None` or float
            The step length used in the gradient computation.
            `None` is interpreted as selecting the step according to the dtype
            of the space. ``step = 1e-8`` for ``float64`` spaces and
            ``step = 1e-4`` for ``float32`` spaces.

        Examples
        --------
        >>> import odl
        >>> space = odl.rn(3)
        >>> func = odl.solvers.L2NormSquared(space)
        >>> grad = NumericalGradient(func)
        >>> grad([1, 1, 1])
        rn(3).element([2.0, 2.0, 2.0])

        The gradient gives the correct value with sufficiently small step size:

        >>> grad([1, 1, 1]) == func.gradient([1, 1, 1])
        True

        If the step is too large the result is not correct:

        >>> grad = NumericalGradient(func, step=0.5)
        >>> grad([1, 1, 1])
        rn(3).element([2.5, 2.5, 2.5])

        But it can be improved by using the more accurate ``method='central'``:

        >>> grad = NumericalGradient(func, method='central', step=0.5)
        >>> grad([1, 1, 1])
        rn(3).element([2.0, 2.0, 2.0])

        Notes
        -----
        If the function is :math:`f(x)` and stepsize :math:`h` is used, the
        gradient is computed as follows.

        If ``method='backward'`` is used:

        .. math::
            \\frac{f(x) - f(x - h)}{h}

        if ``method='forward'`` is used:

        .. math::
            \\frac{f(x + h) - f(x)}{h}

        if ``method='central'`` is used:

        .. math::
            \\frac{f(x + h/2) - f(x - h/2)}{h}

        The number of function evaluations is ``functional.domain.size + 1`` if
        ``'backward'`` or ``'forward'`` is used and
        ``2 * functional.domain.size`` if ``'central'`` is used.
        """
        if not isinstance(functional, Functional):
            raise TypeError('`functional` has to be a `Functional` instance')

        if not isinstance(functional.domain, FnBase):
            raise TypeError('`functional.domain` has to be a `FnBase` '
                            'instance')

        self.functional = functional
        if step is None:
            self.step = np.sqrt(np.finfo(functional.domain.dtype).eps)
        else:
            self.step = float(step)

        self.method = str(method).lower()
        if method not in ('backward', 'forward', 'central'):
            raise ValueError('`method` not understood')

        Operator.__init__(self, functional.domain, functional.domain,
                          linear=functional.is_linear)

    def _call(self, x):
        """Return ``self(x)``."""
        dfdx = self.domain.zero()

        if self.method == 'backward':
            fx = self.functional(x)
            dx = self.domain.zero()
            for i in range(self.domain.size):
                dx[i-1] = 0  # reset step from last iteration
                dx[i] = self.step
                dfdx[i] = fx - self.functional(x - dx)
        elif self.method == 'forward':
            fx = self.functional(x)
            dx = self.domain.zero()
            for i in range(self.domain.size):
                dx[i-1] = 0  # reset step from last iteration
                dx[i] = self.step
                dfdx[i] = self.functional(x + dx) - fx
        elif self.method == 'central':
            dx = self.domain.zero()
            for i in range(self.domain.size):
                dx[i-1] = 0  # reset step from last iteration
                dx[i] = self.step / 2
                dfdx[i] = self.functional(x + dx) - self.functional(x - dx)
        else:
            raise RuntimeError('unknown method')

        dfdx /= self.step
        return dfdx


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
