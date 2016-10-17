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
from odl.operator import Operator
from odl.space.base_ntuples import FnBase


__all__ = ('NumericalDerivative', 'NumericalGradient',)


class NumericalDerivative(Operator):

    """The derivative of an operator by finite differences.

    See Also
    --------
    NumericalGradient : Compute gradient of a functional
    """

    def __init__(self, operator, point, method='forward', step=None):
        """Initialize a new instance.

        Parameters
        ----------
        operator : `Operator`
            The operator whose derivative should be computed numerically. Its
            domain and range must be `FnBase` spaces.
        point : ``operator.domain`` `element-like`
            The point to compute the derivative in.
        method : {'backward', 'forward', 'central'}
            The method to use to compute the derivative.
        step : float
            The step length used in the derivative computation.
            Default: selects the step according to the dtype of the space.

        Examples
        --------
        Compute a numerical estimate of the derivative (Hessian) of the squared
        L2 norm:

        >>> space = odl.rn(3)
        >>> func = odl.solvers.L2NormSquared(space)
        >>> hess = NumericalDerivative(func.gradient, [1, 1, 1])
        >>> hess([0, 0, 1])
        rn(3).element([0.0, 0.0, 2.0])

        Find the Hessian matrix:

        >>> odl.matrix_representation(hess)
        array([[ 2.,  0.,  0.],
               [ 0.,  2.,  0.],
               [ 0.,  0.,  2.]])

        Notes
        -----
        If the operator is :math:`A` and step size :math:`h` is used, the
        derivative in the point :math:`x` and direction :math:`dx` is computed
        as follows.

        ``method='backward'``:

        .. math::
            \\partial A(x)(dx) =
            (A(x) - A(x - dx \\cdot h / \| dx \|))
            \\cdot \\frac{\| dx \|}{h}

        ``method='forward'``:

        .. math::

            \\partial A(x)(dx) =
            (A(x + dx \\cdot h / \| dx \|) - A(x))
            \\cdot \\frac{\| dx \|}{h}

        ``method='central'``:

        .. math::
            \\partial A(x)(dx) =
            (A(x + dx \\cdot h / (2 \| dx \|)) -
             A(x - dx \\cdot h / (2 \| dx \|))
            \\cdot \\frac{\| dx \|}{h}

        The number of operator evaluations is ``2``, regardless of parameters.
        """
        if not isinstance(operator, Operator):
            raise TypeError('`operator` has to be an `Operator` instance')

        if not isinstance(operator.domain, FnBase):
            raise TypeError('`operator.domain` has to be an `FnBase` '
                            'instance')
        if not isinstance(operator.range, FnBase):
            raise TypeError('`operator.range` has to be an `FnBase` '
                            'instance')

        self.operator = operator
        self.point = operator.domain.element(point)

        if step is None:
            # Use half of the number of digits as machine epsilon, this
            # "usually" gives a good balance between precision and numerical
            # stability.
            self.step = np.sqrt(np.finfo(operator.domain.dtype).eps)
        else:
            self.step = float(step)

        self.method, method_in = str(method).lower(), method
        if self.method not in ('backward', 'forward', 'central'):
            raise ValueError("`method` '{}' not understood").format(method_in)

        Operator.__init__(self, operator.domain, operator.range,
                          linear=True)

    def _call(self, dx):
        """Return ``self(x)``."""
        x = self.point

        dx_norm = dx.norm()
        if dx_norm == 0:
            return 0

        scaled_dx = dx * (self.step / dx_norm)

        if self.method == 'backward':
            dAdx = self.operator(x) - self.operator(x - scaled_dx)
        elif self.method == 'forward':
            dAdx = self.operator(x + scaled_dx) - self.operator(x)
        elif self.method == 'central':
            dAdx = (self.operator(x + scaled_dx / 2) -
                    self.operator(x - scaled_dx / 2))
        else:
            raise RuntimeError('unknown method')

        return dAdx * (dx_norm / self.step)


class NumericalGradient(Operator):

    """The gradient of a `Functional` computed by finite differences.

    See Also
    --------
    NumericalDerivative : Compute directional derivative
    """

    def __init__(self, functional, method='forward', step=None):
        """Initialize a new instance.

        Parameters
        ----------
        functional : `Functional`
            The functional whose gradient should be computed. Its domain must
            be an `FnBase` space.
        method : {'backward', 'forward', 'central'}
            The method to use to compute the gradient.
        step : float
            The step length used in the derivative computation.
            Default: selects the step according to the dtype of the space.

        Examples
        --------
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
        If the functional is :math:`f` and step size :math:`h` is used, the
        gradient is computed as follows.

        ``method='backward'``:

        .. math::
            (\\nabla f(x))_i = \\frac{f(x) - f(x - h e_i)}{h}

        ``method='forward'``:

        .. math::
            (\\nabla f(x))_i = \\frac{f(x + h e_i) - f(x)}{h}

        ``method='central'``:

        .. math::
            (\\nabla f(x))_i = \\frac{f(x + (h/2) e_i) - f(x - (h/2) e_i)}{h}

        The number of function evaluations is ``functional.domain.size + 1`` if
        ``'backward'`` or ``'forward'`` is used and
        ``2 * functional.domain.size`` if ``'central'`` is used.
        On large domains this will be computationally infeasible.
        """
        if not isinstance(functional, Functional):
            raise TypeError('`functional` has to be a `Functional` instance')

        if not isinstance(functional.domain, FnBase):
            raise TypeError('`functional.domain` has to be an `FnBase` '
                            'instance')

        self.functional = functional
        if step is None:
            # Use half of the number of digits as machine epsilon, this
            # "usually" gives a good balance between precision and numerical
            # stability.
            self.step = np.sqrt(np.finfo(functional.domain.dtype).eps)
        else:
            self.step = float(step)

        self.method, method_in = str(method).lower(), method
        if self.method not in ('backward', 'forward', 'central'):
            raise ValueError("`method` '{}' not understood").format(method_in)

        Operator.__init__(self, functional.domain, functional.domain,
                          linear=functional.is_linear)

    def _call(self, x):
        """Return ``self(x)``."""
        # The algorithm takes finite differences in one dimension at a time
        # reusing the dx vector to improve efficiency.
        dfdx = self.domain.zero()
        dx = self.domain.zero()

        if self.method == 'backward':
            fx = self.functional(x)
            for i in range(self.domain.size):
                dx[i - 1] = 0  # reset step from last iteration
                dx[i] = self.step
                dfdx[i] = fx - self.functional(x - dx)
        elif self.method == 'forward':
            fx = self.functional(x)
            for i in range(self.domain.size):
                dx[i - 1] = 0  # reset step from last iteration
                dx[i] = self.step
                dfdx[i] = self.functional(x + dx) - fx
        elif self.method == 'central':
            for i in range(self.domain.size):
                dx[i - 1] = 0  # reset step from last iteration
                dx[i] = self.step / 2
                dfdx[i] = self.functional(x + dx) - self.functional(x - dx)
        else:
            raise RuntimeError('unknown method')

        dfdx /= self.step
        return dfdx

    def derivative(self, point):
        """Return the derivative in ``point``.

        The derivative of the gradient is often called the Hessian.

        Parameters
        ----------
        point : `domain` `element-like`
            The point that the derivative should be taken in.

        Returns
        -------
        derivative : `NumericalDerivative`
            Numerical estimate of the derivative. Uses the same method as this
            operator does, but with half the number of significant digits in
            the step size in order to preserve numerical stability.

        Examples
        --------
        Compute a numerical estimate of the derivative of the squared L2 norm:

        >>> space = odl.rn(3)
        >>> func = odl.solvers.L2NormSquared(space)
        >>> grad = NumericalGradient(func)
        >>> hess = grad.derivative([1, 1, 1])
        >>> hess([1, 0, 0])
        rn(3).element([2.0, 0.0, 0.0])

        Find the Hessian matrix:

        >>> odl.matrix_representation(hess)
        array([[ 2.,  0.,  0.],
               [ 0.,  2.,  0.],
               [ 0.,  0.,  2.]])
        """
        return NumericalDerivative(self, point,
                                   method=self.method, step=np.sqrt(self.step))

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
