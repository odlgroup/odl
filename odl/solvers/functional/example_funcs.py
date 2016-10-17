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

"""Example functionals used in optimization."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.solvers.functional.functional import Functional
from odl.operator.operator import Operator
from odl.space.base_ntuples import FnBase
from odl.space.npy_ntuples import MatVecOperator


__all__ = ('RosenbrockFunctional',)


class RosenbrockFunctional(Functional):

    """The well-known `Rosenbrock function`_ on ``R^n``.

    This function is usually used as a test problem in smooth optimization.

    Notes
    -----
    The functional is defined for :math:`x` in :math:`\\mathbb{R}^n`,
    :math:`n \\geq 2`, as

    .. math::
        \sum_{i=1}^{n - 1} c (x_{i+1} - x_i^2)^2 + (1 - x_i)^2

    Where :math:`c` is a constant usually set to 100 which determines how "ill-
    behaved" the function should be.
    It has a minimum at :math:`x = [1, \\dots, 1]`, independent of :math:`c`.

    There are two definitions of the n-dimensional Rosenbrock function found in
    the literature. One is the product of 2-dimensional Rosenbrock functions,
    which is not the one used here. This one extends the pattern of the 2d
    Rosenbrock function so all dimensions depend on each other in sequence.

    References
    ----------
    .. _Rosenbrock function: en.wikipedia.org/wiki/Rosenbrock_function
    """

    def __init__(self, space, scale=100.0):
        """Initialize a new instance.

        Parameters
        ----------
        space : `FnBase`
            Domain of the functional.
        scale : positive float, optional
            The scale ``c`` in the functional determining how "ill-behaved" the
            functional should be.

        Examples
        --------
        Initialize and call the functional:

        >>> r2 = odl.rn(2)
        >>> functional = RosenbrockFunctional(r2)
        >>> functional([1, 1])  # optimum is 0 at [1, 1]
        0.0
        >>> functional([0, 1])
        101.0

        The functional can also be used in higher dimensions:

        >>> r5 = odl.rn(5)
        >>> functional = RosenbrockFunctional(r5)
        >>> functional([1, 1, 1, 1, 1])
        0.0

        We can change how much the function is ill behaved via ``scale``:

        >>> r2 = odl.rn(2)
        >>> functional = RosenbrockFunctional(r2, scale=2)
        >>> functional([1, 1])  # optimum is still 0 at [1, 1]
        0.0
        >>> functional([0, 1])  # much lower variation
        3.0
        """
        self.scale = float(scale)
        if not isinstance(space, FnBase):
            raise ValueError('`space` must be an `FnBase`')
        if space.size < 2:
            raise ValueError('`space` must be at least two dimensional')
        super().__init__(space=space, linear=False, grad_lipschitz=np.inf)

    def _call(self, x):
        """Return ``self(x)``."""
        result = 0
        for i in range(0, self.domain.size - 1):
            result += (self.scale * (x[i + 1] - x[i] ** 2) ** 2 +
                       (x[i] - 1) ** 2)

        return result

    @property
    def gradient(self):
        """Gradient operator of the Rosenbrock functional."""
        functional = self
        c = self.scale

        class RosenbrockGradient(Operator):

            """The gradient operator of the Rosenbrock functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x, out):
                """Apply the gradient operator to the given point."""
                for i in range(1, self.domain.size - 1):
                    out[i] = (2 * c * (x[i] - x[i - 1]**2) -
                              4 * c * (x[i + 1] - x[i]**2) * x[i] -
                              2 * (1 - x[i]))
                out[0] = (-4 * c * (x[1] - x[0] ** 2) * x[0] +
                          2 * (x[0] - 1))
                out[-1] = 2 * c * (x[-1] - x[-2] ** 2)

            def derivative(self, x):
                """The derivative of the gradient.

                This is also known as the Hessian.
                """

                # TODO: Implement optimized version of this that does not need
                # a matrix.
                shape = (functional.domain.size, functional.domain.size)
                matrix = np.zeros(shape)

                # Straightforward computation
                for i in range(0, self.domain.size - 1):
                    matrix[i, i] = (2 * c + 2 + 12 * c * x[i] ** 2 -
                                    4 * c * x[i + 1])
                    matrix[i + 1, i] = -4 * c * x[i]
                    matrix[i, i + 1] = -4 * c * x[i]
                matrix[-1, -1] = 2 * c
                matrix[0, 0] = 2 + 12 * c * x[0] ** 2 - 4 * c * x[1]
                return MatVecOperator(matrix, self.domain, self.range)

        return RosenbrockGradient()


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
