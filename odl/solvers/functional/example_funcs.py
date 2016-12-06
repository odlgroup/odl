# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Example functionals used in optimization."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.solvers.functional.functional import Functional
from odl.operator import Operator, MatrixOperator
from odl.space.base_tensors import TensorSpace


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
        space : `TensorSpace`
            Domain of the functional.
        scale : positive float, optional
            The scale ``c`` in the functional determining how
            "ill-behaved" the functional should be. Larger value means
            worse behavior.

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

        We can change how much the function is ill-behaved via ``scale``:

        >>> r2 = odl.rn(2)
        >>> functional = RosenbrockFunctional(r2, scale=2)
        >>> functional([1, 1])  # optimum is still 0 at [1, 1]
        0.0
        >>> functional([0, 1])  # much lower variation
        3.0
        """
        self.scale = float(scale)
        if not isinstance(space, TensorSpace):
            raise ValueError('`space` must be a `TensorSpace` instance, '
                             'got {!r}'.format(space))
        if space.size < 2:
            raise ValueError('`space.size` must be >= 2, got {}'
                             ''.format(space.size))
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
                return MatrixOperator(matrix, self.domain, self.range)

        return RosenbrockGradient()


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
