# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Step length computation for optimization schemes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from abc import ABCMeta, abstractmethod
import numpy as np

from odl.util.utility import with_metaclass


__all__ = ('LineSearch', 'BacktrackingLineSearch', 'ConstantLineSearch',
           'LineSearchFromIterNum')


class LineSearch(with_metaclass(ABCMeta, object)):

    """Abstract base class for line search step length methods."""

    @abstractmethod
    def __call__(self, x, direction, dir_derivative):
        """Calculate step length in direction.

        Parameters
        ----------
        x : `LinearSpaceElement`
            The current point
        direction : `LinearSpaceElement`
            Search direction in which the line search should be computed
        dir_derivative : float
            Directional derivative along the ``direction``

        Returns
        -------
        step : float
            Computed step length.
        """


class BacktrackingLineSearch(LineSearch):

    """Backtracking line search for step length calculation.

    This methods approximately finds the longest step length fulfilling
    the Armijo-Goldstein condition.

    The line search algorithm is described in [BV2004]_, page 464
    (`book available online
    <http://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_) and
    [GNS2009]_, pages 378--379. See also
    `Backtracking_line_search
    <https://en.wikipedia.org/wiki/Backtracking_line_search>`_.
    """

    def __init__(self, function, tau=0.5, discount=0.01, alpha=1.0,
                 max_num_iter=None, estimate_step=False):
        """Initialize a new instance.

        Parameters
        ----------
        function : callable
            The cost function of the optimization problem to be solved.
            If function is not a `Functional`, the argument `dir_derivative`
        tau : float, optional
            The amount the step length is decreased in each iteration,
            as long as it does not fulfill the decrease condition.
            The step length is updated as ``step_length *= tau``.
        discount : float, optional
            The "discount factor" on ``step length * direction derivative``,
            yielding the threshold under which the function value must lie to
            be accepted (see the references).
        alpha : float, optional
            The initial guess for the step length.
        max_num_iter : int, optional
            Maximum number of iterations allowed each time the line
            search method is called. If ``None``, this number is calculated
            to allow a shortest step length of 10 times machine epsilon.
        estimate_step : bool, optional
            If the last step should be used as a estimate for the next step.

        Examples
        --------
        Create line search

        >>> r3 = odl.rn(3)
        >>> func = odl.solvers.L2NormSquared(r3)
        >>> line_search = BacktrackingLineSearch(func)

        Find step in point x and direction d that decreases the function value.

        >>> x = r3.element([1, 2, 3])
        >>> d = r3.element([-1, -1, -1])
        >>> step_len = line_search(x, d)
        >>> step_len
        1.0
        >>> func(x + step_len * d) < func(x)
        True

        Also works with non-functionals as arguments, but then the
        dir_derivative argument is mandatory

        >>> r3 = odl.rn(3)
        >>> func = lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2
        >>> line_search = BacktrackingLineSearch(func)
        >>> x = r3.element([1, 2, 3])
        >>> d = r3.element([-1, -1, -1])
        >>> dir_derivative = -12
        >>> step_len = line_search(x, d, dir_derivative=dir_derivative)
        >>> step_len
        1.0
        >>> func(x + step_len * d) < func(x)
        True
        """
        self.function = function
        self.tau = float(tau)
        self.discount = float(discount)
        self.estimate_step = bool(estimate_step)
        self.alpha = float(alpha)

        self.total_num_iter = 0
        # Use a default value that allows the shortest step to be < 10 times
        # machine epsilon.
        if max_num_iter is None:
            try:
                dtype = self.function.domain.dtype
            except AttributeError:
                dtype = float
            eps = 10 * np.finfo(dtype).resolution
            self.max_num_iter = int(np.ceil(np.log(eps) / np.log(self.tau)))
        else:
            self.max_num_iter = int(max_num_iter)

    def __call__(self, x, direction, dir_derivative=None):
        """Calculate the optimal step length along a line.

        Parameters
        ----------
        x : `LinearSpaceElement`
            The current point
        direction : `LinearSpaceElement`
            Search direction in which the line search should be computed
        dir_derivative : float, optional
            Directional derivative along the ``direction``
            Default: ``function.gradient(x).inner(direction)``

        Returns
        -------
        step : float
            The computed step length
        """
        fx = self.function(x)
        if dir_derivative is None:
            try:
                gradient = self.function.gradient
            except AttributeError:
                raise ValueError('`dir_derivative` only optional if '
                                 '`function.gradient exists')
            else:
                dir_derivative = gradient(x).inner(direction)
        else:
            dir_derivative = float(dir_derivative)

        if dir_derivative == 0:
            raise ValueError('dir_derivative == 0, no descent can be found')

        if not self.estimate_step:
            alpha = 1.0
        else:
            alpha = self.alpha

        if dir_derivative > 0:
            # We need to move backwards if the direction is an increase
            # direction
            alpha *= -1

        if not np.isfinite(fx):
            raise ValueError('function returned invalid value {} in starting '
                             'point ({})'.format(fx, x))

        # Create temporary
        point = x.copy()

        num_iter = 0
        while True:
            if num_iter > self.max_num_iter:
                raise ValueError('number of iterations exceeded maximum: {}, '
                                 'step length: {}, without finding a '
                                 'sufficient decrease'
                                 ''.format(self.max_num_iter, alpha))

            point.lincomb(1, x, alpha, direction)  # pt = x + alpha * direction
            fval = self.function(point)

            print('alpha', alpha, fx, fval)
            if np.isnan(fval):
                # We do not want to compare against NaN below, and NaN should
                # indicate a user error.
                raise ValueError('function returned NaN in point '
                                 'point ({})'.format(point))

            expected_decrease = np.abs(alpha * dir_derivative * self.discount)
            if (fval <= fx - expected_decrease):
                # Stop iterating if the value decreases sufficiently.
                break

            num_iter += 1
            alpha *= self.tau

        assert fval < fx

        self.total_num_iter += num_iter
        self.alpha = np.abs(alpha)  # Store magnitude
        return alpha


class ConstantLineSearch(LineSearch):

    """Line search object that returns a constant step length."""

    def __init__(self, constant):
        """Initialize a new instance.

        Parameters
        ----------
        constant : float
            The constant step length
        """
        self.constant = float(constant)

    def __call__(self, x, direction, dir_derivative):
        """Calculate the step length at a point.

        All arguments are ignored and are only added to fit the interface.
        """
        return self.constant


class LineSearchFromIterNum(LineSearch):

    """Line search object that returns a step length from a function.

    The returned step length is ``func(iter_count)``.
    """

    def __init__(self, func):
        """Initialize a new instance.

        Parameters
        ----------
        func : callable
            Function that when called with an iteration count should return the
            step length. The iteration count starts at 0.

        Examples
        --------
        Make a step size that is 1.0 for the first 5 iterations, then 0.1:

        >>> def step_length(iter):
        ...     if iter < 5:
        ...         return 1.0
        ...     else:
        ...         return 0.1
        >>> line_search = LineSearchFromIterNum(step_length)
        """
        if not callable(func):
            raise TypeError('`func` must be a callable.')
        self.func = func
        self.iter_count = 0

    def __call__(self, x, direction, dir_derivative):
        """Calculate the step length at a point.

        All arguments are ignored and are only added to fit the interface.
        """
        step = self.func(self.iter_count)
        self.iter_count += 1
        return step


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
