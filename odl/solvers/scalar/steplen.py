# Copyright 2014, 2015 The ODL development group
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

"""Step length computation for optimization schemes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
from abc import ABCMeta, abstractmethod
from math import ceil, log

# Internal
from odl.util.utility import with_metaclass

__all__ = ('StepLength', 'LineSearch',
           'BacktrackingLineSearch', 'ConstantLineSearch')


# TODO: find a good name
class StepLength(with_metaclass(ABCMeta, object)):

    """Abstract base class for step length methods."""

    # TODO: change signature so it reflects the requirements for e.g.
    # Barzilai-Borwein
    @abstractmethod
    def __call__(self, x, direction, dir_derivative):
        """
        Parameters
        ----------
        x : :attr:`~odl.Operator.domain` element
            The current point
        direction : :attr:`~odl.Operator.domain` element
            Search direction in which the line search should be computed
        dir_derivative : float
            Directional derivative along the ``direction``

        Returns
        -------
        alpha : `float`
            The step length
        """


class LineSearch(with_metaclass(ABCMeta, object)):

    """Abstract base class for line search step length methods."""

    @abstractmethod
    def __call__(self, x, direction, dir_derivative):
        """
        Parameters
        ----------
        x : :attr:`~odl.Operator.domain` element
            The current point
        direction : :attr:`~odl.Operator.domain` element
            Search direction in which the line search should be computed
        dir_derivative : `float`
            Directional derivative along the ``direction``

        Returns
        -------
        alpha : `float`
            The step length
        """


class BacktrackingLineSearch(LineSearch):

    """Backtracking line search for step length calculation.

    This methods approximately finds the longest step length fulfilling
    the Armijo-Goldstein condition.

    The line search algorithm is described in [1]_, page 464
    (`book available online
    <http://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_) and
    [2]_, pages 378--379. See also the
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Backtracking_line_search>`_.

    References
    ----------
    .. [1] Boyd, Stephen, and Lieven Vandenberghe. Convex optimization.
       Cambridge university press, 2004. Available at

    .. [2] Pages 378-379 in Griva, Igor, Stephen G. Nash, and
       Ariela Sofer. Linear and nonlinear optimization. Siam, 2009.
    """

    def __init__(self, function, tau=0.5, c=0.01, max_num_iter=None):
        """Initialize a new instance.

        Parameters
        ----------
        function : `callable`
            The cost function of the optimization problem to be solved.
        tau : `float`, optional
            The amount the step length is decreased in each iteration,
            as long as it does not fulfill the decrease condition.
            The step length is updated as ``step_length *= tau``
        c : `float`, optional
            The 'discount factor' on the
            ``step length * direction derivative``,
            which the new point needs to be smaller than in order to
            fulfill the condition and be accepted (see the references).
        max_num_iter : `int`, optional
            Maximum number of iterations allowed each time the line
            search method is called. If not set, this number  is
            calculated to allow a shortest step length of 0.0001.
        """
        self.function = function
        self.tau = tau
        self.c = c
        self.total_num_iter = 0
        # Use a default value that allows the shortest step to be < 0.0001
        # times the original step length
        if max_num_iter is None:
            self.max_num_iter = ceil(log(0.0001 / self.tau))
        else:
            self.max_num_iter = max_num_iter

    def __call__(self, x, direction, dir_derivative):
        """Calculate the optimal step length along a line.

        Parameters
        ----------
        x : :attr:`~odl.Operator.domain` element
            The current point
        direction : :attr:`~odl.Operator.domain` element
            Search direction in which the line search should be computed
        dir_derivative : `float`
            Directional derivative along the ``direction``

        Returns
        -------
        alpha : `float`
            The computed step length
        """
        alpha = 1.0
        fx = self.function(x)
        num_iter = 0
        while ((self.function(x + alpha * direction) >=
                fx + alpha * dir_derivative * self.c) and
               num_iter <= self.max_num_iter):
            num_iter += 1
            alpha *= self.tau
        self.total_num_iter += num_iter
        return alpha


class ConstantLineSearch(LineSearch):

    """Line search object that returns a constant step length."""

    def __init__(self, constant):
        """
        Parameters
        ----------
        constant : `float`
            The constant step length that the 'line search' object should
            return.
        """
        self.constant = constant

    def __call__(self, x, direction, dir_derivative):
        """
        Parameters
        ----------
        x : :attr:`~odl.Operator.domain` element
            The current point
        direction : :attr:`~odl.Operator.domain` element
            Search direction in which the line search should be computed
        dir_derivative : `float`
            Directional derivative along the ``direction``

        Returns
        -------
        alpha : `float`
            The constant step length
        """
        return self.constant


class BarzilaiBorweinStep(object):

    """Barzilai-Borwein method to compute a step length.

    Barzilai-Borwein method to compute a step length
    for gradient descent methods. It does ...

    TODO

    References
    ----------
    .. [1] J. Barzilai and J. M. Borwein. Two-point step size gradient
       methods. IMA Journal of Numerical Analysis, vol. 8, pp. 141--148,
       1988.
    .. [2] M. Raydan. The Barzilai and Borwein method for the large scale
       unconstrained minimization problem. SIAM J. Optim., vol. 7,
       pp. 26--33, 1997.
    """

    def __init__(self, gradf, step0=0.0005):
        """Initialize a new instance.

        Parameters
        ----------
        gradf: :class:`~odl.Operator`
            The gradient of the objective function at a point
        step0: `float`, optional
            Initial steplength parameter
        """
        self.gradf = gradf
        self.step0 = step0

    def __call__(self, x, x0):
        """
        Parameters
        ----------
        x: :attr:`~odl.Operator.domain` element
            The current point
        x0: :attr:`~odl.Operator.domain` element
            The previous point

        Returns
        -------
        step: `float`
            The step lenght
        """
        if x == x0:
            return self.step0

        gradx = self.gradf(x)

        if gradx == self.gradf(x0):
            return self.step0

        errx = x - x0
        grad_diff = gradx - self.gradf(x0)
        recip_step = grad_diff.inner(errx) / errx.norm() ** 2
        return 1.0 / recip_step
