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

"""Newton type optimization schemes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External

# Internal
from odl.solvers.iterative.iterative import conjugate_gradient


__all__ = ('newtons_method',)


def newtons_method(op, x, line_search, num_iter=10, cg_iter=None,
                   partial=None):
    """Newton's method for solving a system of equations.

    This is a general and optimized implementation of Newton's method
    for solving the problem

        :math:`f(x) = 0`.

    of finding a root of a function :math:`f`.

    The algorithm is well-known and there is a vast literature about it.
    Among others, the method is described in [1]_, Sections 9.5 and 10.2
    (`book available online
    <http://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_),
    [2]_,  Section 2.7 for solving nonlinear equations and Section 11.3 for
    it's use in minimization, and wikipedia on  `Newton's_method
    <https://en.wikipedia.org/wiki/Newton's_method>`_.

    Parameters
    ----------
    deriv : :class:`~odl.Operator`
        Gradient of the objective function, :math:`x \mapsto grad f(x)`
    x : element in the domain of ``deriv``
        Starting point of the iteration
    line_search : :class:`~odl.solvers.scalar.steplen.LineSearch`
        Strategy to choose the step length
    num_iter : `int`, optional
        Number of iterations
    cg_iter : `int`, optional
        Number of iterations in the the conjugate gradient solver,
        for comuting the search direction.
    partial : :class:`~odl.solvers.util.partial.Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    Notes
    ----------
    The algorithm works by iteratively solving

        :math:`\partial f(x_k)p_k = -f(x_k)`

    and then updating as

        :math:`x_{k+1} = x_k + \\alpha x_k`,

    where :math:`\\alpha` is a suitable step length (see the
    references). In this implementation the system of equations are
    solved using the conjugate gradient method.

    References
    ----------
    .. [1] Boyd, Stephen, and Lieven Vandenberghe. Convex optimization.
       Cambridge university press, 2004.

    .. [2] Griva, Igor, Stephen G. Nash, and Ariela Sofer. Linear
       and nonlinear optimization. Siam, 2009
    """
    # TODO: update doc
    if cg_iter is None:
        # Motivated by that if it is Ax = b, x and b in Rn, it takes at most n
        # iterations to solve with cg
        cg_iter = op.domain.size

    # TODO: optimize by using lincomb and avoiding to create copies
    for _ in range(num_iter):

        # Initialize the search direction to 0
        search_direction = x.space.zero()

        # Compute hessian (as operator) and gradient in the current point
        hessian = op.derivative(x)
        deriv_in_point = op(x).copy()

        # Solving A*x = b for x, in this case f'(x)*p = -f(x)
        # TODO: Let the user provide/choose method for how to solve this?
        conjugate_gradient(hessian, search_direction,
                           -1 * deriv_in_point, cg_iter)

        # Computing step length
        dir_deriv = search_direction.inner(deriv_in_point)
        step_length = line_search(x, search_direction, dir_deriv)

        # Updating
        x += step_length * search_direction

        if partial is not None:
            partial.send(x)
