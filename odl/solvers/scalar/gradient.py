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

"""Gradient-based optimization schemes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External

# Internal

__all__ = ('steepest_descent',)


# TODO: update all docs


def steepest_descent(deriv, x, line_search, niter=1, partial=None):
    """Steepest descent method to minimize an objective function.

    General implementation of steepest decent (also known as gradient
    decent) for solving

    :math:`min f(x)`

    The algorithm is intended for unconstrained problems. It needs line
    search in order guarantee convergence. With appropriate line search,
    it can also be used for constrained problems where one wants to
    minimize over some given set ``C``. This is done by defining
    :math:`f(x) = \infty` for ``x`` outside ``C``.


    The algorithm is described in [1]_, section 9.3--9.4
    (`book available online
    <http://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_),
    [2]_, Section 12.2, and a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Gradient_descent>`_.

    Parameters
    ----------
    deriv : `odl.Operator`
        Gradient of the objective function, :math:`x \mapsto grad f(x)`
    x : element in the domain of ``deriv``
        Starting point of the iteration
    line_search : :class:`~odl.solvers.LineSearch`
        Strategy to choose the step length
    niter : `int`, optional
        Number of iterations
    partial : :class:`~odl.solvers.util.Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    References
    ----------
    .. [1] Boyd, Stephen, and Lieven Vandenberghe. Convex optimization.
       Cambridge university press, 2004. Available at

    .. [2] Griva, Igor, Stephen G. Nash, and Ariela Sofer. Linear
       and nonlinear optimization. Siam, 2009
    """

    grad = deriv.range.element()
    for _ in range(niter):
        deriv(x, out=grad)
        dir_derivative = -grad.norm() ** 2
        step = line_search(x, -grad, dir_derivative)
        x.lincomb(1, x, -step, grad)

        if partial is not None:
            partial.send(x)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
