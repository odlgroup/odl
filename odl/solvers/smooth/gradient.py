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

"""Gradient-based optimization schemes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


import numpy as np
from odl.solvers.util import ConstantLineSearch


__all__ = ('steepest_descent',)


# TODO: update all docs


def steepest_descent(f, x, line_search=1.0, maxiter=1000, tol=1e-16,
                     projection=None, callback=None):
    """Steepest descent method to minimize an objective function.

    General implementation of steepest decent (also known as gradient
    decent) for solving

        :math:`\min f(x)`

    The algorithm is intended for unconstrained problems. It needs line
    search in order guarantee convergence. With appropriate line search,
    it can also be used for constrained problems where one wants to
    minimize over some given set :math:`C`. This can be done by defining
    :math:`f(x) = \infty` for :math:`x\\not\\in C`, or by providing a
    ``projection`` function that projects the iterates on :math:`C`.

    The algorithm is described in [BV2004]_, section 9.3--9.4
    (`book available online
    <http://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_),
    [GNS2009]_, Section 12.2, and wikipedia
    `Gradient_descent
    <https://en.wikipedia.org/wiki/Gradient_descent>`_.

    Parameters
    ----------
    f : `Functional`
        Goal functional. Needs to have ``f.gradient``.
    x : ``f.domain`` element
        Starting point of the iteration
    line_search : float or `LineSearch`, optional
        Strategy to choose the step length. If a float is given, uses it as a
        fixed step length.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    projection : callable, optional
        Function that can be used to modify the iterates in each iteration,
        for example enforcing positivity. The function should take one
        argument and modify it in-place.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate

    See Also
    --------
    odl.solvers.iterative.iterative.landweber :
        Optimized solver for the case ``f(x) = ||Ax - b||_2^2``
    odl.solvers.iterative.iterative.conjugate_gradient :
        Optimized solver for the case ``f(x) = x^T Ax - 2 x^T b``
    """
    grad = f.gradient
    if x not in grad.domain:
        raise TypeError('`x` {!r} is not in the domain of `grad` {!r}'
                        ''.format(x, grad.domain))

    if not callable(line_search):
        line_search = ConstantLineSearch(line_search)

    grad_x = grad.range.element()
    for _ in range(maxiter):
        grad(x, out=grad_x)

        dir_derivative = -grad_x.norm() ** 2
        if np.abs(dir_derivative) < tol:
            return  # we have converged
        step = line_search(x, -grad_x, dir_derivative)

        x.lincomb(1, x, -step, grad_x)

        if projection is not None:
            projection(x)

        if callback is not None:
            callback(x)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
