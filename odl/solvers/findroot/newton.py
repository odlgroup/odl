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

"""(Quasi-)Newton schemes to find zeros of functions (gradients)."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from odl.operator import IdentityOperator
from odl.solvers.scalar.steplen import ConstantLineSearch
from odl.operator.oputils import matrix_representation

__all__ = ('bfgs_method', 'broydens_method')


# TODO: update all docs


def bfgs_method(f, x, line_search=1.0, maxiter=1000, tol=1e-16, callback=None):
    """Quasi-Newton BFGS method to minimize a differentiable function.

    This is a general and optimized implementation of a quasi-Newton
    method with BFGS update for solving a general unconstrained
    optimization problem

        :math:`\min f(x)`

    for a differentiable function
    :math:`f: \mathcal{X}\\to \mathbb{R}` on a Hilbert space
    :math:`\mathcal{X}`. It does so by finding a zero of the gradient

        :math:`\\nabla f: \mathcal{X} \\to \mathcal{X}`.

    The QN method is an approximate Newton method, where the Hessian
    is approximated and gradually updated in each step. This
    implementation uses the rank-one BFGS update schema where the
    inverse of the Hessian is recalculated in each iteration.

    The algorithm is described in [GNS2009]_, Section 12.3 and in the
    `BFGS Wikipedia article
    <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93\
Goldfarb%E2%80%93Shanno_algorithm>`_

    Parameters
    ----------
    f : `Functional`
        Functional with ``f.gradient``
    x : ``f.domain`` element
        Starting point of the iteration
    line_search : float or `LineSearch`, optional
        Strategy to choose the step length. If a float is given, uses it as a
        fixed step length.
    maxiter : int, optional
        Maximum number of iterations.
        ``tol``.
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate.
    """

    if x not in f.domain:
        raise TypeError('`x` {!r} is not in the domain of `grad` {!r}'
                        ''.format(x, f.domain))

    if not callable(line_search):
        line_search = ConstantLineSearch(line_search)

    grad = f.gradient

    hess = ident = IdentityOperator(f.domain)
    grad_x = grad(x)
    for _ in range(maxiter):
        # Determine a stepsize using line search
        search_dir = -hess(grad_x)
        dir_deriv = search_dir.inner(grad_x)
        if np.abs(dir_deriv) < tol:
            return  # we found an optimum
        step = line_search(x, direction=search_dir, dir_derivative=dir_deriv)

        # Update x
        x_update = search_dir
        x_update *= step
        x += x_update

        grad_x, grad_diff = grad(x), grad_x
        # grad_diff = grad(x) - grad(x_old)
        grad_diff.space.lincomb(-1, grad_diff, 1, grad_x, out=grad_diff)

        y_inner_s = grad_diff.inner(x_update)
        if np.abs(y_inner_s) < tol:
            return

        # Update Hessian
        hess = ((ident - x_update * grad_diff.T / y_inner_s) *
                hess *
                (ident - grad_diff * x_update.T / y_inner_s) +
                x_update * x_update.T / y_inner_s)

        if callback is not None:
            callback(x)


def broydens_method(f, x, line_search=1.0, impl='first', maxiter=1000,
                    tol=1e-16, callback=None):
    """Broyden's first method, a quasi-Newton scheme.

    This is a general and optimized implementation of Broyden's  method,
    a quasi-Newton method for solving a general unconstrained optimization
    problem

        :math:`\min f(x)`

    for a differentiable function
    :math:`f: \mathcal{X}\\to \mathbb{R}` on a Hilbert space
    :math:`\mathcal{X}`. It does so by finding a zero of the gradient

        :math:`\\nabla f: \mathcal{X} \\to \mathcal{X}`

    using a Newton-type update scheme with approximate Hessian.

    The algorithm is described in [Bro1965]_ and [Kva1991]_, and in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Broyden's_method>`_.

    Parameters
    ----------
    f : `Functional`
        Functional with ``f.gradient``
    x : ``f.domain`` element
        Starting point of the iteration
    line_search : float or `LineSearch`, optional
        Strategy to choose the step length. If a float is given, uses it as a
        fixed step length.
    impl : {'first', 'second'}
        What version of Broydens method to use. First is also known as Broydens
        'good' method, while the second is known as Broydens 'bad' method.
    maxiter : int, optional
        Maximum number of iterations.
        ``tol``.
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate.
    """
    if x not in f.domain:
        raise TypeError('`x` {!r} is not in the domain of `grad` {!r}'
                        ''.format(x, f.domain))
    grad = f.gradient

    if not callable(line_search):
        line_search = ConstantLineSearch(line_search)

    ident = IdentityOperator(f.domain)
    hess = IdentityOperator(f.domain)
    grad_x = grad(x)

    for _ in range(maxiter):
        # find step size
        search_dir = -hess(grad_x)
        dir_deriv = search_dir.inner(grad_x)
        if np.abs(dir_deriv) < tol:
            return  # we found an optimum
        step = line_search(x, search_dir, dir_deriv)

        # update x
        delta_x = step * search_dir
        x += delta_x

        # compute new gradient
        grad_x, grad_x_old = grad(x), grad_x
        delta_grad = grad_x - grad_x_old

        # update hessian
        v = hess(delta_grad)
        if impl == 'first':
            divisor = delta_x.inner(v)
            if np.abs(divisor) < tol:
                return
            u = (search_dir - v) * (1 / divisor)
            hess = (ident + u * delta_x.T) * hess
        elif impl == 'second':
            divisor = delta_grad.inner(delta_grad)
            if np.abs(divisor) < tol:
                return
            u = (search_dir - v) * (1 / divisor)
            hess = hess + u * delta_grad.T

        if callback is not None:
            callback(x)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
