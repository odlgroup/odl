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

from odl.operator.default_ops import IdentityOperator


__all__ = ('bfgs_method', 'broydens_first_method', 'broydens_second_method')


# TODO: update all docs


def bfgs_method(grad, x, line_search, niter=1, callback=None):
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
    grad : `Operator`
        Gradient mapping of the objective function, i.e. the mapping
        :math:`x \mapsto \\nabla f(x) \\in \mathcal{X}`
    x : `element` of the domain of ``grad``
        Starting point of the iteration
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : `int`, optional
        Number of iterations
    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate.

    Returns
    -------
    `None`
    """
    hess = ident = IdentityOperator(grad.range)
    grad_x = grad(x)
    for _ in range(niter):
        search_dir = -hess(grad_x)
        dir_deriv = search_dir.inner(grad_x)
        step = line_search(x, direction=search_dir, dir_derivative=dir_deriv)

        x_update = search_dir
        x_update *= step
        x += x_update

        grad_x, grad_diff = grad(x), grad_x
        # grad_diff = grad(x) - grad(x_old)
        grad_diff.space.lincomb(-1, grad_diff, 1, grad_x, out=grad_diff)

        y_inner_s = grad_diff.inner(x_update)
        # TODO: use a small (adjustable) tolerance instead of 0.0
        if y_inner_s == 0.0:
            return

        # Update Hessian
        hess = ((ident - x_update * grad_diff.T / y_inner_s) *
                hess *
                (ident - grad_diff * x_update.T / y_inner_s) +
                x_update * x_update.T / y_inner_s)

        if callback is not None:
            callback(x)


def broydens_first_method(grad, x, line_search, niter=1, callback=None):
    """Broyden's first method, a quasi-Newton scheme.

    This is a general and optimized implementation of Broyden's first
    (or 'good') method, a quasi-Newton method for solving a general
    unconstrained optimization problem

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
    grad : `Operator`
        Gradient mapping of the objective function, i.e. the mapping
        :math:`x \mapsto \\nabla f(x) \\in \mathcal{X}`
    x : `element` of the domain of ``grad``
        Starting point of the iteration
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : `int`, optional
        Number of iterations
    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate.

    Returns
    -------
    `None`
    """
    hess = IdentityOperator(grad.range)
    grad_x = grad(x)
    search_dir = hess(grad_x)

    for _ in range(niter):
        step = line_search(x, search_dir, grad_x)

        u = search_dir.copy()  # Save old hess(grad f(x))

        x_update = search_dir  # Just rename, still writing to search_dir
        x_update *= step
        x += x_update  # x_(k+1) = x_k + s * d

        grad_x, grad_diff = grad(x), grad_x
        # grad_diff = grad(x) - grad(x_old)
        grad_diff.space.lincomb(-1, grad_diff, 1, grad_x, out=grad_diff)

        search_dir = hess(grad_x)  # Calculate new hess(grad f (x))

        v = hess(x_update)  # v = H(s * d)
        scalprod = v.inner(grad_diff)
        if scalprod == 0:
            return

        u *= (step - 1)
        u += search_dir
        u /= scalprod
        hess -= u * v.T

        if callback is not None:
            callback(x)


def broydens_second_method(grad, x, line_search, niter=1, callback=None):
    """Broyden's first method, a quasi-Newton scheme.

    This is a general and optimized implementation of Broyden's second
    (or 'bad') method, a quasi-Newton method for solving a general
    unconstrained optimization problem

        :math:`\min f(x)`

    for a differentiable function
    :math:`f: \mathcal{X}\\to \mathbb{R}` on a Hilbert space
    :math:`\mathcal{X}`. It does so by finding a zero of the gradient

        :math:`\\nabla f: \mathcal{X} \\to \mathcal{X}`

    using a Newton-type update scheme with approximate Hessian.

    The algorithm is described in [Bro1965]_ and [Kva1991]_, and in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Broyden's_method>`_

    Parameters
    ----------
    grad : `Operator`
        Gradient mapping of the objective function, i.e. the mapping
        :math:`x \mapsto \\nabla f(x) \\in \mathcal{X}`
    x : `element` of the domain of ``grad``
        Starting point of the iteration
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : `int`, optional
        Number of iterations
    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate
    """
    # TODO: potentially make the implementation faster by considering
    # performance optimization according to Kvaalen.

    hess = IdentityOperator(grad.range)
    grad_x = grad(x)

    for _ in range(niter):
        search_dir = hess(grad_x)
        step = line_search(x, search_dir, grad_x)

        x_update = search_dir  # Just rename, still writing to search_dir
        x_update *= step
        x += x_update

        grad_x, grad_diff = grad(x), grad_x
        # grad_diff = grad(x) - grad(x_old)
        grad_diff.space.lincomb(-1, grad_diff, 1, grad_x, out=grad_diff)

        grad_diff_norm2 = grad_diff.norm() ** 2
        if grad_diff_norm2 == 0:
            return
        u = (x_update + hess(grad_diff)) / grad_diff_norm2
        hess -= u * grad_diff.T

        if callback is not None:
            callback(x)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
