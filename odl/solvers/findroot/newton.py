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

"""(Quasi-)Newton schemes to find zeros of functions (gradients)."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External

# Internal
from odl.operator.default_ops import IdentityOperator

__all__ = ('bfgs_method', 'broydens_first_method', 'broydens_second_method')


# TODO: update all docs


def bfgs_method(deriv, x, line_search, niter=1, partial=None):
    """Quasi-Newton BFGS method to minimize an objective function.

    General implementation of the Quasi-Newton method with BFGS update
    for solving a general optimization problem

    `min f(x)`

    The QN method is an approximate newton method, where the Hessian
    is approximated and gradually updated in each step. This
    implementation uses the rank-one BFGS update schema where the
    inverse of the Hessian is recalculated in each iteration.

    The algorithm is described in [1]_, Section 12.3 and in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93\
Goldfarb%E2%80%93Shanno_algorithm>`_

    Parameters
    ----------
    deriv : `odl.Operator`
        Derivative of the objective function
    x : element in the domain of `grad_f`
        Starting point of the iteration
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : int, optional
        Number of iterations
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    References
    ----------
    .. [1] Griva, Igor, Stephen G. Nash, and Ariela Sofer. Linear
       and nonlinear optimization. Siam, 2009
    """
    hess = ident = IdentityOperator(deriv.range)
    grad = deriv(x)
    for _ in range(niter):
        search_dir = -hess(grad)
        dir_deriv = search_dir.inner(grad)
        step = line_search(x, direction=search_dir, dir_derivative=dir_deriv)

        update = step * search_dir
        x += update

        grad, grad_old = deriv(x), grad
        grad_update = grad - grad_old

        ys = grad_update.inner(update)
        if ys == 0.0:
            return

        # Update Hessian
        hess = ((ident - update * grad_update.T / ys) *
                hess *
                (ident - grad_update * update.T / ys) +
                update * update.T / ys)

        if partial is not None:
            partial.send(x)


def broydens_first_method(op, x, line_search, niter=1, partial=None):
    """General implementation of Broyden's first (or 'good') method.

    It determines a solution to the equation

    :math:`A(x) = 0`

    for a general (not necessarily differentiable) operator `A`
    using a quasi-Newton approach with approximate Hessian.

    The algorithm is described in [1]_ and [2]_, and in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Broyden's_method>`_.

    Parameters
    ----------
    op : Operator
        Operator for which a zero is computed
    x : element of the domain of `op`
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : int, optional
        Number of iterations
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    References
    ----------
    .. [1] Broyden, Charles G. "A class of methods for solving nonlinear
       simultaneous equations." Mathematics of computation (1965):
       577-593.

    .. [2] Kvaalen, Eric. "A faster Broyden method." BIT Numerical
       Mathematics 31.2 (1991): 369-372.
    """

    # TODO: One Hi call can be removed using linearity

    hess = IdentityOperator(op.range)
    opx = op(x)

    for _ in range(niter):
        p = hess(opx)
        t = line_search(x, p, opx)

        delta_x = t * p
        x += delta_x

        opx, opx_old = op(x), opx
        delta_f = opx - opx_old

        v = hess(delta_x)
        v_delta_f = v.inner(delta_f)
        if v_delta_f == 0:
            return
        u = (delta_x + hess(delta_f)) / (v_delta_f)
        hess -= u * v.T

        if partial is not None:
            partial.send(x)


def broydens_second_method(op, x, line_search, niter=1, partial=None):
    """General implementation of Broyden's second (or 'bad') method.

    It determines a solution to the equation

    :math:`A(x) = 0`

    for a general (not necessarily differentiable) operator `A`
    using a quasi-Newton approach with approximate Hessian.

    The algorithm is described in [1]_ and [2]_, and in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Broyden's_method>`_

    Parameters
    ----------
    op : Operator
        Operator for which a zero is computed
    x : element of the domain of `op`
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : int, optional
        Number of iterations
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    References
    ----------
    .. [1] Broyden, Charles G. "A class of methods for solving nonlinear
       simultaneous equations." Mathematics of computation (1965):
       577-593.

    .. [2] Kvaalen, Eric. "A faster Broyden method." BIT Numerical
       Mathematics 31.2 (1991): 369-372.
    """

    # TODO: potentially make the implementation faster by considering
    # performance optimization according to Kvaalen.

    hess = IdentityOperator(op.range)
    opx = op(x)

    for _ in range(niter):
        p = hess(opx)
        t = line_search(x, p, opx)

        delta_x = t * p
        x += delta_x

        opx, opx_old = op(x), opx
        delta_f = opx - opx_old

        delta_f_norm2 = delta_f.norm() ** 2
        if delta_f_norm2 == 0:
            return
        u = (delta_x + hess(delta_f)) / (delta_f_norm2)
        hess -= u * delta_f.T

        if partial is not None:
            partial.send(x)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
