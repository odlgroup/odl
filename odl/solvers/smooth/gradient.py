# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Gradient-based optimization schemes."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.solvers.util import ConstantLineSearch


__all__ = ('steepest_descent', 'adam')


# TODO: update all docs


def steepest_descent(f, x, line_search=1.0, maxiter=1000, tol=1e-16,
                     projection=None, callback=None):
    r"""Steepest descent method to minimize an objective function.

    General implementation of steepest decent (also known as gradient
    decent) for solving

    .. math::
        \min f(x)

    The algorithm is intended for unconstrained problems. It needs line
    search in order guarantee convergence. With appropriate line search,
    it can also be used for constrained problems where one wants to
    minimize over some given set :math:`C`. This can be done by defining
    :math:`f(x) = \infty` for :math:`x\\not\\in C`, or by providing a
    ``projection`` function that projects the iterates on :math:`C`.

    The algorithm is described in [BV2004], section 9.3--9.4
    (`book available online
    <http://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_),
    [GNS2009], Section 12.2, and wikipedia
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

    References
    ----------
    [BV2004] Boyd, S, and Vandenberghe, L. *Convex optimization*.
    Cambridge university press, 2004.

    [GNS2009] Griva, I, Nash, S G, and Sofer, A. *Linear and nonlinear
    optimization*. Siam, 2009.
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


def adam(f, x, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
         maxiter=1000, tol=1e-16, callback=None):
    r"""ADAM method to minimize an objective function.

    General implementation of ADAM for solving

    .. math::
        \min f(x)

    where :math:`f` is a differentiable functional.

    The algorithm is described in [KB2015] (`arxiv
    <https://arxiv.org/abs/1412.6980>`_). All parameter names and default
    valuesare taken from the article.

    Parameters
    ----------
    f : `Functional`
        Goal functional. Needs to have ``f.gradient``.
    x : ``f.domain`` element
        Starting point of the iteration, updated in place.
    learning_rate : positive float, optional
        Step length of the method.
    beta1 : float in [0, 1), optional
        Update rate for first order moment estimate.
    beta2 : float in [0, 1), optional
        Update rate for second order moment estimate.
    eps : positive float, optional
        A small constant for numerical stability.
    maxiter : int, optional
        Maximum number of iterations.
    tol : positive float, optional
        Tolerance that should be used for terminating the iteration.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.

    See Also
    --------
    odl.solvers.smooth.gradient.steepest_descent : Simple gradient descent.
    odl.solvers.iterative.iterative.landweber :
        Optimized solver for the case ``f(x) = ||Ax - b||_2^2``.
    odl.solvers.iterative.iterative.conjugate_gradient :
        Optimized solver for the case ``f(x) = x^T Ax - 2 x^T b``.

    References
    ----------
    [KB2015] Kingma, D P and Ba, J.
    *Adam: A Method for Stochastic Optimization*, ICLR 2015.
    """
    grad = f.gradient
    if x not in grad.domain:
        raise TypeError('`x` {!r} is not in the domain of `grad` {!r}'
                        ''.format(x, grad.domain))

    m = grad.domain.zero()
    v = grad.domain.zero()

    grad_x = grad.range.element()
    for _ in range(maxiter):
        grad(x, out=grad_x)

        if grad_x.norm() < tol:
            return

        m.lincomb(beta1, m, 1 - beta1, grad_x)
        v.lincomb(beta2, v, 1 - beta2, grad_x ** 2)

        step = learning_rate * np.sqrt(1 - beta2) / (1 - beta1)

        x.lincomb(1, x, -step, m / (np.sqrt(v) + eps))

        if callback is not None:
            callback(x)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
