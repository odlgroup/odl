# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""(Quasi-)Newton schemes to find zeros of functionals."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.solvers.util import ConstantLineSearch
from odl.solvers.iterative.iterative import conjugate_gradient


__all__ = ('newtons_method', 'bfgs_method', 'broydens_method')


def _bfgs_direction(s, y, x, hessinv_estimate=None):
    """Compute ``Hn^-1(x)`` for the L-BFGS method.

    Parameters
    ----------
    s : sequence of `LinearSpaceElement`
        The ``s`` coefficients in the BFGS update, see Notes.
    y : sequence of `LinearSpaceElement`
        The ``y`` coefficients in the BFGS update, see Notes.
    x : `LinearSpaceElement`
        Point in which to evaluate the product.
    hessinv_estimate : `Operator`, optional
        Initial estimate of the hessian ``H0^-1``.

    Returns
    -------
    r : ``x.space`` element
        The result of ``Hn^-1(x)``.

    Notes
    -----
    :math:`H_n^{-1}` is defined recursively as

    .. math::
        H_{n+1}^{-1} =
        \\left(I - \\frac{ s_n y_n^T}{y_n^T s_n} \\right)
        H_{n}^{-1}
        \\left(I - \\frac{ y_n s_n^T}{y_n^T s_n} \\right) +
        \\frac{s_n s_n^T}{y_n^T \, s_n}

    With :math:`H_0^{-1}` given by ``hess_estimate``.
    """
    assert len(s) == len(y)

    r = x.copy()
    alphas = np.zeros(len(s))
    rhos = np.zeros(len(s))

    for i in reversed(range(len(s))):
        rhos[i] = 1.0 / y[i].inner(s[i])
        alphas[i] = rhos[i] * (s[i].inner(r))
        r.lincomb(1, r, -alphas[i], y[i])

    if hessinv_estimate is not None:
        r = hessinv_estimate(r)

    for i in range(len(s)):
        beta = rhos[i] * (y[i].inner(r))
        r.lincomb(1, r, alphas[i] - beta, s[i])

    return r


def _broydens_direction(s, y, x, hessinv_estimate=None, impl='first'):
    """Compute ``Hn^-1(x)`` for Broydens method.

    Parameters
    ----------
    s : sequence of `LinearSpaceElement`'s'
        The ``s`` coefficients in the Broydens update, see Notes.
    y : sequence of `LinearSpaceElement`'s'
        The ``y`` coefficients in the Broydens update, see Notes.
    x : `LinearSpaceElement`
        Point in which to evaluate the product.
    hessinv_estimate : `Operator`, optional
        Initial estimate of the hessian ``H0^-1``.
    impl : {'first', 'second'}, optional
        The type of Broydens method to use.

    Returns
    -------
    r : ``x.space`` element
        The result of ``Hn^-1(x)``.

    Notes
    -----
    For ``impl = 'first'``, :math:`H_n^{-1}` is defined recursively as

    .. math::
        H_{n+1}^{-1} = \\left(I + s_n y_n^T \\right) H_{n}^{-1}

    and for ``impl = 'second'``:

    .. math::
        H_{n+1}^{-1} = H_{n}^{-1} + s_n y_n^T

    With :math:`H_0^{-1}` given by ``hess_estimate``.
    """
    assert len(s) == len(y)

    if hessinv_estimate is not None:
        r = hessinv_estimate(x)
    else:
        r = x.copy()

    for i in range(len(s)):
        if impl == 'first':
            r.lincomb(1, r, y[i].inner(r), s[i])
        elif impl == 'second':
            r.lincomb(1, r, y[i].inner(x), s[i])
        else:
            raise RuntimeError('unknown `impl`')

    return r


def newtons_method(f, x, line_search=1.0, maxiter=1000, tol=1e-16,
                   cg_iter=None, callback=None):
    """Newton's method for minimizing a functional.

    Notes
    -----
    This is a general and optimized implementation of Newton's method
    for solving the problem:

    .. math::
        \min f(x)

    for a differentiable function
    :math:`f: \mathcal{X}\\to \mathbb{R}` on a Hilbert space
    :math:`\mathcal{X}`. It does so by finding a zero of the gradient

    .. math::
        \\nabla f: \mathcal{X} \\to \mathcal{X}.

    of finding a root of a function.

    The algorithm is well-known and there is a vast literature about it.
    Among others, the method is described in [BV2004], Sections 9.5
    and 10.2 (`book available online
    <http://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_),
    [GNS2009],  Section 2.7 for solving nonlinear equations and Section
    11.3 for its use in minimization, and wikipedia on `Newton's_method
    <https://en.wikipedia.org/wiki/Newton's_method>`_.

    The algorithm works by iteratively solving

    .. math::
        \partial f(x_k)p_k = -f(x_k)

    and then updating as

    .. math::
        x_{k+1} = x_k + \\alpha x_k,

    where :math:`\\alpha` is a suitable step length (see the
    references). In this implementation the system of equations are
    solved using the conjugate gradient method.

    Parameters
    ----------
    f : `Functional`
        Goal functional. Needs to have ``f.gradient`` and
        ``f.gradient.derivative``.
    x : ``op.domain`` element
        Starting point of the iteration
    line_search : float or `LineSearch`, optional
        Strategy to choose the step length. If a float is given, uses it as a
        fixed step length.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    cg_iter : int, optional
        Number of iterations in the the conjugate gradient solver,
        for computing the search direction.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate

    References
    ----------
    [BV2004] Boyd, S, and Vandenberghe, L. *Convex optimization*.
    Cambridge university press, 2004.

    [GNS2009] Griva, I, Nash, S G, and Sofer, A. *Linear and nonlinear
    optimization*. Siam, 2009.
    """
    # TODO: update doc
    grad = f.gradient
    if x not in grad.domain:
        raise TypeError('`x` {!r} is not in the domain of `f` {!r}'
                        ''.format(x, grad.domain))

    if not callable(line_search):
        line_search = ConstantLineSearch(line_search)

    if cg_iter is None:
        # Motivated by that if it is Ax = b, x and b in Rn, it takes at most n
        # iterations to solve with cg
        cg_iter = grad.domain.size

    # TODO: optimize by using lincomb and avoiding to create copies
    for _ in range(maxiter):

        # Initialize the search direction to 0
        search_direction = x.space.zero()

        # Compute hessian (as operator) and gradient in the current point
        hessian = grad.derivative(x)
        deriv_in_point = grad(x)

        # Solving A*x = b for x, in this case f''(x)*p = -f'(x)
        # TODO: Let the user provide/choose method for how to solve this?
        try:
            hessian_inverse = hessian.inverse
        except NotImplementedError:
            conjugate_gradient(hessian, search_direction,
                               -deriv_in_point, cg_iter)
        else:
            hessian_inverse(-deriv_in_point, out=search_direction)

        # Computing step length
        dir_deriv = search_direction.inner(deriv_in_point)
        if np.abs(dir_deriv) <= tol:
            return

        step_length = line_search(x, search_direction, dir_deriv)

        # Updating
        x += step_length * search_direction

        if callback is not None:
            callback(x)


def bfgs_method(f, x, line_search=1.0, maxiter=1000, tol=1e-15, num_store=None,
                hessinv_estimate=None, callback=None):
    """Quasi-Newton BFGS method to minimize a differentiable function.

    Can use either the regular BFGS method, or the limited memory BFGS method.

    Notes
    -----
    This is a general and optimized implementation of a quasi-Newton
    method with BFGS update for solving a general unconstrained
    optimization problem

    .. math::
        \min f(x)

    for a differentiable function
    :math:`f: \mathcal{X}\\to \mathbb{R}` on a Hilbert space
    :math:`\mathcal{X}`. It does so by finding a zero of the gradient

    .. math::
        \\nabla f: \mathcal{X} \\to \mathcal{X}.

    The QN method is an approximate Newton method, where the Hessian
    is approximated and gradually updated in each step. This
    implementation uses the rank-one BFGS update schema where the
    inverse of the Hessian is recalculated in each iteration.

    The algorithm is described in [GNS2009], Section 12.3 and in the
    `BFGS Wikipedia article
    <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93\
Goldfarb%E2%80%93Shanno_algorithm>`_

    Parameters
    ----------
    f : `Functional`
        Functional with ``f.gradient``.
    x : ``f.domain`` element
        Starting point of the iteration
    line_search : float or `LineSearch`, optional
        Strategy to choose the step length. If a float is given, uses it as a
        fixed step length.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    num_store : int, optional
        Maximum number of correction factors to store. For ``None``, the method
        is the regular BFGS method. For an integer, the method becomes the
        Limited Memory BFGS method.
    hessinv_estimate : `Operator`, optional
        Initial estimate of the inverse of the Hessian operator. Needs to be an
        operator from ``f.domain`` to ``f.domain``.
        Default: Identity on ``f.domain``
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.

    References
    ----------
    [GNS2009] Griva, I, Nash, S G, and Sofer, A. *Linear and nonlinear
    optimization*. Siam, 2009.
    """
    grad = f.gradient
    if x not in grad.domain:
        raise TypeError('`x` {!r} is not in the domain of `grad` {!r}'
                        ''.format(x, grad.domain))

    if not callable(line_search):
        line_search = ConstantLineSearch(line_search)

    ys = []
    ss = []

    grad_x = grad(x)
    for i in range(maxiter):
        # Determine a stepsize using line search
        search_dir = -_bfgs_direction(ss, ys, grad_x, hessinv_estimate)
        dir_deriv = search_dir.inner(grad_x)
        if np.abs(dir_deriv) == 0:
            return  # we found an optimum
        step = line_search(x, direction=search_dir, dir_derivative=dir_deriv)

        # Update x
        x_update = search_dir
        x_update *= step
        x += x_update

        grad_x, grad_diff = grad(x), grad_x
        # grad_diff = grad(x) - grad(x_old)
        grad_diff.lincomb(-1, grad_diff, 1, grad_x)

        y_inner_s = grad_diff.inner(x_update)

        # Test for convergence
        if np.abs(y_inner_s) < tol:
            if grad_x.norm() < tol:
                return
            else:
                # Reset if needed
                ys = []
                ss = []
                continue

        # Update Hessian
        ys.append(grad_diff)
        ss.append(x_update)
        if num_store is not None:
            # Throw away factors if they are too many.
            ss = ss[-num_store:]
            ys = ys[-num_store:]

        if callback is not None:
            callback(x)
    if callback is not None:
        callback.final()


def broydens_method(f, x, line_search=1.0, impl='first', maxiter=1000,
                    tol=1e-15, hessinv_estimate=None,
                    callback=None):
    """Broyden's first method, a quasi-Newton scheme.

    Notes
    -----
    This is a general and optimized implementation of Broyden's method,
    a quasi-Newton method for solving a general unconstrained optimization
    problem

    .. math::
        \min f(x)

    for a differentiable function
    :math:`f: \mathcal{X}\\to \mathbb{R}` on a Hilbert space
    :math:`\mathcal{X}`. It does so by finding a zero of the gradient

    .. math::
        \\nabla f: \mathcal{X} \\to \mathcal{X}

    using a Newton-type update scheme with approximate Hessian.

    The algorithm is described in [Bro1965] and [Kva1991], and in a
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
    impl : {'first', 'second'}, optional
        What version of Broydens method to use. First is also known as Broydens
        'good' method, while the second is known as Broydens 'bad' method.
    maxiter : int, optional
        Maximum number of iterations.
        ``tol``.
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    hessinv_estimate : `Operator`, optional
        Initial estimate of the inverse of the Hessian operator. Needs to be an
        operator from ``f.domain`` to ``f.domain``.
        Default: Identity on ``f.domain``
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.

    References
    ----------
    [Bro1965] Broyden, C G. *A class of methods for solving nonlinear
    simultaneous equations*. Mathematics of computation, 33 (1965),
    pp 577--593.

    [Kva1991] Kvaalen, E. *A faster Broyden method*. BIT Numerical
    Mathematics 31 (1991), pp 369--372.
    """
    grad = f.gradient
    if x not in grad.domain:
        raise TypeError('`x` {!r} is not in the domain of `grad` {!r}'
                        ''.format(x, grad.domain))

    if not callable(line_search):
        line_search = ConstantLineSearch(line_search)

    ss = []
    ys = []

    grad_x = grad(x)
    for i in range(maxiter):
        # find step size
        search_dir = -_broydens_direction(ss, ys, grad_x,
                                          hessinv_estimate, impl)
        dir_deriv = search_dir.inner(grad_x)
        if np.abs(dir_deriv) == 0:
            return  # we found an optimum

        step = line_search(x, search_dir, dir_deriv)

        # update x
        x_update = step * search_dir
        x += x_update

        # compute new gradient
        grad_x, grad_x_old = grad(x), grad_x
        delta_grad = grad_x - grad_x_old

        # update hessian.
        # TODO: reuse from above
        v = _broydens_direction(ss, ys, delta_grad, hessinv_estimate,
                                impl)
        if impl == 'first':
            divisor = x_update.inner(v)

            # Test for convergence
            if np.abs(divisor) < tol:
                if grad_x.norm() < tol:
                    return
                else:
                    # Reset if needed
                    ys = []
                    ss = []
                    continue
            u = (x_update - v) / divisor
            ss.append(u)
            ys.append(x_update)
        elif impl == 'second':
            divisor = delta_grad.inner(delta_grad)

            # Test for convergence
            if np.abs(divisor) < tol:
                if grad_x.norm() < tol:
                    return
                else:
                    # Reset if needed
                    ys = []
                    ss = []
                    continue
            u = (x_update - v) / divisor
            ss.append(u)
            ys.append(delta_grad)

        if callback is not None:
            callback(x)
    if callback is not None:
        callback.final()


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
