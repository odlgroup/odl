# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Simple iterative type optimization schemes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.operator import IdentityOperator, OperatorComp, OperatorSum
from odl.util import normalized_scalar_param_list
from odl.solvers.util import BacktrackingLineSearch


__all__ = ('landweber', 'conjugate_gradient', 'conjugate_gradient_normal',
           'conjugate_gradient_nonlinear', 'gauss_newton', 'kaczmarz')


# TODO: update all docs


def landweber(op, x, rhs, niter, omega=1, projection=None, callback=None):
    """Optimized implementation of Landweber's method.

    Solves the inverse problem::

        A(x) = rhs

    Parameters
    ----------
    op : `Operator`
        Operator in the inverse problem. ``op.derivative(x).adjoint`` must be
        well-defined for ``x`` in the operator domain.
    x : ``op.domain`` element
        Element to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : ``op.range`` element
        Right-hand side of the equation defining the inverse problem.
    niter : int
        Number of iterations.
    omega : positive float, optional
        Relaxation parameter in the iteration.
    projection : callable, optional
        Function that can be used to modify the iterates in each iteration,
        for example enforcing positivity. The function should take one
        argument and modify it in-place.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.

    Notes
    -----
    This method calculates an approximate least-squares solution of
    the inverse problem of the first kind

    .. math::
        \mathcal{A} (x) = y,

    for a given :math:`y\\in \mathcal{Y}`, i.e. an approximate
    solution :math:`x^*` to

    .. math::
        \min_{x\\in \mathcal{X}} \| \mathcal{A}(x) - y \|_{\mathcal{Y}}^2

    for a (Frechet-) differentiable operator
    :math:`\mathcal{A}: \mathcal{X} \\to \mathcal{Y}` between Hilbert
    spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`. The method
    starts from an initial guess :math:`x_0` and uses the
    iteration

    .. math::
        x_{k+1} = x_k -
                  \omega \ \partial \mathcal{A}(x)^* (\mathcal{A}(x_k) - y),

    where :math:`\partial \mathcal{A}(x)` is the Frechet derivativ
    of :math:`\mathcal{A}` at :math:`x` and :math:`\omega` is a
    relaxation parameter. For linear problems, a choice
    :math:`0 < \omega < 2/\\lVert \mathcal{A}^2\\rVert` guarantees
    convergence, where :math:`\\lVert\mathcal{A}\\rVert` stands for the
    operator norm of :math:`\mathcal{A}`.

    Users may also optionally provide a projection to project each
    iterate onto some subset. For example enforcing positivity.

    This implementation uses a minimum amount of memory copies by
    applying re-usable temporaries and in-place evaluation.

    The method is also described in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Landweber_iteration>`_.
    """
    # TODO: add a book reference

    if x not in op.domain:
        raise TypeError('`x` {!r} is not in the domain of `op` {!r}'
                        ''.format(x, op.domain))

    # Reusable temporaries
    tmp_ran = op.range.element()
    tmp_dom = op.domain.element()

    for _ in range(niter):
        op(x, out=tmp_ran)
        tmp_ran -= rhs
        op.derivative(x).adjoint(tmp_ran, out=tmp_dom)
        x.lincomb(1, x, -omega, tmp_dom)

        if projection is not None:
            projection(x)

        if callback is not None:
            callback(x)


def conjugate_gradient(op, x, rhs, niter, callback=None):
    """Optimized implementation of CG for self-adjoint operators.

    This method solves the inverse problem (of the first kind)::

        A(x) = y

    for a linear and self-adjoint `Operator` ``A``.

    It uses a minimum amount of memory copies by applying re-usable
    temporaries and in-place evaluation.

    The method is described (for linear systems) in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.

    Parameters
    ----------
    op : linear `Operator`
        Operator in the inverse problem. It must be linear and
        self-adjoint. This implies in particular that its domain and
        range are equal.
    x : ``op.domain`` element
        Element to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : ``op.range`` element
        Right-hand side of the equation defining the inverse problem.
    niter : int
        Number of iterations.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.

    See Also
    --------
    conjugate_gradient_normal : Solver for nonsymmetric matrices
    """
    # TODO: add a book reference
    # TODO: update doc

    if op.domain != op.range:
        raise ValueError('operator needs to be self-adjoint')

    if x not in op.domain:
        raise TypeError('`x` {!r} is not in the domain of `op` {!r}'
                        ''.format(x, op.domain))

    r = op(x)
    r.lincomb(1, rhs, -1, r)       # r = rhs - A x
    p = r.copy()
    d = op.domain.element()  # Extra storage for storing A x

    sqnorm_r_old = r.norm() ** 2  # Only recalculate norm after update

    if sqnorm_r_old == 0:  # Return if no step forward
        return

    for _ in range(niter):
        op(p, out=d)  # d = A p

        inner_p_d = p.inner(d)

        if inner_p_d == 0.0:  # Return if step is 0
            return

        alpha = sqnorm_r_old / inner_p_d

        x.lincomb(1, x, alpha, p)            # x = x + alpha*p
        r.lincomb(1, r, -alpha, d)           # r = r - alpha*d

        sqnorm_r_new = r.norm() ** 2

        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new

        p.lincomb(1, r, beta, p)                       # p = s + b * p

        if callback is not None:
            callback(x)


def conjugate_gradient_normal(op, x, rhs, niter=1, callback=None):
    """Optimized implementation of CG for the normal equations.

    This method solves the inverse problem (of the first kind)

    ``A(x) == rhs``

    with a linear `Operator` ``A`` by looking at the normal equations

    ``A.adjoint(A(x)) == A.adjoint(rhs)``

    It uses a minimum amount of memory copies by applying re-usable
    temporaries and in-place evaluation.

    The method is described (for linear systems) in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Conjugate_gradient_method#\
Conjugate_gradient_on_the_normal_equations>`_.

    Parameters
    ----------
    op : `Operator`
        Operator in the inverse problem. If not linear, it must have
        an implementation of `Operator.derivative`, which
        in turn must implement `Operator.adjoint`, i.e.
        the call ``op.derivative(x).adjoint`` must be valid.
    x : ``op.domain`` element
        Element to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : ``op.range`` element
        Right-hand side of the equation defining the inverse problem
    niter : int
        Number of iterations.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.

    See Also
    --------
    conjugate_gradient : Optimized solver for symmetric matrices
    conjugate_gradient_nonlinear : Equivalent solver but for nonlinear case
    conjugate_gradient_normal : Equivalent solver but for nonlinear case
    """
    # TODO: add a book reference
    # TODO: update doc

    if x not in op.domain:
        raise TypeError('`x` {!r} is not in the domain of `op` {!r}'
                        ''.format(x, op.domain))

    d = op(x)
    d.lincomb(1, rhs, -1, d)               # d = rhs - A x
    p = op.derivative(x).adjoint(d)
    s = p.copy()
    q = op.range.element()
    sqnorm_s_old = s.norm() ** 2  # Only recalculate norm after update

    for _ in range(niter):
        op(p, out=q)                       # q = A p
        sqnorm_q = q.norm() ** 2
        if sqnorm_q == 0.0:  # Return if residual is 0
            return

        a = sqnorm_s_old / sqnorm_q
        x.lincomb(1, x, a, p)               # x = x + a*p
        d.lincomb(1, d, -a, q)              # d = d - a*Ap
        op.derivative(p).adjoint(d, out=s)  # s = A^T d

        sqnorm_s_new = s.norm() ** 2
        b = sqnorm_s_new / sqnorm_s_old
        sqnorm_s_old = sqnorm_s_new

        p.lincomb(1, s, b, p)               # p = s + b * p

        if callback is not None:
            callback(x)


def conjugate_gradient_nonlinear(f, x, rhs, niter=1, nreset=0,
                                 line_search=1.0, tol=1e-16, beta_method='FR',
                                 callback=None):
    """Conjugate gradient for nonlinear problems.

    The method is described in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method>`_.

    Parameters
    ----------
    op : `Functional`
        Operator in the inverse problem. If not linear, it must have
        an implementation of `Operator.derivative`, which
        in turn must implement `Operator.adjoint`, i.e.
        the call ``op.derivative(x).adjoint`` must be valid.
    x : ``op.domain`` element
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : ``op.range`` element
        Right-hand side of the equation defining the inverse problem
    niter : int
        Number of iterations per reset.
    nreset : int, optional
        Number of times the solver should be reset. Default: no reset.
    line_search : float or `LineSearch`, optional
        Strategy to choose the step length. If a float is given, uses it as a
        fixed step length. Default: `BacktrackingLineSearch`
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    beta_method : {'FR', 'PR', 'HS', 'DY'}
        Method to calculate ``beta`` in the iterates. TODO
    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate

    See Also
    --------
    conjugate_gradient : Optimized solver for linear and symmetric case
    conjugate_gradient_normal : Equivalent solver but for linear case
    """
    # TODO: add a book reference
    # TODO: update doc

    if x not in f.domain:
        raise TypeError('`x` {!r} is not in the domain of `f` {!r}'
                        ''.format(x, f.domain))

    if not callable(line_search):
        line_search = BacktrackingLineSearch(f, discount=0.05)

    for rest_nr in range(nreset + 1):
        dx = -f.gradient(x)
        dir_derivative = -dx.inner(dx)
        if abs(dir_derivative) < tol:
            return
        a = line_search(x, dx, dir_derivative)
        x.lincomb(1, x, a, dx)  # x = x + a * dx

        dx_old = dx
        s = dx  # for 'HS' and 'DY' beta methods

        for _ in range(niter):
            dx, dx_old = -f.gradient(x), dx

            if beta_method == 'FR':
                beta = dx.inner(dx) / dx_old.inner(dx_old)
            elif beta_method == 'PR':
                beta = dx.inner(dx - dx_old) / dx_old.inner(dx_old)
            elif beta_method == 'HS':
                beta = dx.inner(dx - dx_old) / s.inner(dx - dx_old)
            elif beta_method == 'DY':
                beta = dx.inner(dx) / s.inner(dx - dx_old)
            else:
                raise ValueError('unknown ``beta_method``')

            if abs(beta) < tol:
                return

            s = dx + beta * s
            dir_derivative = -s.inner(s)
            if abs(dir_derivative) < tol:
                return
            a = line_search(x, s, dir_derivative)
            x.lincomb(1, x, a, s)  # x = x + a * s

            if callback is not None:
                callback(x)


def exp_zero_seq(base):
    """Default exponential zero sequence.

    It is defined by

        t_0 = 1.0
        t_m = t_(m-1) / base

    or, in closed form

        t_m = base^(-m-1)

    Parameters
    ----------
    base : float
        Base of the sequence. Its absolute value must be larger than 1.

    Yields
    ------
    val : float
        The next value in the exponential sequence.
    """
    value = 1.0
    while True:
        value /= base
        yield value


def gauss_newton(op, x, rhs, niter, zero_seq=exp_zero_seq(2.0),
                 callback=None):
    """Optimized implementation of a Gauss-Newton method.

    This method solves the inverse problem (of the first kind)::

        A(x) = y

    for a (Frechet-) differentiable `Operator` ``A`` using a
    Gauss-Newton iteration.

    It uses a minimum amount of memory copies by applying re-usable
    temporaries and in-place evaluation.

    A variant of the method applied to a specific problem is described
    in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>`_.

    Parameters
    ----------
    op : `Operator`
        Operator in the inverse problem. If not linear, it must have
        an implementation of `Operator.derivative`, which
        in turn must implement `Operator.adjoint`, i.e.
        the call ``op.derivative(x).adjoint`` must be valid.
    x : ``op.domain`` element
        Element to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : ``op.range`` element
        Right-hand side of the equation defining the inverse problem
    niter : int
        Maximum number of iterations.
    zero_seq : iterable, optional
        Zero sequence whose values are used for the regularization of
        the linearized problem in each Newton step.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.
    """
    if x not in op.domain:
        raise TypeError('`x` {!r} is not in the domain of `op` {!r}'
                        ''.format(x, op.domain))

    x0 = x.copy()
    id_op = IdentityOperator(op.domain)
    dx = op.domain.zero()

    tmp_dom = op.domain.element()
    u = op.domain.element()
    tmp_ran = op.range.element()
    v = op.range.element()

    for _ in range(niter):
        tm = next(zero_seq)
        deriv = op.derivative(x)
        deriv_adjoint = deriv.adjoint

        # v = rhs - op(x) - deriv(x0-x)
        # u = deriv.T(v)
        op(x, out=tmp_ran)              # eval  op(x)
        v.lincomb(1, rhs, -1, tmp_ran)  # assign  v = rhs - op(x)
        tmp_dom.lincomb(1, x0, -1, x)   # assign temp  tmp_dom = x0 - x
        deriv(tmp_dom, out=tmp_ran)     # eval  deriv(x0-x)
        v -= tmp_ran                    # assign  v = rhs-op(x)-deriv(x0-x)
        deriv_adjoint(v, out=u)         # eval/assign  u = deriv.T(v)

        # Solve equation Tikhonov regularized system
        # (deriv.T o deriv + tm * id_op)^-1 u = dx
        tikh_op = OperatorSum(OperatorComp(deriv.adjoint, deriv),
                              tm * id_op, tmp_dom)

        # TODO: allow user to select other method
        conjugate_gradient(tikh_op, dx, u, 3)

        # Update x
        x.lincomb(1, x0, 1, dx)  # x = x0 + dx

        if callback is not None:
            callback(x)


def kaczmarz(ops, x, rhs, niter, omega=1, projection=None,
             callback=None):
    """Optimized implementation of Kaczmarz's method.

    Solves the inverse problem given by the set of equations::

        A_n(x) = rhs_n

    This is also known as the Landweber-Kaczmarz's method, since the method
    coincides with the Landweber method for a single operator.

    Parameters
    ----------
    ops : sequence of `Operator`'s
        Operators in the inverse problem. ``op[i].derivative(x).adjoint`` must
        be well-defined for ``x`` in the operator domain and for all ``i``.
    x : ``op.domain`` element
        Element to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : sequence of ``ops[i].range`` elements
        Right-hand side of the equation defining the inverse problem.
    niter : int
        Number of iterations.
    omega : positive float or sequence of positive floats, optional
        Relaxation parameter in the iteration. If a single float is given the
        same step is used for all operators, otherwise separate steps are used.
    projection : callable, optional
        Function that can be used to modify the iterates in each iteration,
        for example enforcing positivity. The function should take one
        argument and modify it in-place.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.

    Notes
    -----
    This method calculates an approximate least-squares solution of
    the inverse problem of the first kind

    .. math::
        \mathcal{A}_i (x) = y_i \\quad 1 \\leq i \\leq n,

    for a given :math:`y_n \\in \mathcal{Y}_n`, i.e. an approximate
    solution :math:`x^*` to

    .. math::
        \min_{x\\in \mathcal{X}}
        \\sum_{i=1}^n \| \mathcal{A}_i(x) - y_i \|_{\mathcal{Y}_i}^2

    for a (Frechet-) differentiable operator
    :math:`\mathcal{A}: \mathcal{X} \\to \mathcal{Y}` between Hilbert
    spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`. The method
    starts from an initial guess :math:`x_0` and uses the
    iteration

    .. math::
        x_{k+1} = x_k - \omega_{[k]} \ \partial \mathcal{A}_{[k]}(x_k)^*
                                 (\mathcal{A}_{[k]}(x_k) - y_{[k]}),

    where :math:`\partial \mathcal{A}_{[k]}(x_k)` is the Frechet derivative
    of :math:`\mathcal{A}_{[k]}` at :math:`x_k`, :math:`\omega_{[k]}` is a
    relaxation parameter and :math:`[k] := k \\text{ mod } n`.

    For linear problems, a choice
    :math:`0 < \omega_i < 2/\\lVert \mathcal{A}_{i}^2\\rVert` guarantees
    convergence, where :math:`\|\mathcal{A}_{i}\|` stands for the
    operator norm of :math:`\mathcal{A}_{i}`.

    This implementation uses a minimum amount of memory copies by
    applying re-usable temporaries and in-place evaluation.

    The method is also described in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Kaczmarz_method>`_. and in Natterer, F.
    Mathematical Methods in Image Reconstruction, section 5.3.2.

    See Also
    --------
    landweber
    """
    domain = ops[0].domain
    if any(domain != opi.domain for opi in ops):
        raise ValueError('`opi[i].domain` are not all equal')

    if x not in domain:
        raise TypeError('`x` {!r} is not in the domain of `ops` {!r}'
                        ''.format(x, domain))

    if len(ops) != len(rhs):
        raise ValueError('`number of `ops` {} does not match number of '
                         '`rhs` {}'.format(len(ops), len(rhs)))

    omega = normalized_scalar_param_list(omega, len(ops), param_conv=float)

    # Reusable elements in the range, one per type of space
    ranges = [opi.range for opi in ops]
    unique_ranges = set(ranges)
    tmp_rans = {ran: ran.element() for ran in unique_ranges}

    # Single reusable element in the domain
    tmp_dom = domain.element()

    # Iteratively find solution
    for _ in range(niter):
        for i in range(len(ops)):
            # Find residual
            tmp_ran = tmp_rans[ops[i].range]
            ops[i](x, out=tmp_ran)
            tmp_ran -= rhs[i]

            # Update x
            ops[i].derivative(x).adjoint(tmp_ran, out=tmp_dom)
            x.lincomb(1, x, -omega[i], tmp_dom)

            if projection is not None:
                projection(x)

            if callback is not None:
                callback(x)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
