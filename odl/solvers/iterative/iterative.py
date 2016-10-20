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

"""Simple iterative type optimization schemes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.operator import IdentityOperator, OperatorComp, OperatorSum


__all__ = ('landweber', 'conjugate_gradient', 'conjugate_gradient_normal',
           'gauss_newton')


# TODO: update all docs


def landweber(op, x, rhs, niter=1, omega=1, projection=None, callback=None):
    """Optimized implementation of Landweber's method.

    This method calculates an approximate least-squares solution of
    the inverse problem of the first kind

        :math:`\mathcal{A} (x) = y`,

    for a given :math:`y\\in \mathcal{Y}`, i.e. an approximate
    solution :math:`x^*` to

        :math:`\min_{x\\in \mathcal{X}}
        \\lVert \mathcal{A}(x) - y \\rVert_{\mathcal{Y}}^2`

    for a (Frechet-) differentiable operator
    :math:`\mathcal{A}: \mathcal{X} \\to \mathcal{Y}` between Hilbert
    spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`. The method
    starts from an initial guess :math:`x_0` and uses the
    iteration

    :math:`x_{k+1} = x_k -
    \omega \ \partial \mathcal{A}(x)^* (\mathcal{A}(x_k) - y)`,

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

    Parameters
    ----------
    op : `Operator`
        Operator in the inverse problem. It must have a `Operator.derivative`
        property, which returns a new operator which in turn has an
        `Operator.adjoint` property, i.e. ``op.derivative(x).adjoint`` must be
        well-defined for ``x`` in the operator domain.
    x : ``op.domain`` element
        Element to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : ``op.range`` element
        Right-hand side of the equation defining the inverse problem
    niter : int, optional
        Maximum number of iterations
    omega : positive float, optional
        Relaxation parameter in the iteration
    projection : callable, optional
        Function that can be used to modify the iterates in each iteration,
        for example enforcing positivity. The function should take one
        argument and modify it in-place.
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    None
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


def conjugate_gradient(op, x, rhs, niter=1, callback=None):
    """Optimized implementation of CG for self-adjoint operators.

    This method solves the inverse problem (of the first kind)

    :math:`A x = y`

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
        Right-hand side of the equation defining the inverse problem
    niter : int, optional
        Maximum number of iterations
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    None

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
    """Optimized implementation of CG for the normal equation.

    This method solves the normal equation

    :math:`A^* A x = A^* y`

    to the inverse problem (of the first kind)

    :math:`A x = y`

    with a linear `Operator` ``A``.

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
    niter : int, optional
        Maximum number of iterations
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    None

    See Also
    --------
    conjugate_gradient : Optimized solver for symmetric matrices
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
        Base of the sequence. Its absolute value must be larger than
        1.

    Yields
    ------
    val : float
        The next value in the exponential sequence
    """
    value = 1.0
    while True:
        value /= base
        yield value


def gauss_newton(op, x, rhs, niter=1, zero_seq=exp_zero_seq(2.0),
                 callback=None):
    """Optimized implementation of a Gauss-Newton method.

    This method solves the inverse problem (of the first kind)

    :math:`A (x) = y`

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
    niter : int, optional
        Maximum number of iterations
    zero_seq : iterable, optional
        Zero sequence whose values are used for the regularization of
        the linearized problem in each Newton step
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    None
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

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
