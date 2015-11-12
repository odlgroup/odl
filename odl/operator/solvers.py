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

"""General and optimized equation system solvers in linear spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import next, object, range
from odl.util.utility import with_metaclass

# External
from abc import ABCMeta, abstractmethod
from math import log, ceil

# Internal
from odl.operator.operator import OperatorComp, OperatorSum
from odl.operator.default_ops import IdentityOperator


class Partial(with_metaclass(ABCMeta, object)):

    """Abstract base class for sending partial results of iterations."""

    @abstractmethod
    def send(self, result):
        """Send the result to the partial object."""


class StorePartial(Partial):

    """Simple object for storing all partial results of the solvers."""

    def __init__(self):
        self.results = []

    def send(self, result):
        """Append result to results list."""
        self.results.append(result.copy())

    def __iter__(self):
        return self.results.__iter__()


class ForEachPartial(Partial):

    """Simple object for applying a function to each iterate."""

    def __init__(self, function):
        self.function = function

    def send(self, result):
        """Applies function to result."""
        self.function(result)


class PrintIterationPartial(Partial):

    """Print the interation count."""

    def __init__(self):
        self.iter = 0

    def send(self, _):
        """Print the current iteration."""
        print("iter = {}".format(self.iter))
        self.iter += 1


class PrintStatusPartial(Partial):

    """Print the interation count and current norm of each iterate."""

    def __init__(self):
        self.iter = 0

    def send(self, result):
        """Print the current iteration and norm."""
        print("iter = {}, norm = {}".format(self.iter, result.norm()))
        self.iter += 1


def landweber(op, x, rhs, niter=1, omega=1, partial=None):
    """Optimized implementation of Landweber's method.

    This method solves the inverse problem (of the first kind)

    :math:`A (x) = y`

    for a (Frechet-) differentiable operator `A` using the iteration

    :math:`x <- x - \omega * (A')^* (A(x) - y)`

    It uses a minimum amount of memory copies by applying re-usable
    temporaries and in-place evaluation.

    The method is described in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Landweber_iteration>`_.

    Parameters
    ----------
    op : :class:`Operator`
        Operator in the inverse problem. It must have a `derivative`
        property, which returns a new operator which in turn has an
        `adjoint` property, i.e. `op.derivative(x).adjoint` must be
        well-defined for ``x`` in the operator domain.
    x : element of the domain of ``op``
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : element of the range of ``op``
        Right-hand side of the equation defining the inverse problem
    niter : `int`, optional
        Maximum number of iterations
    omega : positive float
        Relaxation parameter, must lie between 0 and :math:`2/||A||`,
        the operator norm of `A`, to guarantee convergence.
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    None
    """
    # TODO: add a book reference

    # Reusable temporaries
    tmp_ran = op.range.element()
    tmp_dom = op.domain.element()

    for _ in range(niter):
        op(x, out=tmp_ran)
        tmp_ran -= rhs
        op.derivative(x).adjoint(tmp_ran, out=tmp_dom)
        x.lincomb(1, x, -omega, tmp_dom)

        if partial is not None:
            partial.send(x)


def conjugate_gradient(op, x, rhs, niter=1, partial=None):
    """Optimized implementation of CG for self-adjoint operators.

    This method solves the inverse problem (of the first kind)

    :math:`A x = y`

    for a linear and self-adjoint operator `A`.

    It uses a minimum amount of memory copies by applying re-usable
    temporaries and in-place evaluation.

    The method is described (for linear systems) in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.

    Parameters
    ----------
    op : :class:`Operator`
        Operator in the inverse problem. It must be linear and
        self-adjoint. This implies in particular that its domain and
        range are equal.
    x : element of the domain of ``op``
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : element of the range of ``op``
        Right-hand side of the equation defining the inverse problem
    niter : `int`, optional
        Maximum number of iterations
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    None
    """
    # TODO: add a book reference

    if op.domain != op.range:
        raise TypeError('Operator needs to be self adjoint')

    r = op(x)
    r.lincomb(1, rhs, -1, r)       # r = rhs - A x
    p = r.copy()
    Ap = op.domain.element()  # Extra storage for storing A x

    sqnorm_r_old = r.norm()**2  # Only recalculate norm after update

    for _ in range(niter):
        op(p, out=Ap)  # Ap = A p

        alpha = sqnorm_r_old / p.inner(Ap)

        if alpha == 0.0:  # Return if residual is 0
            return

        x.lincomb(1, x, alpha, p)            # x = x + alpha*p
        r.lincomb(1, r, -alpha, Ap)           # r = r - alpha*p

        sqnorm_r_new = r.norm()**2

        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new

        p.lincomb(1, r, beta, p)                       # p = s + b * p

        if partial is not None:
            partial.send(x)


def conjugate_gradient_normal(op, x, rhs, niter=1, partial=None):
    """Optimized implementation of CG for the normal equation.

    This method solves the normal equation

    :math:`A^* A x = A^* y`

    to the inverse problem (of the first kind)

    :math:`A x = y`

    with a linear operator ``A``.

    It uses a minimum amount of memory copies by applying re-usable
    temporaries and in-place evaluation.

    The method is described (for linear systems) in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Conjugate_gradient_method#\
Conjugate_gradient_on_the_normal_equations>`_.

    Parameters
    ----------
    op : :class:`Operator`
        Operator in the inverse problem. It must be linear and implement
        the `adjoint` property.
    x : element of the domain of ``op``
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : element of the range of ``op``
        Right-hand side of the equation defining the inverse problem
    niter : `int`, optional
        Maximum number of iterations
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    None
    """
    # TODO: add a book reference

    d = op(x)
    d.lincomb(1, rhs, -1, d)       # d = rhs - A x
    p = op.derivative(x).adjoint(d)
    s = p.copy()
    q = op.range.element()
    sqnorm_s_old = s.norm()**2  # Only recalculate norm after update

    for _ in range(niter):
        op(p, out=q)  # q = A p
        sqnorm_q = q.norm()**2
        if sqnorm_q == 0.0:  # Return if residual is 0
            return

        a = sqnorm_s_old / sqnorm_q
        x.lincomb(1, x, a, p)                       # x = x + a*p
        d.lincomb(1, d, -a, q)                      # d = d - a*Ap
        op.derivative(p).adjoint(d, out=s)  # s = A^T d

        sqnorm_s_new = s.norm()**2
        b = sqnorm_s_new / sqnorm_s_old
        sqnorm_s_old = sqnorm_s_new

        p.lincomb(1, s, b, p)                       # p = s + b * p

        if partial is not None:
            partial.send(x)


def exp_zero_seq(base):
    """The default exponential zero sequence.

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
                 partial=None):
    """Optimized implementation of a Gauss-Newton method.

    This method solves the inverse problem (of the first kind)

    :math:`A (x) = y`

    for a (Frechet-) differentiable operator `A` using a
    Gauss-Newton iteration.

    It uses a minimum amount of memory copies by applying re-usable
    temporaries and in-place evaluation.

    A variant of the method applied to a specific problem is described
    in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>`_.

    Parameters
    ----------
    op : :class:`Operator`
        Operator in the inverse problem. It must have a `derivative`
        property, which returns a new operator which in turn has an
        `adjoint` property, i.e. `op.derivative(x).adjoint` must be
        well-defined for ``x`` in the operator domain.
    x : element of the domain of ``op``
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    rhs : element of the range of ``op``
        Right-hand side of the equation defining the inverse problem
    niter : `int`, optional
        Maximum number of iterations
    zero_seq : `iterable`, optional
        Zero sequence whose values are used for the regularization of
        the linearized problem in each Newton step
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    None
    """
    x0 = x.copy()
    I = IdentityOperator(op.domain)
    dx = x.space.zero()

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

        # Solve equation system
        # (deriv.T o deriv + tm * I)^-1 u = dx
        A = OperatorSum(OperatorComp(deriv.adjoint, deriv),
                        tm * I, tmp_dom)

        # TODO: allow user to select other method
        conjugate_gradient(A, dx, u, 3)

        # Update x
        x.lincomb(1, x0, 1, dx)  # x = x0 + dx

        if partial is not None:
            partial.send(x)


class LineSearch(object):

    """Base class for line search methods to calculate step lenghts. """

    def __call__(self, x, direction, gradf):
        """
        Parameters
        ----------
        x : domain element
            The current point
        direction : domain element
            Search direction in which the line search should be computed
        dir_derivative : `float`
            Directional derivative along the ``direction``

        Returns
        -------
        alpha : `float`
            The step length
        """
        raise NotImplementedError


class BacktrackingLineSearch(LineSearch):

    """Backtracking line search for step length calculation.

    This methods approximately finds the longest step lenght fulfilling
    the Armijo-Goldstein condition.

    The line search algorithm is described in [1]_, page 464
    (`book available online
    <http://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`_) and
    [2]_, pages 378--379. See also the
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Backtracking_line_search>`_.

    References
    ----------
    .. [1] Boyd, Stephen, and Lieven Vandenberghe. Convex optimization.
       Cambridge university press, 2004. Available at

    .. [2] Pages 378-379 in Griva, Igor, Stephen G. Nash, and
       Ariela Sofer. Linear and nonlinear optimization. Siam, 2009.
    """

    def __init__(self, function, tau=0.5, c=0.01, max_num_iter=None):
        """Initialize a new instance.

        Parameters
        ----------
        function : `callable`
            The cost function of the optimization problem to be solved.
        tau : `float`, optional
            The amount the step length is decreased in each iteration,
            as long as it does not fulfill the decrease condition.
            The step length is updated as step_length *= tau
        c : float, optional
            The 'discount factor' on the
            `step length * direction derivative`,
            which the new point needs to be smaller than in order to
            fulfill the condition and be accepted (see the references).
        max_num_iter : `int`, optional
            Maximum number of iterations allowed each time the line
            search method is called. If not set, this number  is
            calculated to allow a shortest step length of 0.0001.
        """
        self.function = function
        self.tau = tau
        self.c = c
        self.total_num_iter = 0
        # Use a default value that allows the shortest step to be < 0.0001
        # times the original step length
        if max_num_iter is None:
            self.max_num_iter = ceil(log(0.0001/self.tau))
        else:
            self.max_num_iter = max_num_iter

    def __call__(self, x, direction, dir_derivative):
        """Calculate the optimal step length along a line.

        Parameters
        ----------
        x : domain element
            The current point
        direction : domain element
            Search direction in which the line search should be computed
        dir_derivative : float
            Directional derivative along the `direction`

        Returns
        -------
        alpha : float
            The computed step length
        """
        alpha = 1.0
        fx = self.function(x)
        num_iter = 0
        while ((self.function(x + alpha * direction) >=
                fx + alpha * dir_derivative * self.c) and
               num_iter <= self.max_num_iter):
            num_iter += 1
            alpha *= self.tau
        self.total_num_iter += num_iter
        return alpha


class ConstantLineSearch(LineSearch):

    """Line search object that returns a constant step length."""

    def __init__(self, constant):
        """
        Parameters
        ----------
        constant : float
            The constant step length that the 'line search' object should
            return.
        """
        self.constant = constant

    def __call__(self, x, direction, dir_derivative):
        """
        Parameters
        ----------
        x : domain element
            The current point
        direction : domain element
            Search direction in which the line search should be computed
        dir_derivative : float
            Directional derivative along the `direction`

        Returns
        -------
        alpha : float
            The constant step length
        """
        return self.constant


def quasi_newton_bfgs(deriv, x, line_search, niter=1, partial=None):
    """Quasi-Newton method to minimize an objective function.

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
    niter : `int`, optional
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


def steepest_decent(deriv, x, line_search, niter=1, partial=None):
    """Steepest decent method to minimize an objective function.

    General implementation of steepest decent (also known as gradient
    decent) for solving

    :math:`min f(x)`

    The algorithm is intended for unconstrained problems. It needs line
    search in order guarantee convergence. With appropriate line search,
    it can also be used for constrained problems where one wants to
    minimize over some given set `C`. This is done by defining
    :math:`f(x) = \infty` for ``x`` outside `C`.


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
    x : element in the domain of `deriv`
        Starting point of the iteration
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : `int`, optional
        Number of iterations
    partial : `Partial`, optional
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
        dir_derivative = -grad.norm()**2
        step = line_search(x, -grad, dir_derivative)
        x.lincomb(1, x, -step, grad)

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
    x : element of the domain of ``op``
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : `int`, optional
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

    Hi = IdentityOperator(op.range)
    opx = op(x)

    for _ in range(niter):
        p = Hi(opx)
        t = line_search(x, p, opx)

        delta_x = t*p
        x += delta_x

        opx, opx_old = op(x), opx
        delta_f = opx - opx_old

        v = Hi(delta_x)
        v_delta_f = v.inner(delta_f)
        if v_delta_f == 0:
            return
        u = (delta_x + Hi(delta_f))/(v_delta_f)
        Hi -= u * v.T

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
    x : element of the domain of ``op``
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    line_search : `LineSearch`
        Strategy to choose the step length
    niter : `int`, optional
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

    Hi = IdentityOperator(op.range)
    opx = op(x)

    for _ in range(niter):
        p = Hi(opx)
        t = line_search(x, p, opx)

        delta_x = t*p
        x += delta_x

        opx, opx_old = op(x), opx
        delta_f = opx - opx_old

        delta_f_norm2 = delta_f.norm()**2
        if delta_f_norm2 == 0:
            return
        u = (delta_x + Hi(delta_f))/(delta_f_norm2)
        Hi -= u * delta_f.T

        if partial is not None:
            partial.send(x)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
