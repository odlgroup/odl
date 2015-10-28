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
from math import log, ceil

# ODL imports
from odl.operator.operator import OperatorComp, OperatorSum
from odl.operator.default_ops import IdentityOperator


class StorePartial(object):
    """ Simple object for storing all partial results of the solvers
    """
    def __init__(self):
        self.results = []

    def send(self, result):
        """ append result to results list
        """
        self.results.append(result.copy())

    def __iter__(self):
        return self.results.__iter__()


class ForEachPartial(object):
    """ Simple object for applying a function to each iterate
    """
    def __init__(self, function):
        self.function = function

    def send(self, result):
        """ Applies function to result
        """
        self.function(result)


class PrintIterationPartial(object):
    """ Prints the interation count
    """
    def __init__(self):
        self.iter = 0

    def send(self, _):
        """ Print the current iteration
        """
        print("iter = {}".format(self.iter))
        self.iter += 1


class PrintStatusPartial(object):
    """ Prints the interation count and current norm of each iterate
    """
    def __init__(self):
        self.iter = 0

    def send(self, result):
        """ Print the current iteration and norm
        """
        print("iter = {}, norm = {}".format(self.iter, result.norm()))
        self.iter += 1


def landweber(op, x, rhs, niter=1, omega=1, partial=None):
    """ General and efficient implementation of Landweber iteration

    x <- x - omega * (A')^* (Ax - rhs)
    """

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
    """ Optimized version of CGN, uses no temporaries etc.
    """
    if op.domain != op.range:
        raise TypeError('Operator needs to be self adjoint')
    
    r = op(x)
    r.lincomb(1, rhs, -1, r)       # r = rhs - A x
    p = r.copy()
    Ap = op.domain.element() #Extra storage for storing A x
    
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
    """ Optimized version of CGN, uses no temporaries etc.
    """
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


def exp_zero_seq(scale):
    """ The default zero sequence given by:
        t_m = scale ^ (-m-1)
    """
    value = 1.0
    while True:
        value /= scale
        yield value


def gauss_newton(op, x, rhs, niter=1, zero_seq=exp_zero_seq(2.0),
                 partial=None):
    """ Solves op(x) = rhs using the gauss newton method. The inner-solver
    uses conjugate gradient.
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
        op(x, out=tmp_ran)      # eval        op(x)
        v.lincomb(1, rhs, -1, tmp_ran)  # assign      v = rhs - op(x)
        tmp_dom.lincomb(1, x0, -1, x)  # assign temp  tmp_dom = x0 - x
        deriv(tmp_dom, out=tmp_ran)   # eval        deriv(x0-x)
        v -= tmp_ran                    # assign      v = rhs-op(x)-deriv(x0-x)
        deriv_adjoint(v, out=u)       # eval/assign u = deriv.T(v)

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

class BacktrackingLineSearch(object):
    """ Backtracking line search, 
    a search scheme based on the Armijo-Goldstein condition.
    """
    def __init__(self, function, tau=0.8, c=0.7, max_num_iter=0):
        self.function = function
        self.tau = tau
        self.c = c
        self.total_num_iter = 0
        #If max_num_iter is specified it sets this value, otherwise sets a value that allows the shortest step to be < 0.0001 of original step length
        if max_num_iter == 0:
            self.max_num_iter = ceil(log(0.0001/self.tau))
        else:
            self.max_num_iter = max_num_iter

    def __call__(self, x, direction, gradf):
        alpha = 1.0
        decrease = gradf.inner(direction)
        fx = self.function(x)
        num_iter = 0        
        while self.function(x + alpha * direction) >= fx + alpha * decrease * self.c and num_iter <= self.max_num_iter:
            num_iter += 1
            alpha *= self.tau
        self.total_num_iter += num_iter
        return alpha

class ConstantLineSearch(object):
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, x, direction, gradf):
        return self.constant

def quasi_newton(op, x, line_search, niter=1, partial=None):
    """ General implementation of the quasi newton method for solving

    op(x) == 0
    """
    I = IdentityOperator(op.range)
    Bi = IdentityOperator(op.range)
    # Reusable temporaries
    for _ in range(niter):
        opx = op(x)
        print(opx.norm())
        p = Bi(-opx)
        alpha = line_search(x, p, opx)
        x_old = x.copy()
        s = alpha * p
        x += s
        y = op(x) - op(x_old)
        x_old = x
        ys = y.inner(s)

        if ys == 0.0:
            return

        Bi = (I - s * y.T / ys) * Bi *  (I - y * s.T / ys) + s * s.T / ys

        if partial is not None:
            partial.send(x)

def steepest_decent(deriv_op, x, line_search, niter=1, print_iter_num=None):
    """ General implementation of steepest decent for solving min f(x)
    for x in C, where we define f(x) = infty if x is not in C. Needs to
    be done in the line search.
    """
    
    for _ in range(niter):
        if print_iter_num is not None:
            print('Iteration', _)
        grad = deriv_op(x)
        step = line_search(x, -grad, grad)
        x.lincomb(1, x, -step, grad)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
