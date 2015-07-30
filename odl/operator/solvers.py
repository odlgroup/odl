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

"""
General and optimized equation system solvers in linear spaces.
"""

# Imports for common Python 2/3 codebase
from __future__ import (division, print_function, unicode_literals,
                        absolute_import)
from builtins import object, next, range
from future import standard_library

# ODL imports
from odl.operator.operator import LinearOperatorComposition, LinearOperatorSum
from odl.operator.default import IdentityOperator

standard_library.install_aliases()


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


def landweber(operator, x, rhs, iterations=1, omega=1, part_results=None):
    """ General and efficient implementation of Landweber iteration

    x <- x - omega * (A')^* (Ax - rhs)
    """

    # Reusable temporaries
    tmp_ran = operator.range.element()
    tmp_dom = operator.domain.element()

    for _ in range(iterations):
        operator.apply(x, tmp_ran)
        tmp_ran -= rhs
        operator.derivative(x).adjoint.apply(tmp_ran, tmp_dom)
        x.lincomb(1, x, -omega, tmp_dom)

        if part_results is not None:
            part_results.send(x)


def conjugate_gradient(operator, x, rhs, iterations=1, part_results=None):
    """ Optimized version of CGN, uses no temporaries etc.
    """
    d = operator(x)
    d.lincomb(1, rhs, -1, d)       # d = rhs - A x
    p = operator.T(d)
    s = p.copy()
    q = operator.range.element()
    norms_old = s.norm()**2           # Only recalculate norm after update

    for _ in range(iterations):
        operator.apply(p, q)                        # q = A p
        qnorm = q.norm()**2
        if qnorm == 0.0:  # Return if residual is 0
            return

        a = norms_old / qnorm
        x.lincomb(1, x, a, p)                       # x = x + a*p
        d.lincomb(1, d, -a, q)                      # d = d - a*q
        operator.derivative(p).adjoint.apply(d, s)  # s = A^T d

        norms_new = s.norm()**2
        b = norms_new/norms_old
        norms_old = norms_new

        p.lincomb(1, s, b, p)                       # p = s + b * p

        if part_results is not None:
            part_results.send(x)


def exp_zero_seq(scale):
    """ The default zero sequence given by:
        t_m = scale ^ (-m-1)
    """
    value = 1.0
    while True:
        value /= scale
        yield value


def gauss_newton(operator, x, rhs, iterations=1, zero_seq=exp_zero_seq(2.0),
                 part_results=None):
    """ Solves op(x) = rhs using the gauss newton method. The inner-solver
    uses conjugate gradient.
    """
    x0 = x.copy()
    I = IdentityOperator(operator.domain)
    dx = x.space.zero()

    tmp_dom = operator.domain.element()
    u = operator.domain.element()
    tmp_ran = operator.range.element()
    v = operator.range.element()

    for _ in range(iterations):
        tm = next(zero_seq)
        deriv = operator.derivative(x)
        deriv_adjoint = deriv.adjoint

        # v = rhs - op(x) - deriv(x0-x)
        # u = deriv.T(v)
        operator.apply(x, tmp_ran)      # eval        op(x)
        v.lincomb(1, rhs, -1, tmp_ran)  # assign      v = rhs - op(x)
        tmp_dom.lincomb(1, x0, -1, x)  # assign temp  tmp_dom = x0 - x
        deriv.apply(tmp_dom, tmp_ran)   # eval        deriv(x0-x)
        v -= tmp_ran                    # assign      v = rhs-op(x)-deriv(x0-x)
        deriv_adjoint.apply(v, u)       # eval/assign u = deriv.T(v)

        # Solve equation system
        # (deriv.T o deriv + tm * I)^-1 u = dx
        A = LinearOperatorSum(LinearOperatorComposition(deriv.T, deriv),
                              tm * I, tmp_dom)

        # TODO allow user to select other method
        conjugate_gradient(A, dx, u, 3)

        # Update x
        x.lincomb(1, x0, 1, dx)  # x = x0 + dx

        if part_results is not None:
            part_results.send(x)
