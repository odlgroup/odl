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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str, super

# External module imports
import numpy as np
import pytest

# Internal module imports
import odl
from odl.util.testutils import all_almost_equal


class ResidualOp(odl.Operator):
    """Calculate op(x) - rhs."""

    def __init__(self, op, rhs):
        super().__init__(op.domain, op.range, linear=False)
        self.op = op
        self.rhs = rhs.copy()

    def _apply(self, x, out):
        self.op(x, out)
        out -= self.rhs

    def derivative(self, x):
        return self.op.derivative(x)


# Example problem of the form min f(x) = 0, for f(x) the non-convex
# Rosenbrock function often used in optimization (see e.g.
# https://en.wikipedia.org/wiki/Rosenbrock_function). In this case a = 1 and
# b = 100, giving a globally optimal solution (a,a**2) = (1,1). """


def rosenbrock_function(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


class RosenbrockDerivOp(odl.Operator):
    def __init__(self):
        dom = ran = odl.Rn(2)
        super().__init__(domain=dom, range=ran)

    def _apply(self, x, out):
        out[:] = [-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
                  200 * (x[1] - x[0] ** 2)]

    def derivative(self, x):
        matrix = np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]],
                           [-400 * x[0], 200]])
        return odl.MatVecOperator(self.domain, self.range, matrix)


def test_newton_solver_quadratic():
    # Test for Newton's method on a QP-problem of dimension 3

    # Fixed matrix
    H = np.array([[3, 1, 1],
                  [1, 2, 0.5],
                  [1, 0.5, 5]])

    # Vector representation
    n = H.shape[0]
    rn = odl.Rn(n)
    xvec = rn.element(1)
    c = rn.element(np.random.rand(n))

    # Optimal solution, found by solving 0 = gradf(x) = Hx + c
    x_opt = np.linalg.solve(H, -c)

    # Create derivative operator operator
    Aop = odl.MatVecOperator(rn, rn, H)
    deriv_op = ResidualOp(Aop, -c)

    # Create line search object
    line_search = odl.solvers.BacktrackingLineSearch(
        lambda x: x.inner(Aop(x) / 2.0 + c), 0.5, 0.05, 10)

    # Solve using Newton's method
    odl.solvers.newtons_method(deriv_op, xvec, line_search, num_iter=20,
                               cg_iter=3)

    assert all_almost_equal(xvec, x_opt, places=6)


def test_newton_solver_rosenbrock():
    # Test of Newton's method by minimizing Rosenbrock function in two
    # dimensions

    # Creating the vector space
    n = 2
    rn = odl.Rn(n)

    # The optimal solution to the problem
    x_opt = np.array([1, 1])

    # Create derivative operator and line search object
    ros_deriv_op = RosenbrockDerivOp()
    line_search = odl.solvers.BacktrackingLineSearch(
        rosenbrock_function, 0.5, 0.05, 10)

    # Initial guess
    x = rn.element([-1, 1])

    # Solving the problem
    odl.solvers.newtons_method(ros_deriv_op, x, line_search, num_iter=20)

    assert all_almost_equal(x, x_opt, places=6)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
