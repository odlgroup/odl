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

"""Test iterative solvers."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import odl
from odl.util.testutils import all_almost_equal
import pytest
import numpy as np


# Find the valid projectors
@pytest.fixture(scope="module",
                params=['steepest_descent',
                        'landweber',
                        'conjugate_gradient',
                        'conjugate_gradient_normal',
                        'mlem'])
def iterative_solver(request):
    """Return a solver given by a name with interface solve(op, x, rhs)."""
    solver_name = request.param

    if solver_name == 'steepest_descent':
        def solver(op, x, rhs):
            norm2 = op.adjoint(op(x)).norm() / x.norm()
            func = odl.solvers.L2NormSquared(op.domain) * (op - rhs)

            odl.solvers.steepest_descent(func, x, line_search=0.5 / norm2)
    elif solver_name == 'landweber':
        def solver(op, x, rhs):
            norm2 = op.adjoint(op(x)).norm() / x.norm()
            odl.solvers.landweber(op, x, rhs, niter=10, omega=0.5 / norm2)
    elif solver_name == 'conjugate_gradient':
        def solver(op, x, rhs):
            odl.solvers.conjugate_gradient(op, x, rhs, niter=10)
    elif solver_name == 'conjugate_gradient_normal':
        def solver(op, x, rhs):
            odl.solvers.conjugate_gradient_normal(op, x, rhs, niter=10)
    elif solver_name == 'mlem':
        def solver(op, x, rhs):
            odl.solvers.mlem(op, x, rhs, niter=10)
    else:
        raise ValueError('solver not valid')

    return solver


# Define some interesting problems
@pytest.fixture(scope="module",
                params=['MatVec',
                        'Identity'])
def optimization_problem(request):
    problem_name = request.param

    if problem_name == 'MatVec':
        # Define problem
        op_arr = np.eye(5) * 5 + np.ones([5, 5])
        op = odl.MatVecOperator(op_arr)

        # Simple right hand side
        rhs = op.range.one()

        return op, rhs
    elif problem_name == 'Identity':
        # Define problem
        space = odl.uniform_discr(0, 1, 5)
        op = odl.IdentityOperator(space)

        # Simple right hand side
        rhs = op.range.element([0, 0, 1, 0, 0])

        return op, rhs
    else:
        raise ValueError('problem not valid')


def test_solver(optimization_problem, iterative_solver):
    """Test iterative solver for solving some simple problems."""

    # Solve within 1%
    places = 2

    op, rhs = optimization_problem

    # Solve problem
    x = op.domain.one()
    iterative_solver(op, x, rhs)

    # Assert residual is small
    assert all_almost_equal(op(x), rhs, places)


def test_steepst_descent():
    """Test steepest descent on the rosenbrock function in 3d."""

    space = odl.rn(3)
    scale = 1  # only mildly ill-behaved
    rosenbrock = odl.solvers.RosenbrockFunctional(space, scale)

    # Create line search object
    line_search = odl.solvers.BacktrackingLineSearch(
        rosenbrock, 0.1, 0.01)

    # Initial guess
    x = rosenbrock.domain.zero()

    # Solving the problem
    odl.solvers.steepest_descent(rosenbrock, x, maxiter=40,
                                 line_search=line_search)

    assert all_almost_equal(x, [1, 1, 1], places=2)

if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
