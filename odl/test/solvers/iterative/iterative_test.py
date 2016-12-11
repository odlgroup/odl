# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test iterative solvers."""

from __future__ import division
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
                        'conjugate_gradient_nonlinear_FR',
                        'conjugate_gradient_nonlinear_PR',
                        'conjugate_gradient_nonlinear_HS',
                        'conjugate_gradient_nonlinear_DY',
                        'mlem',
                        'osmlem',
                        'kaczmarz'])
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
            odl.solvers.landweber(op, x, rhs, niter=50, omega=0.5 / norm2)
    elif solver_name == 'conjugate_gradient':
        def solver(op, x, rhs):
            odl.solvers.conjugate_gradient(op, x, rhs, niter=10)
    elif solver_name == 'conjugate_gradient_normal':
        def solver(op, x, rhs):
            odl.solvers.conjugate_gradient_normal(op, x, rhs, niter=10)
    elif solver_name.startswith('conjugate_gradient_nonlinear'):
        def solver(op, x, rhs):
            beta_method = solver_name.split('_')[-1]
            func = odl.solvers.L2NormSquared(op.domain) * (op - rhs)
            odl.solvers.conjugate_gradient_nonlinear(func, x, rhs, niter=20,
                                                     beta_method=beta_method)
    elif solver_name == 'mlem':
        def solver(op, x, rhs):
            odl.solvers.mlem(op, x, rhs, niter=10)
    elif solver_name == 'osmlem':
        def solver(op, x, rhs):
            odl.solvers.osmlem([op, op], x, [rhs, rhs], niter=10)
    elif solver_name == 'kaczmarz':
        def solver(op, x, rhs):
            norm2 = op.adjoint(op(x)).norm() / x.norm()
            odl.solvers.kaczmarz([op, op], x, [rhs, rhs], niter=20,
                                 omega=0.5 / norm2)
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
        op = odl.MatrixOperator(op_arr)

        # Simple right hand side
        rhs = op.range.one()

        # Initial guess
        x = op.domain.element([0.6, 0.8, 1.0, 1.2, 1.4])

        return op, x, rhs
    elif problem_name == 'Identity':
        # Define problem
        space = odl.uniform_discr(0, 1, 5)
        op = odl.IdentityOperator(space)

        # Simple right hand side
        rhs = op.range.element([0, 0, 1, 0, 0])

        # Initial guess
        x = op.domain.element([0.6, 0.8, 1.0, 1.2, 1.4])

        return op, x, rhs
    else:
        raise ValueError('problem not valid')


def test_solver(optimization_problem, iterative_solver):
    """Test iterative solver for solving some simple problems."""

    # Solve within 1%
    places = 2

    op, x, rhs = optimization_problem

    # Solve problem
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
