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
                        'adam',
                        'landweber',
                        'conjugate_gradient',
                        'conjugate_gradient_normal',
                        'mlem',
                        'osmlem',
                        'kaczmarz'])
def iterative_solver(request):
    """Return a solver given by a name with interface solve(op, x, rhs)."""
    solver_name = request.param

    if solver_name == 'steepest_descent':
        def solver(op, x, rhs):
            space = op.domain
            norm2 = space.norm(op.adjoint(op(x))) / space.norm(x)
            func = odl.solvers.L2NormSquared(op.domain) * (op - rhs)

            odl.solvers.steepest_descent(func, x, line_search=0.5 / norm2)

    elif solver_name == 'adam':
        def solver(op, x, rhs):
            space = op.domain
            norm2 = space.norm(op.adjoint(op(x))) / space.norm(x)
            func = odl.solvers.L2NormSquared(op.domain) * (op - rhs)

            odl.solvers.adam(func, x, learning_rate=4.0 / norm2, maxiter=150)

    elif solver_name == 'landweber':
        def solver(op, x, rhs):
            space = op.domain
            norm2 = space.norm(op.adjoint(op(x))) / space.norm(x)
            odl.solvers.landweber(op, x, rhs, niter=50, omega=0.5 / norm2)

    elif solver_name == 'conjugate_gradient':
        def solver(op, x, rhs):
            odl.solvers.conjugate_gradient(op, x, rhs, niter=10)

    elif solver_name == 'conjugate_gradient_normal':
        def solver(op, x, rhs):
            odl.solvers.conjugate_gradient_normal(op, x, rhs, niter=10)

    elif solver_name == 'mlem':
        def solver(op, x, rhs):
            odl.solvers.mlem(op, x, rhs, niter=10)

    elif solver_name == 'osmlem':
        def solver(op, x, rhs):
            odl.solvers.osmlem([op, op], x, [rhs, rhs], niter=10)

    elif solver_name == 'kaczmarz':
        def solver(op, x, rhs):
            space = op.domain
            norm2 = space.norm(op.adjoint(op(x))) / space.norm(x)
            odl.solvers.kaczmarz(
                [op, op], x, [rhs, rhs], niter=20, omega=0.5 / norm2
            )

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
    op, x, rhs = optimization_problem

    iterative_solver(op, x, rhs)
    assert all_almost_equal(op(x), rhs, ndigits=2)


def test_steepst_descent():
    """Test steepest descent on the rosenbrock function in 3d."""
    space = odl.rn(3)
    scale = 1  # only mildly ill-behaved
    rosenbrock = odl.solvers.RosenbrockFunctional(space, scale)

    line_search = odl.solvers.BacktrackingLineSearch(
        rosenbrock, tau=0.1, discount=0.01
    )
    x = rosenbrock.domain.zero()
    odl.solvers.steepest_descent(
        rosenbrock, x, maxiter=40, line_search=line_search
    )

    assert all_almost_equal(x, [1, 1, 1], ndigits=2)


if __name__ == '__main__':
    odl.util.test_file(__file__)
