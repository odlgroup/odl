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
                        'conjugate_gradient_normal'])
def iterative_solver(request):
    """Return a solver given by a name with interface solve(op, x, rhs)."""
    solver_name = request.param

    if solver_name == 'steepest_descent':
        def solver(op, x, rhs):
            norm2 = op.adjoint(op(x)).norm() / x.norm()

            # Define gradient as ``Ax - b``
            gradient_op = op.adjoint * odl.ResidualOperator(op, rhs)
            odl.solvers.steepest_descent(gradient_op, x, niter=10,
                                         line_search=0.5 / norm2)
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
    else:
        raise ValueError('solver not valid')

    return solver


def test_solver(iterative_solver):
    """Test discrete X-ray transform using ASTRA for reconstruction."""

    # Solve within 1%
    places = 2

    # Define problem
    op_arr = np.eye(5) * 5 + np.ones([5, 5])
    op = odl.MatVecOperator(op_arr)

    # Simple right hand side
    rhs = op.range.one()

    # Solve problem
    x = op.domain.one()
    iterative_solver(op, x, rhs)

    # Assert residual is small
    assert all_almost_equal(op(x), rhs, places)

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
