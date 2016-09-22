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

"""Test for the Chambolle-Pock solver."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import pytest

# Internal
import odl
from odl.util.testutils import all_almost_equal, noise_element


@pytest.fixture(scope="module", params=['l2_squared', 'l2_squared_scaled'])
def functional(request):
    """Return a functional whose optimal value is at 0."""
    name = request.param

    if name == 'l2_squared':
        space = odl.rn(3)
        return odl.solvers.L2NormSquared(space)
    elif name == 'l2_squared_scaled':
        # Use ||grad f||, where constant boundary condition causes the solution
        # to be f = 0
        space = odl.uniform_discr(0, 1, 5)
        scaling = odl.MultiplyOperator(space.element([1, 2, 3, 5, 5]),
                                       domain=space)
        return odl.solvers.L2NormSquared(space) * scaling
    else:
        assert False


@pytest.fixture(scope="module", params=['constant', 'backtracking'])
def functional_and_linesearch(request, functional):
    """Return a functional whose optimal value is at 0."""
    name = request.param

    if name == 'constant':
        return functional, 1.0
    else:
        return functional, odl.solvers.BacktrackingLineSearch(functional)


@pytest.fixture(scope="module", params=['first', 'second'])
def broyden_impl(request):
    return request.param


def test_bfgs_solver(functional_and_linesearch):
    """Test a quasi-newton solver."""
    functional, line_search = functional_and_linesearch

    x = noise_element(functional.domain)
    odl.solvers.bfgs_method(functional, x, maxiter=50, tol=1e-4,
                            line_search=line_search)

    assert functional(x) < 1e-3


def test_broydens_method(functional_and_linesearch, broyden_impl):
    """Test a quasi-newton solver."""
    functional, line_search = functional_and_linesearch

    x = noise_element(functional.domain)
    odl.solvers.broydens_method(functional, x, maxiter=50, tol=1e-4,
                                line_search=line_search, impl=broyden_impl)

    assert functional(x) < 1e-3


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
