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

"""Test for the smooth solvers."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import odl


def test_backtracking_line_search():
    """Test some basic properties of BacktrackingLineSearch."""
    space = odl.rn(2)

    func = odl.solvers.L2NormSquared(space)

    line_search = odl.solvers.BacktrackingLineSearch(func)

    x = space.element([1, 0])
    for direction in [space.element([1, 0]),
                      space.element([-1, 0]),
                      space.element([-1, -1])]:
        dir_derivative = func.gradient(x).inner(direction)

        steplen = line_search(x, direction, dir_derivative)
        assert func(x + steplen * direction) < func(x)


def test_constant_line_search():
    """Test some basic properties of BacktrackingLineSearch."""
    space = odl.rn(2)

    func = odl.solvers.L2NormSquared(space)

    line_search = odl.solvers.ConstantLineSearch(0.57)

    x = space.element([1, 0])
    for direction in [space.element([1, 0]),
                      space.element([-1, 0]),
                      space.element([-1, -1])]:
        dir_derivative = func.gradient(x).inner(direction)

        steplen = line_search(x, direction, dir_derivative)
        assert steplen == 0.57


def test_line_search_from_iternum():
    """Test some basic properties of LineSearchFromIterNum."""
    space = odl.rn(2)

    func = odl.solvers.L2NormSquared(space)

    line_search = odl.solvers.LineSearchFromIterNum(lambda n: 1 / (n + 1))

    x = space.element([1, 0])
    for n, direction in enumerate([space.element([1, 0]),
                                   space.element([-1, 0]),
                                   space.element([-1, -1])]):
        dir_derivative = func.gradient(x).inner(direction)

        steplen = line_search(x, direction, dir_derivative)
        assert steplen == 1 / (n + 1)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
