# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the smooth solvers."""

from __future__ import division
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
