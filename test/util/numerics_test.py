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

# External
import pytest
import numpy as np

# Internal
from odl.util.numerics import apply_on_boundary
from odl.util.testutils import all_equal


def test_apply_on_boundary_default():

    # 1d
    arr = np.ones(5)
    result = apply_on_boundary(arr, lambda x: x * 2)
    assert all_equal(arr, [1, 1, 1, 1, 1])
    assert all_equal(result, [2, 1, 1, 1, 2])

    # 3d
    arr = np.ones((3, 4, 5))
    result = apply_on_boundary(arr, lambda x: x * 2)
    true_arr = 2 * np.ones((3, 4, 5))
    true_arr[1:-1, 1:-1, 1:-1] = 1
    assert all_equal(result, true_arr)


def test_apply_on_boundary_func_sequence_2d():

    arr = np.ones((3, 5))
    result = apply_on_boundary(arr, [lambda x: x * 2, lambda x: x * 3])
    assert all_equal(result, [[2, 2, 2, 2, 2],
                              [3, 1, 1, 1, 3],
                              [2, 2, 2, 2, 2]])


def test_apply_on_boundary_multiple_times_2d():

    arr = np.ones((3, 5))
    result = apply_on_boundary(arr, lambda x: x * 2, only_once=False)
    assert all_equal(result, [[4, 2, 2, 2, 4],
                              [2, 1, 1, 1, 2],
                              [4, 2, 2, 2, 4]])


def test_apply_on_boundary_which_boundaries():

    # 1d
    arr = np.ones(5)
    which = ((True, False),)
    result = apply_on_boundary(arr, lambda x: x * 2, which_boundaries=which)
    assert all_equal(result, [2, 1, 1, 1, 1])

    # 2d
    arr = np.ones((3, 5))
    which = ((True, False), True)
    result = apply_on_boundary(arr, lambda x: x * 2, which_boundaries=which)
    assert all_equal(result, [[2, 2, 2, 2, 2],
                              [2, 1, 1, 1, 2],
                              [2, 1, 1, 1, 2]])


def test_apply_on_boundary_which_boundaries_multiple_times_2d():

    # 2d
    arr = np.ones((3, 5))
    which = ((True, False), True)
    result = apply_on_boundary(arr, lambda x: x * 2, which_boundaries=which,
                               only_once=False)
    assert all_equal(result, [[4, 2, 2, 2, 4],
                              [2, 1, 1, 1, 2],
                              [2, 1, 1, 1, 2]])


def test_apply_on_boundary_axis_order_2d():

    arr = np.ones((3, 5))
    axis_order = (-1, -2)
    result = apply_on_boundary(arr, [lambda x: x * 3, lambda x: x * 2],
                               axis_order=axis_order)
    assert all_equal(result, [[3, 2, 2, 2, 3],
                              [3, 1, 1, 1, 3],
                              [3, 2, 2, 2, 3]])

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
