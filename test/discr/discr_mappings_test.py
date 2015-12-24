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
import odl
from odl.discr.grid import sparse_meshgrid
from odl.discr.discr_mappings import (
    GridCollocation, NearestInterpolation, LinearInterpolation)
from odl.util.testutils import all_almost_equal, all_equal, almost_equal


def test_nearest_interpolation_1d_complex():
    intv = odl.Interval(0, 1)
    grid = odl.uniform_sampling(intv, 5, as_midp=True)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    space = odl.FunctionSpace(intv, field=odl.ComplexNumbers())
    dspace = odl.Fn(grid.ntotal, dtype='complex128')
    interp_op = NearestInterpolation(space, grid, dspace)
    values = np.arange(5) + 1j * np.arange(1, 6)
    function = interp_op(values)

    # Evaluate at single point
    val = function(0.35)  # closest to index 1 -> 1 + 2j
    assert val == 1.0 + 2.0j
    # Input array, with and without output array
    pts = np.array([0.4, 0.0, 0.65, 0.95])
    true_arr = [1 + 2j, 0 + 1j, 3 + 4j, 4 + 5j]
    assert all_equal(function(pts), true_arr)
    # Should also work with a (1, N) array
    pts = pts[None, :]
    assert all_equal(function(pts), true_arr)
    out = np.empty(4, dtype='complex128')
    function(pts, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    # Same as array for 1d
    mg = sparse_meshgrid([0.4, 0.0, 0.65, 0.95])
    true_mg = [1 + 2j, 0 + 1j, 3 + 4j, 4 + 5j]
    assert all_equal(function(mg), true_mg)
    function(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_1d_variants():
    intv = odl.Interval(0, 1)
    grid = odl.uniform_sampling(intv, 5, as_midp=True)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    space = odl.FunctionSpace(intv)
    dspace = odl.Fn(grid.ntotal, dtype='float64')

    # 'left' variant
    interp_op = NearestInterpolation(space, grid, dspace, variant='left')
    values = np.arange(5, dtype='float64')
    function = interp_op(values)

    # Testing two midpoints and the extreme values
    pts = np.array([0.4, 0.8, 0.0, 1.0])
    true_arr = [1, 3, 0, 4]
    assert all_equal(function(pts), true_arr)

    # 'right' variant
    interp_op = NearestInterpolation(space, grid, dspace, variant='right')
    values = np.arange(5)
    function = interp_op(values)

    # Testing two midpoints and the extreme values
    pts = np.array([0.4, 0.8, 0.0, 1.0])
    true_arr = [2, 4, 0, 4]
    assert all_equal(function(pts), true_arr)


def test_nearest_interpolation_2d_float():
    rect = odl.Rectangle([0, 0], [1, 1])
    grid = odl.uniform_sampling(rect, [4, 2], as_midp=True)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    space = odl.FunctionSpace(rect)
    dspace = odl.Fn(grid.ntotal, dtype='float64')
    interp_op = NearestInterpolation(space, grid, dspace)
    values = np.arange(8, dtype='float64')
    function = interp_op(values)

    # Evaluate at single point
    val = function([0.3, 0.6])  # closest to index (1, 1) -> 3
    assert val == 3.0
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = [3, 7]
    assert all_equal(function(pts.T), true_arr)
    out = np.empty(2, dtype='float64')
    function(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
    # Indices: (1, 3) x (0, 1)
    true_mg = [[2, 3],
               [6, 7]]
    assert all_equal(function(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    function(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_2d_string():
    rect = odl.Rectangle([0, 0], [1, 1])
    grid = odl.uniform_sampling(rect, [4, 2], as_midp=True)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    space = odl.FunctionSet(rect, odl.Strings(1))
    dspace = odl.Ntuples(grid.ntotal, dtype='U1')
    interp_op = NearestInterpolation(space, grid, dspace)
    values = np.array([c for c in 'mystring'])
    function = interp_op(values)

    # Evaluate at single point
    val = function([0.3, 0.6])  # closest to index (1, 1) -> 3
    assert val == 't'
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = ['t', 'g']
    assert all_equal(function(pts.T), true_arr)
    out = np.empty(2, dtype='U1')
    function(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
    # Indices: (1, 3) x (0, 1)
    true_mg = [['s', 't'],
               ['n', 'g']]
    assert all_equal(function(mg), true_mg)
    out = np.empty((2, 2), dtype='U1')
    function(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_2d_fortran_ordering():
    rect = odl.Rectangle([0, 0], [1, 1])
    grid = odl.uniform_sampling(rect, [4, 2], as_midp=True)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    space = odl.FunctionSpace(rect)
    dspace = odl.Fn(grid.ntotal, dtype='float64')
    interp_op = NearestInterpolation(space, grid, dspace, order='F')
    values = np.arange(8, dtype='float64')
    function = interp_op(values)

    # Evaluate at single point
    val = function([0.3, 0.6])  # closest to index (1, 1) -> 5
    assert val == 5.0
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = [5, 7]
    assert all_equal(function(pts.T), true_arr)
    out = np.empty(2, dtype='float64')
    function(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0], order='F')
    # Indices: (0, 1) x (1, 3)
    # Values are different since we did not reshuffle values accordingly
    true_mg = [[1, 3],
               [5, 7]]
    assert all_equal(function(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    function(mg, out=out)
    assert all_equal(out, true_mg)


def test_linear_interpolation_1d():
    intv = odl.Interval(0, 1)
    grid = odl.uniform_sampling(intv, 5, as_midp=True)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    space = odl.FunctionSpace(intv)
    dspace = odl.Fn(grid.ntotal, dtype='float64')
    interp_op = LinearInterpolation(space, grid, dspace)
    values = np.arange(1, 6, dtype='float64')
    function = interp_op(values)

    print('--- Single point ---')
    # Evaluate at single point
    val = function(0.35)
    true_val = 0.75 * 2 + 0.25 * 3
    assert almost_equal(val, true_val)
    print('--- Array ---')
    # Input array, with and without output array
    pts = np.array([0.4, 0.0, 0.65, 0.95])
    true_arr = [2.5, 0.5, 3.75, 3.75]
    assert all_almost_equal(function(pts), true_arr)


def test_linear_interpolation_2d():
    rect = odl.Rectangle([0, 0], [1, 1])
    grid = odl.uniform_sampling(rect, [4, 2], as_midp=True)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    space = odl.FunctionSpace(rect)
    dspace = odl.Fn(grid.ntotal, dtype='float64')
    interp_op = LinearInterpolation(space, grid, dspace)
    values = np.arange(1, 9, dtype='float64')
    function = interp_op(values)

    # Evaluate at single point
    val = function([0.3, 0.6])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val = ((1 - l1) * (1 - l2) * 1 +
                (1 - l1) * l2 * 2 +
                l1 * (1 - l2) * 3 +
                l1 * l2 * 4)
    assert almost_equal(val, true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val_1 = ((1 - l1) * (1 - l2) * 1 +
                  (1 - l1) * l2 * 2 +
                  l1 * (1 - l2) * 3 +
                  l1 * l2 * 4)
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    true_val_2 = (1 - l1) * 1
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    l2 = (1.0 - 0.75) / (0.75 - 0.25)
    true_val_3 = (1 - l1) * (1 - l2) * 8
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(function(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    function(pts.T, out=out)
    assert all_equal(out, true_arr)

#    # Input meshgrid, with and without output array
#    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
#    # Indices: (1, 3) x (0, 1)
#    true_mg = [[2, 3],
#               [6, 7]]
#    assert all_equal(function(mg), true_mg)
#    out = np.empty((2, 2), dtype='float64')
#    function(mg, out=out)
#    assert all_equal(out, true_mg)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
