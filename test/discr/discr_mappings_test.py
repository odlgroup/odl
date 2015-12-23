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
from odl.util.testutils import all_equal


def test_nearest_interpolation_2d():

    rect = odl.Rectangle([0, 0], [1, 1])
    grid = odl.uniform_sampling(rect, [4, 2], as_midp=True)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    # Scalar case using float
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

    # Non-scalar case using 1-character strings
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


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
