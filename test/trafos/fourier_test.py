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

# External module imports
import pytest
from math import pi
import numpy as np

# ODL imports
import odl
from odl.trafos.fourier import reciprocal
from odl.util.testutils import all_almost_equal, all_equal


def test_reciprocal_1d_odd():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=11, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)
    rgrid = reciprocal(grid, halfcomplex=False)

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)

    # Should be symmetric
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)
    assert all_almost_equal(rgrid.center, 0)


def test_reciprocal_1d_odd_halfcomplex():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=11, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)
    rgrid = reciprocal(grid, halfcomplex=True)

    assert all_equal(rgrid.shape, (n + 1) / 2)
    assert all_almost_equal(rgrid.stride, true_recip_stride)

    # Max should be zero
    assert all_almost_equal(rgrid.max_pt, 0)


def test_reciprocal_1d_even():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=10, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, halfcomplex=False, even_shift=False)

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # No point should be closer to 0 than half a recip stride
    tol = 0.999 * true_recip_stride / 2
    assert not rgrid.approx_contains(0, tol=tol)

    # With shift (to the left)
    rgrid = reciprocal(grid, halfcomplex=False, even_shift=True)

    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Zero should be at index n/2
    assert all_almost_equal(rgrid[n / 2], 0)


def test_reciprocal_1d_even_halfcomplex():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=10, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, halfcomplex=True, even_shift=False)

    assert all_equal(rgrid.shape, n / 2 + 1)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max point should be half a positinve recip stride
    assert all_almost_equal(rgrid.max_pt, true_recip_stride / 2)

    # With shift (to the left)
    rgrid = reciprocal(grid, halfcomplex=True, even_shift=True)

    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max point should be zero
    assert all_almost_equal(rgrid.max_pt, 0)


def test_reciprocal_nd():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    grid = odl.uniform_sampling(cube, num_nodes=(3, 4, 5), as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, halfcomplex=False, even_shift=False)

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
