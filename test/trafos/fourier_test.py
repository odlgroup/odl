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
from itertools import product
from math import pi
import numpy as np
import pytest

# ODL imports
import odl
from odl.trafos.fourier import reciprocal, _shift_list, dft_preproc_data
from odl.util.testutils import all_almost_equal, all_equal


def test_shift_list():

    length = 3

    # Test single value
    shift = True
    lst = _shift_list(shift, length)

    assert all_equal(lst, [True] * length)

    # Test existing sequence
    shift = (False,) * length
    lst = _shift_list(shift, length)

    assert all_equal(lst, [False] * length)

    # Too long sequence, gets truncated but ok
    shift = (False,) * (length + 1)
    lst = _shift_list(shift, length)

    assert all_equal(lst, [False] * length)

    # Test iterable
    def alternating():
        i = 0
        while 1:
            yield bool(i % 2)
            i += 1

    lst = _shift_list(alternating(), length)

    assert all_equal(lst, [False, True, False])

    # Too short sequence, should raise
    shift = (False,) * (length - 1)
    with pytest.raises(ValueError):
        _shift_list(shift, length)

    # Iterable returning too few entries, should throw
    def alternating_short():
        i = 0
        while i < length - 1:
            yield bool(i % 2)
            i += 1

    with pytest.raises(ValueError):
        _shift_list(alternating_short(), length)


def test_reciprocal_1d_odd():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=11, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Should be symmetric
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)
    assert all_almost_equal(rgrid.center, 0)
    # Zero should be at index n // 2
    assert all_almost_equal(rgrid[n // 2], 0)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # No point should be closer to 0 than half a recip stride
    tol = 0.999 * true_recip_stride / 2
    assert not rgrid.approx_contains(0, tol=tol)


def test_reciprocal_1d_odd_halfcomplex():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=11, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=True)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, (n + 1) / 2)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max should be zero
    assert all_almost_equal(rgrid.max_pt, 0)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=True)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, (n + 1) / 2)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max point should be half a positive recip stride
    assert all_almost_equal(rgrid.max_pt, -true_recip_stride / 2)


def test_reciprocal_1d_even():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=10, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Should be symmetric
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)
    assert all_almost_equal(rgrid.center, 0)
    # No point should be closer to 0 than half a recip stride
    tol = 0.999 * true_recip_stride / 2
    assert not rgrid.approx_contains(0, tol=tol)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Zero should be at index n // 2
    assert all_almost_equal(rgrid[n // 2], 0)


def test_reciprocal_1d_even_halfcomplex():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=10, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=True)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n / 2 + 1)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max point should be half a positive recip stride
    assert all_almost_equal(rgrid.max_pt, true_recip_stride / 2)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=True)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n / 2 + 1)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max should be zero
    assert all_almost_equal(rgrid.max_pt, 0)


def test_reciprocal_nd():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    grid = odl.uniform_sampling(cube, num_nodes=(3, 4, 5), as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift altogether
    rgrid = reciprocal(grid, shift=False, halfcomplex=False)

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)


def test_reciprocal_nd_shift_list():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    grid = odl.uniform_sampling(cube, num_nodes=(3, 4, 5), as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)
    shift = [False, True, False]

    true_recip_stride = 2 * pi / (s * n)

    # Shift only the even dimension, then zero must be contained
    rgrid = reciprocal(grid, shift=shift, halfcomplex=False)
    noshift = np.where(np.logical_not(shift))

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    assert all_almost_equal(rgrid.min_pt[noshift], -rgrid.max_pt[noshift])
    assert all_almost_equal(rgrid[n // 2], [0] * 3)


def test_reciprocal_nd_halfcomplex():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    grid = odl.uniform_sampling(cube, num_nodes=(3, 4, 5), as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)
    stride_last = 2 * pi / (s[-1] * n[-1])
    n[-1] = n[-1] // 2 + 1

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=True)
    assert all_equal(rgrid.shape, n)
    assert rgrid.max_pt[-1] == 0  # last dim is odd

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=True)
    assert all_equal(rgrid.shape, n)
    assert rgrid.max_pt[-1] == -stride_last / 2


def test_dft_preproc_data():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    shape = (2, 3, 4)
    discr = odl.uniform_discr(
        odl.FunctionSpace(cube, field=odl.ComplexNumbers()),
        shape)

    # With shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * ((i + j + k) % 2))

    dfunc = discr.one()
    dft_preproc_data(dfunc, shift=True)

    assert all_almost_equal(dfunc.ntuple, correct_arr)

    # Without shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        argsum = sum((idx * (1 - 1 / shp))
                     for idx, shp in zip((i, j, k), shape))

        correct_arr.append(np.exp(1j * np.pi * argsum))

    dfunc = discr.one()
    dft_preproc_data(dfunc, shift=False)

    assert all_almost_equal(dfunc.ntuple, correct_arr)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
