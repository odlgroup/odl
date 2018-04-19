# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

from itertools import product

import numpy as np
import pytest

import odl
from odl.trafos.util.ft_utils import (
    dft_preprocess_data, realspace_grid, reciprocal_grid)
from odl.util import all_almost_equal, all_equal
from odl.util.testutils import simple_fixture

# --- pytest fixtures --- #


halfcomplex = simple_fixture('halfcomplex', [True, False])
shift = simple_fixture('shift', [True, False])
parity = simple_fixture('parity', ['even', 'odd'])
sign = simple_fixture('sign', ['-', '+'])


# --- reciprocal_grid --- #


def test_reciprocal_grid_1d(halfcomplex, shift, parity):

    shape = 10 if parity == 'even' else 11
    grid = odl.uniform_grid(0, 1, shape=shape)
    s = grid.stride
    n = np.array(grid.shape)

    rgrid = reciprocal_grid(grid, shift=shift, halfcomplex=halfcomplex)

    # Independent of halfcomplex, shift and parity
    true_recip_stride = 2 * np.pi / (s * n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)

    if halfcomplex:
        assert all_equal(rgrid.shape, n // 2 + 1)

        if parity == 'odd' and shift:
            # Max point should be half a negative recip stride
            assert all_almost_equal(rgrid.max_pt, -true_recip_stride / 2)
        elif parity == 'even' and not shift:
            # Max point should be half a positive recip stride
            assert all_almost_equal(rgrid.max_pt, true_recip_stride / 2)
        elif (parity == 'odd' and not shift) or (parity == 'even' and shift):
            # Max should be zero
            assert all_almost_equal(rgrid.max_pt, 0)
        else:
            raise RuntimeError('parameter combination not covered')
    else:  # halfcomplex = False
        assert all_equal(rgrid.shape, n)

        if (parity == 'even' and shift) or (parity == 'odd' and not shift):
            # Zero should be at index n // 2
            assert all_almost_equal(rgrid[n // 2], 0)
        elif (parity == 'odd' and shift) or (parity == 'even' and not shift):
            # No point should be closer to 0 than half a recip stride
            atol = 0.999 * true_recip_stride / 2
            assert not rgrid.approx_contains(0, atol=atol)
        else:
            raise RuntimeError('parameter combination not covered')

        if not shift:
            # Grid Should be symmetric
            assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)
            if parity == 'odd':
                # Midpoint should be 0
                assert all_almost_equal(rgrid.mid_pt, 0)

    # Inverting should give back the original
    irgrid = realspace_grid(rgrid, grid.min_pt, halfcomplex=halfcomplex,
                            halfcx_parity=parity)
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_grid_nd():

    grid = odl.uniform_grid([0] * 3, [1] * 3, shape=(3, 4, 5))
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * np.pi / (s * n)

    # Without shift altogether
    rgrid = reciprocal_grid(grid, shift=False, halfcomplex=False)

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)

    # Inverting should give back the original
    irgrid = realspace_grid(rgrid, grid.min_pt, halfcomplex=False)
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_grid_nd_shift_list():

    grid = odl.uniform_grid([0] * 3, [1] * 3, shape=(3, 4, 5))
    s = grid.stride
    n = np.array(grid.shape)
    shift = [False, True, False]

    true_recip_stride = 2 * np.pi / (s * n)

    # Shift only the even dimension, then zero must be contained
    rgrid = reciprocal_grid(grid, shift=shift, halfcomplex=False)
    noshift = np.where(np.logical_not(shift))

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    assert all_almost_equal(rgrid.min_pt[noshift], -rgrid.max_pt[noshift])
    assert all_almost_equal(rgrid[n // 2], [0] * 3)

    # Inverting should give back the original
    irgrid = realspace_grid(rgrid, grid.min_pt, halfcomplex=False)
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_grid_nd_axes():

    grid = odl.uniform_grid([0] * 3, [1] * 3, shape=(3, 4, 5))
    s = grid.stride
    n = np.array(grid.shape)
    axes_list = [[1, -1], [0], 0, [0, 2, 1], [2, 0]]

    for axes in axes_list:
        active = np.zeros(grid.ndim, dtype=bool)
        active[axes] = True
        inactive = np.logical_not(active)

        true_recip_stride = np.empty(grid.ndim)
        true_recip_stride[active] = 2 * np.pi / (s[active] * n[active])
        true_recip_stride[inactive] = s[inactive]

        # Without shift altogether
        rgrid = reciprocal_grid(grid, shift=False, axes=axes,
                                halfcomplex=False)

        assert all_equal(rgrid.shape, n)
        assert all_almost_equal(rgrid.stride, true_recip_stride)
        assert all_almost_equal(rgrid.min_pt[active], -rgrid.max_pt[active])
        assert all_equal(rgrid.min_pt[inactive], grid.min_pt[inactive])
        assert all_equal(rgrid.max_pt[inactive], grid.max_pt[inactive])

        # Inverting should give back the original
        irgrid = realspace_grid(rgrid, grid.min_pt, axes=axes,
                                halfcomplex=False)
        assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_grid_nd_halfcomplex():

    grid = odl.uniform_grid([0] * 3, [1] * 3, shape=(3, 4, 5))
    s = grid.stride
    n = np.array(grid.shape)
    stride_last = 2 * np.pi / (s[-1] * n[-1])
    n[-1] = n[-1] // 2 + 1

    # Without shift
    rgrid = reciprocal_grid(grid, shift=False, halfcomplex=True)
    assert all_equal(rgrid.shape, n)
    assert rgrid.max_pt[-1] == 0  # last dim is odd

    # With shift
    rgrid = reciprocal_grid(grid, shift=True, halfcomplex=True)
    assert all_equal(rgrid.shape, n)
    assert rgrid.max_pt[-1] == -stride_last / 2

    # Inverting should give back the original
    irgrid = realspace_grid(rgrid, grid.min_pt, halfcomplex=True,
                            halfcx_parity='odd')
    assert irgrid.approx_equals(grid, atol=1e-6)

    with pytest.raises(ValueError):
        realspace_grid(rgrid, grid.min_pt, halfcomplex=True,
                       halfcx_parity='+')


# --- dft_preprocess_data --- #


def test_dft_preprocess_data(sign):

    shape = (2, 3, 4)

    # With shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append((1 + 1j) * (1 - 2 * ((i + j + k) % 2)))

    arr = np.ones(shape, dtype='complex64') * (1 + 1j)
    preproc = dft_preprocess_data(arr, shift=True, sign=sign)  # out-of-place
    dft_preprocess_data(arr, shift=True, out=arr, sign=sign)  # in-place

    assert all_almost_equal(preproc.ravel(), correct_arr)
    assert all_almost_equal(arr.ravel(), correct_arr)

    # Without shift
    imag = 1j if sign == '-' else -1j
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        argsum = sum((idx * (1 - 1 / shp))
                     for idx, shp in zip((i, j, k), shape))

        correct_arr.append((1 + 1j) * np.exp(imag * np.pi * argsum))

    arr = np.ones(shape, dtype='complex64') * (1 + 1j)
    dft_preprocess_data(arr, shift=False, out=arr, sign=sign)

    assert all_almost_equal(arr.ravel(), correct_arr)

    # Bad input
    with pytest.raises(ValueError):
        dft_preprocess_data(arr, out=arr, sign=1)

    arr = np.zeros(shape, dtype='S2')
    with pytest.raises(ValueError):
        dft_preprocess_data(arr)


def test_dft_preprocess_data_halfcomplex(sign):

    shape = (2, 3, 4)

    # With shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * ((i + j + k) % 2))

    arr = np.ones(shape, dtype='float64')
    preproc = dft_preprocess_data(arr, shift=True, sign=sign)  # out-of-place
    out = np.empty_like(arr)
    dft_preprocess_data(arr, shift=True, out=out, sign=sign)  # in-place
    dft_preprocess_data(arr, shift=True, out=arr, sign=sign)  # in-place
    assert all_almost_equal(preproc.ravel(), correct_arr)
    assert all_almost_equal(arr.ravel(), correct_arr)
    assert all_almost_equal(out.ravel(), correct_arr)

    # Without shift
    imag = 1j if sign == '-' else -1j
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        argsum = sum((idx * (1 - 1 / shp))
                     for idx, shp in zip((i, j, k), shape))

        correct_arr.append(np.exp(imag * np.pi * argsum))

    arr = np.ones(shape, dtype='float64')
    preproc = dft_preprocess_data(arr, shift=False, sign=sign)
    assert all_almost_equal(preproc.ravel(), correct_arr)

    # Non-float input works, too
    arr = np.ones(shape, dtype='int')
    preproc = dft_preprocess_data(arr, shift=False, sign=sign)
    assert all_almost_equal(preproc.ravel(), correct_arr)

    # In-place modification not possible for float array and no shift
    arr = np.ones(shape, dtype='float64')
    with pytest.raises(ValueError):
        dft_preprocess_data(arr, shift=False, out=arr, sign=sign)


def test_dft_preprocess_data_with_axes(sign):

    shape = (2, 3, 4)

    axes = 1  # Only middle index counts
    correct_arr = []
    for _, j, __ in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * (j % 2))

    arr = np.ones(shape, dtype='complex64')
    dft_preprocess_data(arr, shift=True, axes=axes, out=arr, sign=sign)

    assert all_almost_equal(arr.ravel(), correct_arr)

    axes = [0, -1]  # First and last
    correct_arr = []
    for i, _, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * ((i + k) % 2))

    arr = np.ones(shape, dtype='complex64')
    dft_preprocess_data(arr, shift=True, axes=axes, out=arr, sign=sign)

    assert all_almost_equal(arr.ravel(), correct_arr)


if __name__ == '__main__':
    odl.util.test_file(__file__)
