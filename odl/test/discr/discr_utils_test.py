# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `discr_utils`."""

from __future__ import division

import numpy as np
import pytest

import odl
from odl.discr.discr_utils import (
    linear_interpolator, nearest_interpolator, per_axis_interpolator,
    point_collocation)
from odl.discr.grid import sparse_meshgrid
from odl.util.testutils import all_almost_equal, all_equal


def test_nearest_interpolation_1d_complex():
    """Test nearest neighbor interpolation in 1d with complex values."""
    coord_vecs = [[0.1, 0.3, 0.5, 0.7, 0.9]]
    f = np.array([0 + 1j, 1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j], dtype="complex128")
    interpolator = nearest_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator(0.35)  # closest to index 1 -> 1 + 2j
    assert val == 1.0 + 2.0j
    # Input array, with and without output array
    pts = np.array([0.39, 0.0, 0.65, 0.95])
    true_arr = [1 + 2j, 0 + 1j, 3 + 4j, 4 + 5j]
    assert all_equal(interpolator(pts), true_arr)
    # Should also work with a (1, N) array
    pts = pts[None, :]
    assert all_equal(interpolator(pts), true_arr)
    out = np.empty(4, dtype='complex128')
    interpolator(pts, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    # Same as array for 1d
    mg = sparse_meshgrid([0.39, 0.0, 0.65, 0.95])
    true_mg = [1 + 2j, 0 + 1j, 3 + 4j, 4 + 5j]
    assert all_equal(interpolator(mg), true_mg)
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_2d():
    """Test nearest neighbor interpolation in 2d."""
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    f = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype="float64").reshape((4, 2))
    interpolator = nearest_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])  # closest to index (1, 1) -> 3
    assert val == 3.0
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = [3, 7]
    assert all_equal(interpolator(pts.T), true_arr)
    out = np.empty(2, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
    # Indices: (1, 3) x (0, 1)
    true_mg = [[2, 3],
               [6, 7]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_2d_string():
    """Test nearest neighbor interpolation in 2d with string values."""
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    f = np.array(list('mystring'), dtype='U1').reshape((4, 2))
    interpolator = nearest_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])  # closest to index (1, 1) -> 3
    assert val == u't'
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = np.array(['t', 'g'], dtype='U1')
    assert all_equal(interpolator(pts.T), true_arr)
    out = np.empty(2, dtype='U1')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
    # Indices: (1, 3) x (0, 1)
    true_mg = np.array([['s', 't'],
                        ['n', 'g']], dtype='U1')
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='U1')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_linear_interpolation_1d():
    """Test linear interpolation in 1d."""
    coord_vecs = [[0.1, 0.3, 0.5, 0.7, 0.9]]
    f = np.array([1, 2, 3, 4, 5], dtype="float64")
    interpolator = linear_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator(0.35)
    true_val = 0.75 * 2 + 0.25 * 3
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([0.4, 0.0, 0.65, 0.95])
    true_arr = [2.5, 0.5, 3.75, 3.75]
    assert all_almost_equal(interpolator(pts), true_arr)


def test_linear_interpolation_2d():
    """Test linear interpolation in 2d."""
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    f = np.arange(1, 9, dtype='float64').reshape((4, 2))
    interpolator = linear_interpolator(f, coord_vecs)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val = (
        (1 - l1) * (1 - l2) * f[0, 0]
        + (1 - l1) * l2 * f[0, 1]
        + l1 * (1 - l2) * f[1, 0]
        + l1 * l2 * f[1, 1]
    )
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val_1 = (
        (1 - l1) * (1 - l2) * f[0, 0]
        + (1 - l1) * l2 * f[0, 1]
        + l1 * (1 - l2) * f[1, 0]
        + l1 * l2 * f[1, 1]
    )
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    # l2 = 0
    true_val_2 = (1 - l1) * f[0, 0]  # only lower left contributes
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    l2 = (1.0 - 0.75) / (0.75 - 0.25)
    true_val_3 = (1 - l1) * (1 - l2) * f[3, 1]  # lower left only
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(interpolator(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)

    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 0.75])
    # Indices: (1, 3) x (0, 1)
    lx1 = (0.3 - 0.125) / (0.375 - 0.125)
    lx2 = (1.0 - 0.875) / (0.875 - 0.625)
    ly1 = (0.4 - 0.25) / (0.75 - 0.25)
    # ly2 = 0
    true_val_11 = (
        (1 - lx1) * (1 - ly1) * f[0, 0]
        + (1 - lx1) * ly1 * f[0, 1]
        + lx1 * (1 - ly1) * f[1, 0]
        + lx1 * ly1 * f[1, 1]
    )
    true_val_12 = (
        (1 - lx1) * f[0, 1]
        + lx1 * f[1, 1]  # ly2 = 0
    )
    true_val_21 = (
        (1 - lx2) * (1 - ly1) * f[3, 0]
        + (1 - lx2) * ly1 * f[3, 1]   # high node 1.0, no upper
    )
    true_val_22 = (1 - lx2) * f[3, 1]  # ly2 = 0, no upper for 1.0
    true_mg = [[true_val_11, true_val_12],
               [true_val_21, true_val_22]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_per_axis_interpolation():
    """Test different interpolation schemes per axis."""
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    interp = ['linear', 'nearest']
    f = np.arange(1, 9, dtype='float64').reshape((4, 2))
    interpolator = per_axis_interpolator(f, coord_vecs, interp)

    # Evaluate at single point
    val = interpolator([0.3, 0.5])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    # 0.5 equally far from both neighbors -> NN chooses 0.75
    true_val = (1 - l1) * f[0, 1] + l1 * f[1, 1]
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    true_val_1 = (1 - l1) * f[0, 1] + l1 * f[1, 1]
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    true_val_2 = (1 - l1) * f[0, 0]  # only lower left contributes
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    true_val_3 = (1 - l1) * f[3, 1]  # lower left only
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(interpolator(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)

    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 0.85])
    # Indices: (1, 3) x (0, 1)
    lx1 = (0.3 - 0.125) / (0.375 - 0.125)
    lx2 = (1.0 - 0.875) / (0.875 - 0.625)
    true_val_11 = (1 - lx1) * f[0, 0] + lx1 * f[1, 0]
    true_val_12 = ((1 - lx1) * f[0, 1] + lx1 * f[1, 1])
    true_val_21 = (1 - lx2) * f[3, 0]
    true_val_22 = (1 - lx2) * f[3, 1]
    true_mg = [[true_val_11, true_val_12],
               [true_val_21, true_val_22]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_collocation_interpolation_identity():
    """Check if collocation is left-inverse to interpolation."""
    # Interpolation followed by collocation on the same grid should be
    # the identity
    coord_vecs = [[0.125, 0.375, 0.625, 0.875], [0.25, 0.75]]
    f = np.arange(1, 9, dtype='float64').reshape((4, 2))
    interpolators = [
        nearest_interpolator(f, coord_vecs),
        linear_interpolator(f, coord_vecs),
        per_axis_interpolator(f, coord_vecs, interp=['linear', 'nearest']),
    ]

    for interpolator in interpolators:
        mg = sparse_meshgrid(*coord_vecs)
        ident_f = point_collocation(interpolator, mg)
        assert all_almost_equal(ident_f, f)


if __name__ == '__main__':
    odl.util.test_file(__file__)
