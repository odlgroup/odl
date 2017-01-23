# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `discr_mappings`."""

from __future__ import division
import numpy as np
import pytest

import odl
from odl.discr.grid import sparse_meshgrid
from odl.discr.discr_mappings import (
    PointCollocation, NearestInterpolation, LinearInterpolation,
    PerAxisInterpolation)
from odl.util.testutils import (
    all_almost_equal, all_equal, almost_equal)


def test_nearest_interpolation_1d_complex(tspace_impl):
    intv = odl.IntervalProd(0, 1)
    part = odl.uniform_partition_fromintv(intv, 5, nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    space = odl.FunctionSpace(intv, field=odl.ComplexNumbers())
    dspace = odl.cn(part.shape)
    interp_op = NearestInterpolation(space, part, dspace)
    function = interp_op([0 + 1j, 1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j])

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
    intv = odl.IntervalProd(0, 1)
    part = odl.uniform_partition_fromintv(intv, 5, nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    space = odl.FunctionSpace(intv)
    dspace = odl.rn(part.shape)

    # 'left' variant
    interp_op = NearestInterpolation(space, part, dspace, variant='left')
    function = interp_op([0, 1, 2, 3, 4])

    # Testing two midpoints and the extreme values
    pts = np.array([0.4, 0.8, 0.0, 1.0])
    true_arr = [1, 3, 0, 4]
    assert all_equal(function(pts), true_arr)

    # 'right' variant
    interp_op = NearestInterpolation(space, part, dspace, variant='right')
    function = interp_op([0, 1, 2, 3, 4])

    # Testing two midpoints and the extreme values
    pts = np.array([0.4, 0.8, 0.0, 1.0])
    true_arr = [2, 4, 0, 4]
    assert all_equal(function(pts), true_arr)


def test_nearest_interpolation_2d_float():
    rect = odl.IntervalProd([0, 0], [1, 1])
    part = odl.uniform_partition_fromintv(rect, [4, 2], nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    space = odl.FunctionSpace(rect)
    dspace = odl.rn(part.shape)
    interp_op = NearestInterpolation(space, part, dspace)
    function = interp_op(np.reshape([0, 1, 2, 3, 4, 5, 6, 7], part.shape))

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
    rect = odl.IntervalProd([0, 0], [1, 1])
    part = odl.uniform_partition_fromintv(rect, [4, 2], nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    space = odl.FunctionSet(rect, odl.Strings(1))
    dspace = odl.tensor_set(part.shape, dtype='U1')
    interp_op = NearestInterpolation(space, part, dspace)
    values = np.array([c for c in 'mystring']).reshape(dspace.shape)
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


def test_linear_interpolation_1d():
    intv = odl.IntervalProd(0, 1)
    part = odl.uniform_partition_fromintv(intv, 5, nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    space = odl.FunctionSpace(intv)
    dspace = odl.rn(part.shape)
    interp_op = LinearInterpolation(space, part, dspace)
    function = interp_op([1, 2, 3, 4, 5])

    # Evaluate at single point
    val = function(0.35)
    true_val = 0.75 * 2 + 0.25 * 3
    assert almost_equal(val, true_val)

    # Input array, with and without output array
    pts = np.array([0.4, 0.0, 0.65, 0.95])
    true_arr = [2.5, 0.5, 3.75, 3.75]
    assert all_almost_equal(function(pts), true_arr)


def test_linear_interpolation_2d():
    rect = odl.IntervalProd([0, 0], [1, 1])
    part = odl.uniform_partition_fromintv(rect, [4, 2], nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    space = odl.FunctionSpace(rect)
    dspace = odl.rn(part.shape)
    interp_op = LinearInterpolation(space, part, dspace)
    values = np.arange(1, 9, dtype='float64').reshape(part.shape)
    function = interp_op(values)
    rvals = values.reshape([4, 2])

    # Evaluate at single point
    val = function([0.3, 0.6])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val = ((1 - l1) * (1 - l2) * rvals[0, 0] +
                (1 - l1) * l2 * rvals[0, 1] +
                l1 * (1 - l2) * rvals[1, 0] +
                l1 * l2 * rvals[1, 1])
    assert almost_equal(val, true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val_1 = ((1 - l1) * (1 - l2) * rvals[0, 0] +
                  (1 - l1) * l2 * rvals[0, 1] +
                  l1 * (1 - l2) * rvals[1, 0] +
                  l1 * l2 * rvals[1, 1])
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    # l2 = 0
    true_val_2 = (1 - l1) * rvals[0, 0]  # only lower left contributes
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    l2 = (1.0 - 0.75) / (0.75 - 0.25)
    true_val_3 = (1 - l1) * (1 - l2) * rvals[3, 1]  # lower left only
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(function(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    function(pts.T, out=out)
    assert all_equal(out, true_arr)

    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 0.75])
    # Indices: (1, 3) x (0, 1)
    lx1 = (0.3 - 0.125) / (0.375 - 0.125)
    lx2 = (1.0 - 0.875) / (0.875 - 0.625)
    ly1 = (0.4 - 0.25) / (0.75 - 0.25)
    # ly2 = 0
    true_val_11 = ((1 - lx1) * (1 - ly1) * rvals[0, 0] +
                   (1 - lx1) * ly1 * rvals[0, 1] +
                   lx1 * (1 - ly1) * rvals[1, 0] +
                   lx1 * ly1 * rvals[1, 1])
    true_val_12 = ((1 - lx1) * rvals[0, 1] + lx1 * rvals[1, 1])  # ly2 = 0
    true_val_21 = ((1 - lx2) * (1 - ly1) * rvals[3, 0] +
                   (1 - lx2) * ly1 * rvals[3, 1])  # high node 1.0, no upper
    true_val_22 = (1 - lx2) * rvals[3, 1]  # ly2 = 0, no upper for 1.0
    true_mg = [[true_val_11, true_val_12],
               [true_val_21, true_val_22]]
    assert all_equal(function(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    function(mg, out=out)
    assert all_equal(out, true_mg)


def test_per_axis_interpolation():
    rect = odl.IntervalProd([0, 0], [1, 1])
    part = odl.uniform_partition_fromintv(rect, [4, 2], nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    space = odl.FunctionSpace(rect)
    dspace = odl.rn(part.shape)
    schemes = ['linear', 'nearest']
    variants = [None, 'right']
    interp_op = PerAxisInterpolation(space, part, dspace, schemes=schemes,
                                     nn_variants=variants)
    values = np.arange(1, 9, dtype='float64').reshape(part.shape)
    function = interp_op(values)
    rvals = values.reshape([4, 2])

    # Evaluate at single point
    val = function([0.3, 0.5])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    # 0.5 equally far from both neighbors -> 'right' chooses 0.75
    true_val = (1 - l1) * rvals[0, 1] + l1 * rvals[1, 1]
    assert almost_equal(val, true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    true_val_1 = (1 - l1) * rvals[0, 1] + l1 * rvals[1, 1]
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    true_val_2 = (1 - l1) * rvals[0, 0]  # only lower left contributes
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    true_val_3 = (1 - l1) * rvals[3, 1]  # lower left only
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(function(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    function(pts.T, out=out)
    assert all_equal(out, true_arr)

    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 0.85])
    # Indices: (1, 3) x (0, 1)
    lx1 = (0.3 - 0.125) / (0.375 - 0.125)
    lx2 = (1.0 - 0.875) / (0.875 - 0.625)
    true_val_11 = (1 - lx1) * rvals[0, 0] + lx1 * rvals[1, 0]
    true_val_12 = ((1 - lx1) * rvals[0, 1] + lx1 * rvals[1, 1])
    true_val_21 = (1 - lx2) * rvals[3, 0]
    true_val_22 = (1 - lx2) * rvals[3, 1]
    true_mg = [[true_val_11, true_val_12],
               [true_val_21, true_val_22]]
    assert all_equal(function(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    function(mg, out=out)
    assert all_equal(out, true_mg)


def test_collocation_interpolation_identity():
    # Check if interpolation followed by collocation on the same grid
    # is the identity
    rect = odl.IntervalProd([0, 0], [1, 1])
    part = odl.uniform_partition_fromintv(rect, [4, 2])
    space = odl.FunctionSpace(rect)
    dspace = odl.rn(part.shape)

    coll_op = PointCollocation(space, part, dspace)
    interp_ops = [
        NearestInterpolation(space, part, dspace, variant='left'),
        NearestInterpolation(space, part, dspace, variant='right'),
        LinearInterpolation(space, part, dspace),
        PerAxisInterpolation(space, part, dspace,
                             schemes=['linear', 'nearest'])]

    values = np.arange(1, 9, dtype='float64').reshape(dspace.shape)

    for interp_op in interp_ops:
        ident_values = coll_op(interp_op(values))
        assert all_almost_equal(ident_values, values)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
