# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

import numpy as np
import pytest

import odl
from odl.discr.grid import RectGrid, sparse_meshgrid, uniform_grid
from odl.util.testutils import all_equal

# ---- RectGrid ---- #


def test_RectGrid_init():
    sorted1 = np.array([2, 3, 4, 5])
    sorted2 = np.array([-4, -2, 0, 2, 4])
    sorted3 = np.linspace(-1, 2, 50)
    scalar = 0.5

    # Just test if the code runs
    RectGrid(sorted1)
    RectGrid(sorted1, sorted2)
    RectGrid(sorted1, sorted1)
    RectGrid(sorted1, sorted2, sorted3)
    RectGrid(sorted2, scalar, sorted1)


def test_RectGrid_init_raise():
    # Check different error scenarios

    # Good input
    sorted1 = np.array([2, 3, 4, 5])
    sorted2 = np.array([-4, -2, 0, 2, 4])

    # Bad input
    unsorted = np.arange(4)
    unsorted[2] = -1
    with_dups = np.arange(4)
    with_dups[3] = 2
    unsorted_with_dups = unsorted.copy()
    unsorted_with_dups[3] = 0
    with_nan = np.arange(4, dtype=float)
    with_nan[3] = np.nan
    with_inf = np.arange(4, dtype=float)
    with_inf[3] = np.inf
    empty = np.arange(0)
    bad_shape = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        RectGrid(sorted1, unsorted, sorted2)

    with pytest.raises(ValueError):
        RectGrid(sorted1, with_dups, sorted2)

    with pytest.raises(ValueError):
        RectGrid(sorted1, unsorted_with_dups, sorted2)

    with pytest.raises(ValueError):
        RectGrid(sorted1, with_nan, sorted2)

    with pytest.raises(ValueError):
        RectGrid(sorted1, with_inf, sorted2)

    with pytest.raises(ValueError):
        RectGrid(sorted1, empty, sorted2)

    with pytest.raises(ValueError):
        RectGrid(sorted1, bad_shape)


def test_RectGrid_ndim():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec3 = np.linspace(-2, 2, 50)

    grid1 = RectGrid(vec1)
    grid2 = RectGrid(vec1, vec2, vec3)

    assert grid1.ndim == 1
    assert grid2.ndim == 3


def test_RectGrid_shape():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.linspace(-2, 2, 50)
    scalar = 0.5

    grid1 = RectGrid(vec1)
    grid2 = RectGrid(vec1, vec2)
    grid3 = RectGrid(scalar, vec2)

    assert grid1.shape == (4,)
    assert grid2.shape == (4, 50)
    assert grid3.shape == (1, 50)


def test_RectGrid_size():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.linspace(-2, 2, 50)
    scalar = 0.5

    grid1 = RectGrid(vec1)
    grid2 = RectGrid(vec1, vec2)
    grid3 = RectGrid(scalar, vec2)

    assert grid1.size == 4
    assert grid2.size == 200
    assert grid3.size == 50


def test_RectGrid_minpt_maxpt():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec3 = np.array([-1, 0])
    scalar = 0.5

    grid = RectGrid(vec1, vec2, vec3)
    assert all_equal(grid.min_pt, (2, -4, -1))
    assert all_equal(grid.max_pt, (5, 4, 0))

    grid = RectGrid(vec1, scalar, vec2, scalar)
    assert all_equal(grid.min_pt, (2, 0.5, -4, 0.5))
    assert all_equal(grid.max_pt, (5, 0.5, 4, 0.5))


def test_RectGrid_element():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])

    grid = RectGrid(vec1, vec2)
    some_pt = grid.element()
    assert some_pt in grid


def _test_eq(x, y):
    """Test equality of x and y."""
    assert x == y
    assert not x != y
    assert hash(x) == hash(y)


def _test_neq(x, y):
    """Test non-equality of x and y."""
    assert x != y
    assert not x == y
    assert hash(x) != hash(y)


def test_RectGrid_equals():
    """Test grid equality checks and hash."""
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])

    grid1 = RectGrid(vec1)
    grid2 = RectGrid(vec1, vec2)
    grid2_again = RectGrid(vec1, vec2)
    grid2_rev = RectGrid(vec2, vec1)

    _test_eq(grid1, grid1)
    _test_eq(grid2, grid2)
    _test_eq(grid2, grid2_again)

    _test_neq(grid1, grid2)
    _test_neq(grid2, grid2_rev)
    assert grid2 != (vec1, vec2)

    # Fuzzy check
    grid1 = RectGrid(vec1, vec2)
    grid2 = RectGrid(vec1 + (0.1, 0.05, 0, -0.1),
                     vec2 + (0.1, 0.05, 0, -0.1, -0.1))
    assert grid1.approx_equals(grid1, atol=0.0)
    assert grid1.approx_equals(grid2, atol=0.15)
    assert grid2.approx_equals(grid1, atol=0.15)

    grid2 = RectGrid(vec1 + (0.11, 0.05, 0, -0.1),
                     vec2 + (0.1, 0.05, 0, -0.1, -0.1))
    assert not grid1.approx_equals(grid2, atol=0.1)
    grid2 = RectGrid(vec1 + (0.1, 0.05, 0, -0.1),
                     vec2 + (0.1, 0.05, 0, -0.11, -0.1))
    assert not grid1.approx_equals(grid2, atol=0.1)


def test_RectGrid_contains():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])

    grid = RectGrid(vec1, vec2)

    point_list = []
    for x in vec1:
        for y in vec2:
            point_list.append((x, y))

    assert all(p in grid for p in point_list)

    assert not (0, 0) in grid
    assert (0, 0) not in grid
    assert (2, 0, 0) not in grid
    assert [None, object] not in grid

    # Fuzzy check
    assert grid.approx_contains((2.1, -2.1), atol=0.15)
    assert not grid.approx_contains((2.2, -2.1), atol=0.15)

    # 1d points
    grid = RectGrid(vec1)
    assert 3 in grid
    assert 7 not in grid


def test_RectGrid_is_subgrid():
    vec1 = np.array([2, 3, 4, 5])
    vec1_sup = np.array([2, 3, 4, 5, 6, 7])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec2_sup = np.array([-6, -4, -2, 0, 2, 4, 6])
    vec2_sub = np.array([-4, -2, 0, 2])
    scalar = 0.5

    grid = RectGrid(vec1, vec2)
    assert grid.is_subgrid(grid)

    sup_grid = RectGrid(vec1_sup, vec2_sup)
    assert grid.is_subgrid(sup_grid)
    assert not sup_grid.is_subgrid(grid)

    not_sup_grid = RectGrid(vec1_sup, vec2_sub)
    assert not grid.is_subgrid(not_sup_grid)
    assert not not_sup_grid.is_subgrid(grid)

    # Fuzzy check
    fuzzy_vec1_sup = vec1_sup + (0.1, 0.05, 0, -0.1, 0, 0.1)
    fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.1, 0.05)
    fuzzy_sup_grid = RectGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.15)

    fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.11, 0.05)
    fuzzy_sup_grid = RectGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
    assert not grid.is_subgrid(fuzzy_sup_grid, atol=0.1)

    # Changes in the non-overlapping part don't matter
    fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.05, 0.11)
    fuzzy_sup_grid = RectGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.15)

    # With degenerate axis
    grid = RectGrid(vec1, scalar, vec2)
    sup_grid = RectGrid(vec1_sup, scalar, vec2_sup)
    assert grid.is_subgrid(sup_grid)

    fuzzy_sup_grid = RectGrid(vec1, scalar + 0.1, vec2)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.15)


def test_RectGrid_insert():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec3 = np.array([-1, 0])
    scalar = 0.5

    grid = RectGrid(vec1, vec2)
    grid2 = RectGrid(scalar, vec3)

    # Test all positions
    ins_grid = grid.insert(0, grid2)
    assert ins_grid == RectGrid(scalar, vec3, vec1, vec2)

    ins_grid = grid.insert(1, grid2)
    assert ins_grid == RectGrid(vec1, scalar, vec3, vec2)

    ins_grid = grid.insert(2, grid2)
    assert ins_grid == RectGrid(vec1, vec2, scalar, vec3)

    ins_grid = grid.insert(-1, grid2)
    assert ins_grid == RectGrid(vec1, scalar, vec3, vec2)

    with pytest.raises(IndexError):
        grid.insert(3, grid2)

    with pytest.raises(IndexError):
        grid.insert(-4, grid2)


def test_RectGrid_points():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    scalar = 0.5

    # C ordering
    points = []
    for x1 in vec1:
        for x2 in vec2:
            points.append(np.array((x1, x2), dtype=float))

    grid = RectGrid(vec1, vec2)
    assert all_equal(points, grid.points())
    assert all_equal(points, grid.points(order='C'))
    assert all_equal(grid.min_pt, grid.points()[0])
    assert all_equal(grid.max_pt, grid.points()[-1])

    # F ordering
    points = []
    for x2 in vec2:
        for x1 in vec1:
            points.append(np.array((x1, x2), dtype=float))

    grid = RectGrid(vec1, vec2)
    assert all_equal(points, grid.points(order='F'))

    # Degenerate axis 1
    points = []
    for x1 in vec1:
        for x2 in vec2:
            points.append(np.array((scalar, x1, x2), dtype=float))

    grid = RectGrid(scalar, vec1, vec2)
    assert all_equal(points, grid.points())

    # Degenerate axis 2
    points = []
    for x1 in vec1:
        for x2 in vec2:
            points.append(np.array((x1, scalar, x2), dtype=float))

    grid = RectGrid(vec1, scalar, vec2)
    assert all_equal(points, grid.points())

    # Degenerate axis 3
    points = []
    for x1 in vec1:
        for x2 in vec2:
            points.append(np.array((x1, x2, scalar), dtype=float))

    grid = RectGrid(vec1, vec2, scalar)
    assert all_equal(points, grid.points())

    # Bad input
    with pytest.raises(ValueError):
        grid.points(order='A')


def test_RectGrid_corners():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec3 = np.array([-1, 0])
    scalar = 0.5

    minmax1 = (vec1[0], vec1[-1])
    minmax2 = (vec2[0], vec2[-1])
    minmax3 = (vec3[0], vec3[-1])

    # C ordering
    corners = []
    for x1 in minmax1:
        for x2 in minmax2:
            for x3 in minmax3:
                corners.append(np.array((x1, x2, x3), dtype=float))

    grid = RectGrid(vec1, vec2, vec3)
    assert all_equal(corners, grid.corners())
    assert all_equal(corners, grid.corners(order='C'))

    # minpt and maxpt should appear at the beginning and the end, resp.
    assert all_equal(grid.min_pt, grid.corners()[0])
    assert all_equal(grid.max_pt, grid.corners()[-1])

    # F ordering
    corners = []
    for x3 in minmax3:
        for x2 in minmax2:
            for x1 in minmax1:
                corners.append(np.array((x1, x2, x3), dtype=float))

    assert all_equal(corners, grid.corners(order='F'))

    # Degenerate axis 1
    corners = []
    for x2 in minmax2:
        for x3 in minmax3:
            corners.append(np.array((scalar, x2, x3), dtype=float))

    grid = RectGrid(scalar, vec2, vec3)
    assert all_equal(corners, grid.corners())

    # Degenerate axis 2
    corners = []
    for x1 in minmax1:
        for x3 in minmax3:
            corners.append(np.array((x1, scalar, x3), dtype=float))

    grid = RectGrid(vec1, scalar, vec3)
    assert all_equal(corners, grid.corners())

    # Degenerate axis 3
    corners = []
    for x1 in minmax1:
        for x2 in minmax2:
            corners.append(np.array((x1, x2, scalar), dtype=float))

    grid = RectGrid(vec1, vec2, scalar)
    assert all_equal(corners, grid.corners())

    # All degenerate
    corners = [(scalar, scalar)]
    grid = RectGrid(scalar, scalar)
    assert all_equal(corners, grid.corners())


def test_RectGrid_meshgrid():
    vec1 = (0, 1)
    vec2 = (-1, 0, 1)
    vec3 = (2, 3, 4, 5)

    # Sparse meshgrid
    mgx = np.array(vec1)[:, None, None]
    mgy = np.array(vec2)[None, :, None]
    mgz = np.array(vec3)[None, None, :]

    grid = RectGrid(vec1, vec2, vec3)
    xx, yy, zz = grid.meshgrid
    assert all_equal(mgx, xx)
    assert all_equal(mgy, yy)
    assert all_equal(mgz, zz)

    xx, yy, zz = grid.meshgrid
    assert all_equal(mgx, xx)
    assert all_equal(mgy, yy)
    assert all_equal(mgz, zz)


def test_RectGrid_getitem():
    vec1 = (0, 1, 2)
    vec2 = (-1, 0, 1)
    vec3 = (2, 3, 4, 5)
    vec4 = (1, 3)
    vec1_sub = (1,)
    vec2_sub = (-1,)
    vec3_sub = (3, 4)
    vec4_sub = (1,)

    grid = RectGrid(vec1, vec2, vec3, vec4)

    # Single indices yield points as an array
    assert all_equal(grid[1, 0, 1, 0], (1.0, -1.0, 3.0, 1.0))

    with pytest.raises(IndexError):
        grid[1, 0, 1, 0, 0]

    with pytest.raises(IndexError):
        grid[0, 3, 0, 0]

    # Slices return new RectGrid's
    assert grid == grid[...]

    sub_grid = RectGrid(vec1_sub, vec2_sub, vec3_sub, vec4_sub)
    assert grid[1, 0, 1:3, 0] == sub_grid
    assert grid[-2, :1, 1:3, :1] == sub_grid
    assert grid[1, 0, ..., 1:3, 0] == sub_grid

    sub_grid = RectGrid(vec1_sub, vec2, vec3, vec4)
    assert grid[1, :, :, :] == sub_grid
    assert grid[1, ...] == sub_grid
    assert grid[1] == sub_grid

    sub_grid = RectGrid(vec1, vec2, vec3, vec4_sub)
    assert grid[:, :, :, 0] == sub_grid
    assert grid[..., 0] == sub_grid

    sub_grid = RectGrid(vec1_sub, vec2, vec3, vec4_sub)
    assert grid[1, :, :, 0] == sub_grid
    assert grid[1, ..., 0] == sub_grid
    assert grid[1, :, :, ..., 0] == sub_grid

    # Fewer indices
    assert grid[0] == grid[0, :, :, :]
    assert grid[0, 1:] == grid[0, 1:, :, :]
    assert grid[0, 1:, :-1] == grid[0, 1:, :-1, :]

    # Indexing with lists
    sub_grid = RectGrid([0, 2], vec2, vec3, vec4)
    assert grid[[0, 2]] == sub_grid
    assert grid[[0, 1]] == grid[0:2, ...]

    # Two ellipses not allowed
    with pytest.raises(ValueError):
        grid[1, ..., ..., 0]

    # Too many indices
    with pytest.raises(IndexError):
        grid[1, 0, 1:2, 0, :]

    # Empty axes not allowed
    with pytest.raises(ValueError):
        grid[1, 0, None, 0]

    # One-dimensional grid
    grid = RectGrid(vec3)
    assert grid == grid[...]

    sub_grid = RectGrid(vec3_sub)
    assert grid[1:3] == sub_grid


def test_empty_grid():
    """Check if empty grids behave as expected and all methods work."""
    grid = RectGrid()

    assert grid.ndim == grid.size == len(grid) == 0
    assert grid.shape == ()

    assert grid.coord_vectors == ()
    assert grid.nondegen_byaxis == ()
    assert np.array_equal(grid.min_pt, [])
    assert np.array_equal(grid.max_pt, [])
    assert np.array_equal(grid.mid_pt, [])
    assert np.array_equal(grid.stride, [])
    assert np.array_equal(grid.extent, [])
    out = np.array([])
    grid.min(out=out)
    grid.max(out=out)

    assert grid.is_uniform
    assert grid.convex_hull() == odl.IntervalProd([], [])

    same = RectGrid()
    assert grid == same
    assert hash(grid) == hash(same)
    other = RectGrid([0, 2, 3])
    assert grid != other
    assert grid.is_subgrid(other)
    assert [] in grid
    assert 1.0 not in grid

    assert grid.insert(0, other) == other
    assert other.insert(0, grid) == other
    assert other.insert(1, grid) == other
    assert grid.squeeze() == grid
    assert np.array_equal(grid.points(), np.array([]).reshape((0, 0)))
    assert grid.corner_grid() == grid
    assert np.array_equal(grid.corners(), np.array([]).reshape((0, 0)))
    assert grid.meshgrid == ()

    assert grid[[]] == grid
    assert np.array_equal(np.asarray(grid), np.array([]).reshape((0, 0)))
    assert grid == uniform_grid([], [], ())
    repr(grid)


# ---- uniform_grid ---- #


def test_uniform_init():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    # Check correct initialization of coord_vectors
    grid = uniform_grid(minpt, maxpt, shape)
    vec1 = (0.75, 1.25)
    vec2 = (0,)
    vec3 = (-5, -2, 1)
    assert all_equal(grid.coord_vectors, (vec1, vec2, vec3))


def test_uniform_init_raise():
    # Check different error scenarios
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)
    nonpos_shape1 = (2, 0, 3)
    nonpos_shape2 = (-2, 1, 3)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt, nonpos_shape1)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt, nonpos_shape2)

    minpt_with_nan = (0.75, 0, np.nan)
    minpt_with_inf = (0.75, 0, np.inf)
    maxpt_with_nan = (1.25, np.nan, 1)
    maxpt_with_inf = (1.25, np.inf, 1)
    shape_with_nan = (2, np.nan, 3)
    shape_with_inf = (2, np.inf, 3)

    with pytest.raises(ValueError):
        uniform_grid(minpt_with_nan, maxpt, shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt_with_inf, maxpt, shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt_with_nan, shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt_with_inf, shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt, shape_with_nan)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt, shape_with_inf)

    maxpt_smaller_minpt1 = (0.7, 0, 1)
    maxpt_smaller_minpt2 = (1.25, -1, 1)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt_smaller_minpt1, shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt_smaller_minpt2, shape)

    too_short_minpt = (0.75, 0)
    too_long_minpt = (0.75, 0, -5, 2)
    too_short_maxpt = (0, 1)
    too_long_maxpt = (1.25, 0, 1, 25)
    too_short_shape = (2, 3)
    too_long_shape = (2, 1, 4, 3)
    bad_dim_shape = ((1, 2), (3, 4))

    with pytest.raises(ValueError):
        uniform_grid(too_short_minpt, maxpt, shape)

    with pytest.raises(ValueError):
        uniform_grid(too_long_minpt, maxpt, shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, too_short_maxpt, shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, too_long_maxpt, shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt, too_short_shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt, too_long_shape)

    maxpt_eq_minpt_at_shape_larger_than_1 = (0.75, 0, 1)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt_eq_minpt_at_shape_larger_than_1,
                     shape)

    with pytest.raises(ValueError):
        uniform_grid(minpt, maxpt, bad_dim_shape)


def test_uniform_mid_pt():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    mid_pt = (1, 0, -2)

    grid = uniform_grid(minpt, maxpt, shape)
    assert all_equal(grid.mid_pt, mid_pt)


def test_uniform_stride():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    stride = (0.5, 0, 3)

    grid = uniform_grid(minpt, maxpt, shape)
    assert all_equal(grid.stride, stride)


def test_uniform_is_subgrid():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 5)

    # Optimized cases
    grid = uniform_grid(minpt, maxpt, shape)
    assert grid.is_subgrid(grid)

    smaller_shape = (1, 1, 5)
    not_sup_grid = uniform_grid(minpt, maxpt, smaller_shape)
    assert not grid.is_subgrid(not_sup_grid)

    larger_minpt = (0.85, 0, -4)
    not_sup_grid = uniform_grid(larger_minpt, maxpt, shape)
    assert not grid.is_subgrid(not_sup_grid)

    smaller_maxpt = (1.15, 0, 0)
    not_sup_grid = uniform_grid(minpt, smaller_maxpt, shape)
    assert not grid.is_subgrid(not_sup_grid)

    # Real checks
    minpt_sup1 = (-0.25, -2, -5)
    maxpt_sup1 = (1.25, 2, 1)
    shape_sup1 = (4, 3, 9)
    sup_grid = uniform_grid(minpt_sup1, maxpt_sup1, shape_sup1)
    assert grid.is_subgrid(sup_grid)
    assert not sup_grid.is_subgrid(grid)

    minpt_sup2 = (0.5, 0, -5)
    maxpt_sup2 = (1.5, 0, 1)
    shape_sup2 = (5, 1, 9)
    sup_grid = uniform_grid(minpt_sup2, maxpt_sup2, shape_sup2)
    assert grid.is_subgrid(sup_grid)
    assert not sup_grid.is_subgrid(grid)

    shape_not_sup1 = (4, 3, 10)
    not_sup_grid = uniform_grid(minpt_sup1, maxpt_sup1, shape_not_sup1)
    assert not grid.is_subgrid(not_sup_grid)
    assert not not_sup_grid.is_subgrid(grid)

    minpt_not_sup1 = (-0.25, -2.5, -5)
    not_sup_grid = uniform_grid(minpt_not_sup1, maxpt_sup1, shape_sup1)
    assert not grid.is_subgrid(not_sup_grid)
    assert not not_sup_grid.is_subgrid(grid)

    maxpt_not_sup1 = (1.35, 2.0001, 1)
    not_sup_grid = uniform_grid(minpt_sup1, maxpt_not_sup1, shape_sup1)
    assert not grid.is_subgrid(not_sup_grid)
    assert not not_sup_grid.is_subgrid(grid)

    # Should also work for RectGrid's
    vec1_sup = (0.75, 1, 1.25, 7)
    vec2_sup = (0,)
    vec3_sup = (-5, -3.5, -3, -2, -0.5, 0, 1, 9.5)

    tensor_sup_grid = RectGrid(vec1_sup, vec2_sup, vec3_sup)
    assert grid.is_subgrid(tensor_sup_grid)

    vec1_not_sup = (1, 1.25, 7)
    vec2_not_sup = (0,)
    vec3_not_sup = (-4, -2, 1)

    tensor_not_sup_grid = RectGrid(vec1_not_sup, vec2_not_sup, vec3_not_sup)
    assert not grid.is_subgrid(tensor_not_sup_grid)

    # Fuzzy check
    shape_sup = (4, 3, 9)

    minpt_fuzzy_sup1 = (-0.24, -2, -5.01)
    minpt_fuzzy_sup2 = (-0.24, -2, -5)
    maxpt_fuzzy_sup1 = (1.24, 2, 1)
    maxpt_fuzzy_sup2 = (1.25, 2, 1.01)

    fuzzy_sup_grid = uniform_grid(minpt_fuzzy_sup1, maxpt_fuzzy_sup1,
                                  shape_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.015)
    assert not grid.is_subgrid(fuzzy_sup_grid, atol=0.005)

    fuzzy_sup_grid = uniform_grid(minpt_fuzzy_sup2, maxpt_fuzzy_sup2,
                                  shape_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.015)
    assert not grid.is_subgrid(fuzzy_sup_grid, atol=0.005)


def test_uniform_insert():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    minpt2 = (1, 1)
    maxpt2 = (3, 1)
    shape2 = (5, 1)

    grid = uniform_grid(minpt, maxpt, shape)
    grid2 = uniform_grid(minpt2, maxpt2, shape2)

    # Test all positions
    ins_grid = grid.insert(0, grid2)
    ins_minpt = minpt2 + minpt
    ins_maxpt = maxpt2 + maxpt
    ins_shape = shape2 + shape
    assert ins_grid == uniform_grid(ins_minpt, ins_maxpt, ins_shape)

    ins_grid = grid.insert(1, grid2)
    ins_minpt = minpt[:1] + minpt2 + minpt[1:]
    ins_maxpt = maxpt[:1] + maxpt2 + maxpt[1:]
    ins_shape = shape[:1] + shape2 + shape[1:]
    assert ins_grid == uniform_grid(ins_minpt, ins_maxpt, ins_shape)

    ins_grid = grid.insert(2, grid2)
    ins_minpt = minpt[:2] + minpt2 + minpt[2:]
    ins_maxpt = maxpt[:2] + maxpt2 + maxpt[2:]
    ins_shape = shape[:2] + shape2 + shape[2:]
    assert ins_grid == uniform_grid(ins_minpt, ins_maxpt, ins_shape)
    ins_grid = grid.insert(-1, grid2)
    assert ins_grid == uniform_grid(ins_minpt, ins_maxpt, ins_shape)

    ins_grid = grid.insert(3, grid2)
    ins_minpt = minpt + minpt2
    ins_maxpt = maxpt + maxpt2
    ins_shape = shape + shape2
    assert ins_grid == uniform_grid(ins_minpt, ins_maxpt, ins_shape)

    # Insert a RectGrid
    vec = [-1, 0, 3]
    tgrid = RectGrid(vec)
    ins_tgrid = grid.insert(3, tgrid)
    assert isinstance(ins_tgrid, RectGrid)
    assert ins_tgrid == RectGrid(*(grid.coord_vectors + (vec,)))

    with pytest.raises(IndexError):
        grid.insert(4, grid2)

    with pytest.raises(IndexError):
        grid.insert(-5, grid2)

    with pytest.raises(TypeError):
        grid.insert(0, [1, 2])


def test_uniform_getitem():
    minpt = (0.75, 0, -5, 4)
    maxpt = (1.25, 0, 1, 13)
    shape = (2, 1, 5, 4)

    grid = uniform_grid(minpt, maxpt, shape)

    # Single indices yield points as an array
    indices = [1, 0, 1, 1]
    values = [vec[i] for i, vec in zip(indices, grid.coord_vectors)]
    assert all_equal(grid[1, 0, 1, 1], values)

    indices = [0, 0, 4, 3]
    values = [vec[i] for i, vec in zip(indices, grid.coord_vectors)]
    assert all_equal(grid[0, 0, 4, 3], values)

    with pytest.raises(IndexError):
        grid[1, 0, 1, 2, 0]

    with pytest.raises(IndexError):
        grid[1, 1, 6, 2]

    with pytest.raises(IndexError):
        grid[1, 0, 4, 6]

    # Slices return uniform grids
    assert grid == grid[...]

    # Use RectGrid implementation as reference here
    tensor_grid = RectGrid(*grid.coord_vectors)

    test_slice = np.s_[1, :, ::2, ::3]
    assert all_equal(grid[test_slice].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[1:2, :, ::2, ::3].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[1:2, :, ::2, ..., ::3].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)

    test_slice = np.s_[0:1, :, :, 2:4]
    assert all_equal(grid[test_slice].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[:1, :, :, 2:].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[:-1, ..., 2:].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)

    test_slice = np.s_[:, 0, :, :]
    assert all_equal(grid[test_slice].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[:, 0, ...].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[0:2, :, ...].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[...].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)

    test_slice = np.s_[:, :, 0::2, :]
    assert all_equal(grid[test_slice].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[..., 0::2, :].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)

    test_slice = np.s_[..., 1, :]
    assert all_equal(grid[test_slice].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)
    assert all_equal(grid[:, :, 1, :].coord_vectors,
                     tensor_grid[test_slice].coord_vectors)

    # Fewer indices
    assert grid[1:] == grid[1:, :, :, :]
    assert grid[1:, 0] == grid[1:, 0, :, :]
    assert grid[1:, 0, :-1] == grid[1:, 0, :-1, :]

    # Two ellipses not allowed
    with pytest.raises(ValueError):
        grid[1, ..., ..., 0]

    # Too many axes
    with pytest.raises(IndexError):
        grid[1, 0, 1:2, 0, :]

    # New axes not supported
    with pytest.raises(ValueError):
        grid[1, 0, None, 1, 0]

    # Empty axes not allowed
    with pytest.raises(ValueError):
        grid[1, 0, 0:0, 1]
    with pytest.raises(ValueError):
        grid[1, 1:, 0, 1]

    # One-dimensional grid
    grid = uniform_grid(1, 5, 5)
    assert grid == grid[...]

    sub_grid = uniform_grid(1, 5, 3)
    assert grid[::2], sub_grid


# ---- Utilities ---- #

def test_sparse_meshgrid():

    # One array only
    x = np.zeros(2)
    true_mg = (x,)
    assert all_equal(sparse_meshgrid(x), true_mg)

    x = np.zeros((2, 2))
    true_mg = (x,)
    assert all_equal(sparse_meshgrid(x), true_mg)

    # Two arrays
    x, y = np.zeros(2), np.zeros(3)
    true_mg = (x[:, None], y[None, :])
    mg = sparse_meshgrid(x, y)
    assert all_equal(mg, true_mg)
    assert all(vec.flags.c_contiguous for vec in mg)

    # Array-like input
    x, y = [1, 2, 3], [4, 5, 6]
    true_mg = (np.array(x)[:, None], np.array(y)[None, :])
    mg = sparse_meshgrid(x, y)
    assert all_equal(mg, true_mg)
    assert all(vec.flags.c_contiguous for vec in mg)


if __name__ == '__main__':
    odl.util.test_file(__file__)
