# Copyright 2014-2016 The ODL development group
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
from builtins import zip

# External module imports
import pytest
import numpy as np

# ODL imports
from odl.discr.grid import TensorGrid, RegularGrid, sparse_meshgrid
from odl.util.testutils import all_equal


# ---- TensorGrid ---- #


def test_tensorgrid_init():
    sorted1 = np.array([2, 3, 4, 5])
    sorted2 = np.array([-4, -2, 0, 2, 4])
    sorted3 = np.linspace(-1, 2, 50)
    scalar = 0.5

    # Just test if the code runs
    TensorGrid(sorted1)
    TensorGrid(sorted1, sorted2)
    TensorGrid(sorted1, sorted1)
    TensorGrid(sorted1, sorted2, sorted3)
    TensorGrid(sorted2, scalar, sorted1)


def test_tensorgrid_init_raise():
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
        TensorGrid()

    with pytest.raises(ValueError):
        TensorGrid(sorted1, unsorted, sorted2)

    with pytest.raises(ValueError):
        TensorGrid(sorted1, with_dups, sorted2)

    with pytest.raises(ValueError):
        TensorGrid(sorted1, unsorted_with_dups, sorted2)

    with pytest.raises(ValueError):
        TensorGrid(sorted1, with_nan, sorted2)

    with pytest.raises(ValueError):
        TensorGrid(sorted1, with_inf, sorted2)

    with pytest.raises(ValueError):
        TensorGrid(sorted1, empty, sorted2)

    with pytest.raises(ValueError):
        TensorGrid(sorted1, bad_shape)


def test_tensorgrid_ndim():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec3 = np.linspace(-2, 2, 50)

    grid1 = TensorGrid(vec1)
    grid2 = TensorGrid(vec1, vec2, vec3)

    assert grid1.ndim == 1
    assert grid2.ndim == 3


def test_tensorgrid_shape():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.linspace(-2, 2, 50)
    scalar = 0.5

    grid1 = TensorGrid(vec1)
    grid2 = TensorGrid(vec1, vec2)
    grid3 = TensorGrid(scalar, vec2)

    assert grid1.shape == (4,)
    assert grid2.shape == (4, 50)
    assert grid3.shape == (1, 50)


def test_tensorgrid_size():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.linspace(-2, 2, 50)
    scalar = 0.5

    grid1 = TensorGrid(vec1)
    grid2 = TensorGrid(vec1, vec2)
    grid3 = TensorGrid(scalar, vec2)

    assert grid1.size == 4
    assert grid2.size == 200
    assert grid3.size == 50


def test_tensorgrid_minpt_maxpt():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec3 = np.array([-1, 0])
    scalar = 0.5

    grid = TensorGrid(vec1, vec2, vec3)
    assert all_equal(grid.min_pt, (2, -4, -1))
    assert all_equal(grid.max_pt, (5, 4, 0))

    grid = TensorGrid(vec1, scalar, vec2, scalar)
    assert all_equal(grid.min_pt, (2, 0.5, -4, 0.5))
    assert all_equal(grid.max_pt, (5, 0.5, 4, 0.5))


def test_tensorgrid_element():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])

    grid = TensorGrid(vec1, vec2)
    some_pt = grid.element()
    assert some_pt in grid


def test_tensorgrid_equals():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])

    grid1 = TensorGrid(vec1)
    grid2 = TensorGrid(vec1, vec2)
    grid2_again = TensorGrid(vec1, vec2)
    grid2_rev = TensorGrid(vec2, vec1)

    assert grid1 == grid1
    assert not grid1 != grid1
    assert grid2 == grid2
    assert not grid2 != grid2
    assert grid2 == grid2_again
    assert not grid1 == grid2
    assert not grid2 == grid2_rev
    assert not grid2 == (vec1, vec2)

    # Fuzzy check
    grid1 = TensorGrid(vec1, vec2)
    grid2 = TensorGrid(vec1 + (0.1, 0.05, 0, -0.1),
                       vec2 + (0.1, 0.05, 0, -0.1, -0.1))
    assert grid1.approx_equals(grid1, atol=0.0)
    assert grid1.approx_equals(grid2, atol=0.15)
    assert grid2.approx_equals(grid1, atol=0.15)

    grid2 = TensorGrid(vec1 + (0.11, 0.05, 0, -0.1),
                       vec2 + (0.1, 0.05, 0, -0.1, -0.1))
    assert not grid1.approx_equals(grid2, atol=0.1)
    grid2 = TensorGrid(vec1 + (0.1, 0.05, 0, -0.1),
                       vec2 + (0.1, 0.05, 0, -0.11, -0.1))
    assert not grid1.approx_equals(grid2, atol=0.1)


def test_tensorgrid_contains():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])

    grid = TensorGrid(vec1, vec2)

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
    grid = TensorGrid(vec1)
    assert 3 in grid
    assert 7 not in grid


def test_tensorgrid_is_subgrid():
    vec1 = np.array([2, 3, 4, 5])
    vec1_sup = np.array([2, 3, 4, 5, 6, 7])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec2_sup = np.array([-6, -4, -2, 0, 2, 4, 6])
    vec2_sub = np.array([-4, -2, 0, 2])
    scalar = 0.5

    grid = TensorGrid(vec1, vec2)
    assert grid.is_subgrid(grid)

    sup_grid = TensorGrid(vec1_sup, vec2_sup)
    assert grid.is_subgrid(sup_grid)
    assert not sup_grid.is_subgrid(grid)

    not_sup_grid = TensorGrid(vec1_sup, vec2_sub)
    assert not grid.is_subgrid(not_sup_grid)
    assert not not_sup_grid.is_subgrid(grid)

    # Fuzzy check
    fuzzy_vec1_sup = vec1_sup + (0.1, 0.05, 0, -0.1, 0, 0.1)
    fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.1, 0.05)
    fuzzy_sup_grid = TensorGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.15)

    fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.11, 0.05)
    fuzzy_sup_grid = TensorGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
    assert not grid.is_subgrid(fuzzy_sup_grid, atol=0.1)

    # Changes in the non-overlapping part don't matter
    fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.05, 0.11)
    fuzzy_sup_grid = TensorGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.15)

    # With degenerate axis
    grid = TensorGrid(vec1, scalar, vec2)
    sup_grid = TensorGrid(vec1_sup, scalar, vec2_sup)
    assert grid.is_subgrid(sup_grid)

    fuzzy_sup_grid = TensorGrid(vec1, scalar + 0.1, vec2)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.15)


def test_tensorgrid_insert():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    vec3 = np.array([-1, 0])
    scalar = 0.5

    grid = TensorGrid(vec1, vec2)
    grid2 = TensorGrid(scalar, vec3)

    # Test all positions
    ins_grid = grid.insert(0, grid2)
    assert ins_grid == TensorGrid(scalar, vec3, vec1, vec2)

    ins_grid = grid.insert(1, grid2)
    assert ins_grid == TensorGrid(vec1, scalar, vec3, vec2)

    ins_grid = grid.insert(2, grid2)
    assert ins_grid == TensorGrid(vec1, vec2, scalar, vec3)

    ins_grid = grid.insert(-1, grid2)
    assert ins_grid == TensorGrid(vec1, scalar, vec3, vec2)

    with pytest.raises(IndexError):
        grid.insert(3, grid2)

    with pytest.raises(IndexError):
        grid.insert(-4, grid2)


def test_tensorgrid_points():
    vec1 = np.array([2, 3, 4, 5])
    vec2 = np.array([-4, -2, 0, 2, 4])
    scalar = 0.5

    # C ordering
    points = []
    for x1 in vec1:
        for x2 in vec2:
            points.append(np.array((x1, x2), dtype=float))

    grid = TensorGrid(vec1, vec2)
    assert all_equal(points, grid.points())
    assert all_equal(points, grid.points(order='C'))
    assert all_equal(grid.min_pt, grid.points()[0])
    assert all_equal(grid.max_pt, grid.points()[-1])

    # F ordering
    points = []
    for x2 in vec2:
        for x1 in vec1:
            points.append(np.array((x1, x2), dtype=float))

    grid = TensorGrid(vec1, vec2)
    assert all_equal(points, grid.points(order='F'))

    # Degenerate axis 1
    points = []
    for x1 in vec1:
        for x2 in vec2:
            points.append(np.array((scalar, x1, x2), dtype=float))

    grid = TensorGrid(scalar, vec1, vec2)
    assert all_equal(points, grid.points())

    # Degenerate axis 2
    points = []
    for x1 in vec1:
        for x2 in vec2:
            points.append(np.array((x1, scalar, x2), dtype=float))

    grid = TensorGrid(vec1, scalar, vec2)
    assert all_equal(points, grid.points())

    # Degenerate axis 3
    points = []
    for x1 in vec1:
        for x2 in vec2:
            points.append(np.array((x1, x2, scalar), dtype=float))

    grid = TensorGrid(vec1, vec2, scalar)
    assert all_equal(points, grid.points())

    # Bad input
    with pytest.raises(ValueError):
        grid.points(order='A')


def test_tensorgrid_corners():
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

    grid = TensorGrid(vec1, vec2, vec3)
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

    grid = TensorGrid(scalar, vec2, vec3)
    assert all_equal(corners, grid.corners())

    # Degenerate axis 2
    corners = []
    for x1 in minmax1:
        for x3 in minmax3:
            corners.append(np.array((x1, scalar, x3), dtype=float))

    grid = TensorGrid(vec1, scalar, vec3)
    assert all_equal(corners, grid.corners())

    # Degenerate axis 3
    corners = []
    for x1 in minmax1:
        for x2 in minmax2:
            corners.append(np.array((x1, x2, scalar), dtype=float))

    grid = TensorGrid(vec1, vec2, scalar)
    assert all_equal(corners, grid.corners())

    # All degenerate
    corners = [(scalar, scalar)]
    grid = TensorGrid(scalar, scalar)
    assert all_equal(corners, grid.corners())


def test_tensorgrid_meshgrid():
    vec1 = (0, 1)
    vec2 = (-1, 0, 1)
    vec3 = (2, 3, 4, 5)

    # Sparse meshgrid
    mgx = np.array(vec1)[:, None, None]
    mgy = np.array(vec2)[None, :, None]
    mgz = np.array(vec3)[None, None, :]

    grid = TensorGrid(vec1, vec2, vec3)
    xx, yy, zz = grid.meshgrid
    assert all_equal(mgx, xx)
    assert all_equal(mgy, yy)
    assert all_equal(mgz, zz)

    xx, yy, zz = grid.meshgrid
    assert all_equal(mgx, xx)
    assert all_equal(mgy, yy)
    assert all_equal(mgz, zz)


def test_tensorgrid_getitem():
    vec1 = (0, 1, 2)
    vec2 = (-1, 0, 1)
    vec3 = (2, 3, 4, 5)
    vec4 = (1, 3)
    vec1_sub = (1,)
    vec2_sub = (-1,)
    vec3_sub = (3, 4)
    vec4_sub = (1,)

    grid = TensorGrid(vec1, vec2, vec3, vec4)

    # Single indices yield points as an array
    assert all_equal(grid[1, 0, 1, 0], (1.0, -1.0, 3.0, 1.0))

    with pytest.raises(IndexError):
        grid[1, 0, 1, 0, 0]

    with pytest.raises(IndexError):
        grid[0, 3, 0, 0]

    # Slices return new TensorGrid's
    assert grid == grid[...]

    sub_grid = TensorGrid(vec1_sub, vec2_sub, vec3_sub, vec4_sub)
    assert grid[1, 0, 1:3, 0] == sub_grid
    assert grid[-2, :1, 1:3, :1] == sub_grid
    assert grid[1, 0, ..., 1:3, 0] == sub_grid

    sub_grid = TensorGrid(vec1_sub, vec2, vec3, vec4)
    assert grid[1, :, :, :] == sub_grid
    assert grid[1, ...] == sub_grid
    assert grid[1] == sub_grid

    sub_grid = TensorGrid(vec1, vec2, vec3, vec4_sub)
    assert grid[:, :, :, 0] == sub_grid
    assert grid[..., 0] == sub_grid

    sub_grid = TensorGrid(vec1_sub, vec2, vec3, vec4_sub)
    assert grid[1, :, :, 0] == sub_grid
    assert grid[1, ..., 0] == sub_grid
    assert grid[1, :, :, ..., 0] == sub_grid

    # Fewer indices
    assert grid[0] == grid[0, :, :, :]
    assert grid[0, 1:] == grid[0, 1:, :, :]
    assert grid[0, 1:, :-1] == grid[0, 1:, :-1, :]

    # Indexing with lists
    sub_grid = TensorGrid([0, 2], vec2, vec3, vec4)
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
    grid = TensorGrid(vec3)
    assert grid == grid[...]

    sub_grid = TensorGrid(vec3_sub)
    assert grid[1:3] == sub_grid


# ---- RegularGrid ---- #


def test_regulargrid_init():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    # Check correct initialization of coord_vectors
    grid = RegularGrid(minpt, maxpt, shape)
    vec1 = (0.75, 1.25)
    vec2 = (0,)
    vec3 = (-5, -2, 1)
    assert all_equal(grid.coord_vectors, (vec1, vec2, vec3))


def test_regulargrid_init_raise():
    # Check different error scenarios
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)
    nonpos_shape1 = (2, 0, 3)
    nonpos_shape2 = (-2, 1, 3)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt, nonpos_shape1)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt, nonpos_shape2)

    minpt_with_nan = (0.75, 0, np.nan)
    minpt_with_inf = (0.75, 0, np.inf)
    maxpt_with_nan = (1.25, np.nan, 1)
    maxpt_with_inf = (1.25, np.inf, 1)
    shape_with_nan = (2, np.nan, 3)
    shape_with_inf = (2, np.inf, 3)

    with pytest.raises(ValueError):
        RegularGrid(minpt_with_nan, maxpt, shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt_with_inf, maxpt, shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt_with_nan, shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt_with_inf, shape)

    # Shape casting to int raises a TypeError
    with pytest.raises(TypeError):
        RegularGrid(minpt, maxpt, shape_with_nan)

    with pytest.raises(TypeError):
        RegularGrid(minpt, maxpt, shape_with_inf)

    maxpt_smaller_minpt1 = (0.7, 0, 1)
    maxpt_smaller_minpt2 = (1.25, -1, 1)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt_smaller_minpt1, shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt_smaller_minpt2, shape)

    too_short_minpt = (0.75, 0)
    too_long_minpt = (0.75, 0, -5, 2)
    too_short_maxpt = (0, 1)
    too_long_maxpt = (1.25, 0, 1, 25)
    too_short_shape = (2, 3)
    too_long_shape = (2, 1, 4, 3)
    bad_dim_shape = ((1, 2), (3, 4))

    with pytest.raises(ValueError):
        RegularGrid(too_short_minpt, maxpt, shape)

    with pytest.raises(ValueError):
        RegularGrid(too_long_minpt, maxpt, shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt, too_short_maxpt, shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt, too_long_maxpt, shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt, too_short_shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt, too_long_shape)

    maxpt_eq_minpt_at_shape_larger_than_1 = (0.75, 0, 1)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt_eq_minpt_at_shape_larger_than_1,
                    shape)

    with pytest.raises(ValueError):
        RegularGrid(minpt, maxpt, bad_dim_shape)


def test_regulargrid_mid_pt():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    mid_pt = (1, 0, -2)

    grid = RegularGrid(minpt, maxpt, shape)
    assert all_equal(grid.mid_pt, mid_pt)


def test_regulargrid_stride():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    stride = (0.5, 0, 3)

    grid = RegularGrid(minpt, maxpt, shape)
    assert all_equal(grid.stride, stride)


def test_regulargrid_is_subgrid():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 5)

    # Optimized cases
    grid = RegularGrid(minpt, maxpt, shape)
    assert grid.is_subgrid(grid)

    smaller_shape = (1, 1, 5)
    not_sup_grid = RegularGrid(minpt, maxpt, smaller_shape)
    assert not grid.is_subgrid(not_sup_grid)

    larger_minpt = (0.85, 0, -4)
    not_sup_grid = RegularGrid(larger_minpt, maxpt, shape)
    assert not grid.is_subgrid(not_sup_grid)

    smaller_maxpt = (1.15, 0, 0)
    not_sup_grid = RegularGrid(minpt, smaller_maxpt, shape)
    assert not grid.is_subgrid(not_sup_grid)

    # Real checks
    minpt_sup1 = (-0.25, -2, -5)
    maxpt_sup1 = (1.25, 2, 1)
    shape_sup1 = (4, 3, 9)
    sup_grid = RegularGrid(minpt_sup1, maxpt_sup1, shape_sup1)
    assert grid.is_subgrid(sup_grid)
    assert not sup_grid.is_subgrid(grid)

    minpt_sup2 = (0.5, 0, -5)
    maxpt_sup2 = (1.5, 0, 1)
    shape_sup2 = (5, 1, 9)
    sup_grid = RegularGrid(minpt_sup2, maxpt_sup2, shape_sup2)
    assert grid.is_subgrid(sup_grid)
    assert not sup_grid.is_subgrid(grid)

    shape_not_sup1 = (4, 3, 10)
    not_sup_grid = RegularGrid(minpt_sup1, maxpt_sup1, shape_not_sup1)
    assert not grid.is_subgrid(not_sup_grid)
    assert not not_sup_grid.is_subgrid(grid)

    minpt_not_sup1 = (-0.25, -2.5, -5)
    not_sup_grid = RegularGrid(minpt_not_sup1, maxpt_sup1, shape_sup1)
    assert not grid.is_subgrid(not_sup_grid)
    assert not not_sup_grid.is_subgrid(grid)

    maxpt_not_sup1 = (1.35, 2.0001, 1)
    not_sup_grid = RegularGrid(minpt_sup1, maxpt_not_sup1, shape_sup1)
    assert not grid.is_subgrid(not_sup_grid)
    assert not not_sup_grid.is_subgrid(grid)

    # Should also work for TensorGrid's
    vec1_sup = (0.75, 1, 1.25, 7)
    vec2_sup = (0,)
    vec3_sup = (-5, -3.5, -3, -2, -0.5, 0, 1, 9.5)

    tensor_sup_grid = TensorGrid(vec1_sup, vec2_sup, vec3_sup)
    assert grid.is_subgrid(tensor_sup_grid)

    vec1_not_sup = (1, 1.25, 7)
    vec2_not_sup = (0,)
    vec3_not_sup = (-4, -2, 1)

    tensor_not_sup_grid = TensorGrid(vec1_not_sup, vec2_not_sup,
                                     vec3_not_sup)
    assert not grid.is_subgrid(tensor_not_sup_grid)

    # Fuzzy check
    shape_sup = (4, 3, 9)

    minpt_fuzzy_sup1 = (-0.24, -2, -5.01)
    minpt_fuzzy_sup2 = (-0.24, -2, -5)
    maxpt_fuzzy_sup1 = (1.24, 2, 1)
    maxpt_fuzzy_sup2 = (1.25, 2, 1.01)

    fuzzy_sup_grid = RegularGrid(minpt_fuzzy_sup1, maxpt_fuzzy_sup1,
                                 shape_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.015)
    assert not grid.is_subgrid(fuzzy_sup_grid, atol=0.005)

    fuzzy_sup_grid = RegularGrid(minpt_fuzzy_sup2, maxpt_fuzzy_sup2,
                                 shape_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, atol=0.015)
    assert not grid.is_subgrid(fuzzy_sup_grid, atol=0.005)


def test_regulargrid_insert():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    minpt2 = (1, 1)
    maxpt2 = (3, 1)
    shape2 = (5, 1)

    grid = RegularGrid(minpt, maxpt, shape)
    grid2 = RegularGrid(minpt2, maxpt2, shape2)

    # Test all positions
    ins_grid = grid.insert(0, grid2)
    assert isinstance(ins_grid, RegularGrid)
    ins_minpt = minpt2 + minpt
    ins_maxpt = maxpt2 + maxpt
    ins_shape = shape2 + shape
    assert ins_grid == RegularGrid(ins_minpt, ins_maxpt, ins_shape)

    ins_grid = grid.insert(1, grid2)
    ins_minpt = minpt[:1] + minpt2 + minpt[1:]
    ins_maxpt = maxpt[:1] + maxpt2 + maxpt[1:]
    ins_shape = shape[:1] + shape2 + shape[1:]
    assert ins_grid == RegularGrid(ins_minpt, ins_maxpt, ins_shape)

    ins_grid = grid.insert(2, grid2)
    ins_minpt = minpt[:2] + minpt2 + minpt[2:]
    ins_maxpt = maxpt[:2] + maxpt2 + maxpt[2:]
    ins_shape = shape[:2] + shape2 + shape[2:]
    assert ins_grid == RegularGrid(ins_minpt, ins_maxpt, ins_shape)
    ins_grid = grid.insert(-1, grid2)
    assert ins_grid == RegularGrid(ins_minpt, ins_maxpt, ins_shape)

    ins_grid = grid.insert(3, grid2)
    ins_minpt = minpt + minpt2
    ins_maxpt = maxpt + maxpt2
    ins_shape = shape + shape2
    assert ins_grid == RegularGrid(ins_minpt, ins_maxpt, ins_shape)

    # Insert a TensorGrid
    vec = [-1, 0, 3]
    tgrid = TensorGrid(vec)
    ins_tgrid = grid.insert(3, tgrid)
    assert isinstance(ins_tgrid, TensorGrid)
    assert ins_tgrid == TensorGrid(*(grid.coord_vectors + (vec,)))

    with pytest.raises(IndexError):
        grid.insert(4, grid2)

    with pytest.raises(IndexError):
        grid.insert(-5, grid2)

    with pytest.raises(TypeError):
        grid.insert(0, [1, 2])


def test_regulargrid_getitem():
    minpt = (0.75, 0, -5, 4)
    maxpt = (1.25, 0, 1, 13)
    shape = (2, 1, 5, 4)

    grid = RegularGrid(minpt, maxpt, shape)

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

    # Slices return new RegularGrid's
    assert grid == grid[...]

    # Use TensorGrid implementation as reference here
    tensor_grid = TensorGrid(*grid.coord_vectors)

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
    grid = RegularGrid(1, 5, 5)
    assert grid == grid[...]

    sub_grid = RegularGrid(1, 5, 3)
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
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
