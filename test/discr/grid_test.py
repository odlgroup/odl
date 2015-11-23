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
from builtins import range, str, zip

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl import TensorGrid, RegularGrid
from odl.util.testutils import all_equal


def test_init():
    sorted1 = np.arange(2, 6)
    sorted2 = np.arange(-4, 5, 2)
    sorted3 = np.linspace(-1, 2, 50)
    scalar = 0.5

    # Just test if the code runs
    TensorGrid(sorted1)
    TensorGrid(sorted1, sorted2)
    TensorGrid(sorted1, sorted1)
    TensorGrid(sorted1, sorted2, sorted3)
    TensorGrid(sorted2, scalar, sorted1)

    # Check different error scenarios
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


def test_ndim():
    vec1 = np.arange(2, 6)
    vec2 = np.arange(-4, 5, 2)
    vec3 = np.linspace(-2, 2, 50)

    grid1 = TensorGrid(vec1)
    grid2 = TensorGrid(vec1, vec2, vec3)

    assert grid1.ndim == 1
    assert grid2.ndim == 3


def test_shape():
    vec1 = np.arange(2, 6)
    vec2 = np.linspace(-2, 2, 50)
    scalar = 0.5

    grid1 = TensorGrid(vec1)
    grid2 = TensorGrid(vec1, vec2)
    grid3 = TensorGrid(scalar, vec2)

    assert grid1.shape == (4,)
    assert grid2.shape == (4, 50)
    assert grid3.shape == (1, 50)


def test_ntotal():
    vec1 = np.arange(2, 6)
    vec2 = np.linspace(-2, 2, 50)
    scalar = 0.5

    grid1 = TensorGrid(vec1)
    grid2 = TensorGrid(vec1, vec2)
    grid3 = TensorGrid(scalar, vec2)

    assert grid1.ntotal == 4
    assert grid2.ntotal == 200
    assert grid3.ntotal == 50


def test_minpt_maxpt():
    vec1 = np.arange(2, 6)
    vec2 = np.arange(-4, 5, 2)
    vec3 = np.arange(-1, 1)
    scalar = 0.5

    grid = TensorGrid(vec1, vec2, vec3)
    assert all_equal(grid.min_pt, (2, -4, -1))
    assert all_equal(grid.max_pt, (5, 4, 0))

    grid = TensorGrid(vec1, scalar, vec2, scalar)
    assert all_equal(grid.min_pt, (2, 0.5, -4, 0.5))
    assert all_equal(grid.max_pt, (5, 0.5, 4, 0.5))


def test_element():
    vec1 = np.arange(2, 6)
    vec2 = np.arange(-4, 5, 2)

    grid = TensorGrid(vec1, vec2)
    some_pt = grid.element()
    assert some_pt in grid


def test_min_max():
    vec1 = np.arange(2, 6)
    vec2 = np.arange(-4, 5, 2)
    vec3 = np.arange(-1, 1)
    scalar = 0.5

    grid = TensorGrid(vec1, vec2, vec3, as_midp=False)
    assert all_equal(grid.min(), (2, -4, -1))
    assert all_equal(grid.max(), (5, 4, 0))

    grid = TensorGrid(vec1, scalar, vec2, scalar, as_midp=False)
    assert all_equal(grid.min(), (2, 0.5, -4, 0.5))
    assert all_equal(grid.max(), (5, 0.5, 4, 0.5))

    grid = TensorGrid(vec1, vec2, vec3, as_midp=True)
    assert all_equal(grid.min(), (1.5, -5, -1.5))
    assert all_equal(grid.max(), (5.5, 5, 0.5))

    grid = TensorGrid(vec1, scalar, vec2, scalar, as_midp=True)
    assert all_equal(grid.min(), (1.5, 0.5, -5, 0.5))
    assert all_equal(grid.max(), (5.5, 0.5, 5, 0.5))


def test_equals():
    vec1 = np.arange(2, 6)
    vec2 = np.arange(-4, 5, 2)

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
    assert grid1.approx_equals(grid2, tol=0.15)
    assert grid2.approx_equals(grid1, tol=0.15)

    grid2 = TensorGrid(vec1 + (0.11, 0.05, 0, -0.1),
                       vec2 + (0.1, 0.05, 0, -0.1, -0.1))
    assert not grid1.approx_equals(grid2, tol=0.1)
    grid2 = TensorGrid(vec1 + (0.1, 0.05, 0, -0.1),
                       vec2 + (0.1, 0.05, 0, -0.11, -0.1))
    assert not grid1.approx_equals(grid2, tol=0.1)


def test_contains():
    vec1 = np.arange(2, 6)
    vec2 = np.arange(-4, 5, 2)

    grid = TensorGrid(vec1, vec2)

    point_list = []
    for x in vec1:
        for y in vec2:
            point_list.append((x, y))

    assert all(p in grid for p in point_list)

    assert not (0, 0) in grid
    assert (0, 0) not in grid
    assert (2, 0, 0) not in grid

    # Fuzzy check
    assert grid.approx_contains((2.1, -2.1), tol=0.15)
    assert not grid.approx_contains((2.2, -2.1), tol=0.15)

    # 1d points
    grid = TensorGrid(vec1)
    assert 3 in grid
    assert 7 not in grid


def test_tensor_is_subgrid():
    vec1 = np.arange(2, 6)
    vec1_sup = np.arange(2, 8)
    vec2 = np.arange(-4, 5, 2)
    vec2_sup = np.arange(-6, 7, 2)
    vec2_sub = np.arange(-4, 3, 2)
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
    assert grid.is_subgrid(fuzzy_sup_grid, tol=0.15)

    fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.11, 0.05)
    fuzzy_sup_grid = TensorGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
    assert not grid.is_subgrid(fuzzy_sup_grid, tol=0.1)

    # Changes in the non-overlapping part don't matter
    fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.05, 0.11)
    fuzzy_sup_grid = TensorGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, tol=0.15)

    # With degenerate axis
    grid = TensorGrid(vec1, scalar, vec2)
    sup_grid = TensorGrid(vec1_sup, scalar, vec2_sup)
    assert grid.is_subgrid(sup_grid)

    fuzzy_sup_grid = TensorGrid(vec1, scalar + 0.1, vec2)
    assert grid.is_subgrid(fuzzy_sup_grid, tol=0.15)


def test_points():
    vec1 = np.arange(2, 6)
    vec2 = np.arange(-4, 5, 2)
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


def test_corners():
    vec1 = np.arange(2, 6)
    vec2 = np.arange(-4, 5, 2)
    vec3 = np.arange(-1, 1)
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


def test_meshgrid():
    vec1 = (0, 1)
    vec2 = (-1, 0, 1)
    vec3 = (2, 3, 4, 5)

    # Sparse meshgrid
    mgx = np.array(vec1)[:, np.newaxis, np.newaxis]
    mgy = np.array(vec2)[np.newaxis, :, np.newaxis]
    mgz = np.array(vec3)[np.newaxis, np.newaxis, :]

    grid = TensorGrid(vec1, vec2, vec3)
    xx, yy, zz = grid.meshgrid()
    assert all_equal(mgx, xx)
    assert all_equal(mgy, yy)
    assert all_equal(mgz, zz)

    xx, yy, zz = grid.meshgrid(sparse=True)
    assert all_equal(mgx, xx)
    assert all_equal(mgy, yy)
    assert all_equal(mgz, zz)

    # Dense meshgrid
    mgx = np.empty((2, 3, 4))
    for i in range(2):
        mgx[i, :, :] = vec1[i]

    mgy = np.empty((2, 3, 4))
    for i in range(3):
        mgy[:, i, :] = vec2[i]

    mgz = np.empty((2, 3, 4))
    for i in range(4):
        mgz[:, :, i] = vec3[i]

    xx, yy, zz = grid.meshgrid(sparse=False)
    assert all_equal(mgx, xx)
    assert all_equal(mgy, yy)
    assert all_equal(mgz, zz)

    assert all_equal(xx.shape, (2, 3, 4))


def test_tensor_getitem():
    vec1 = (0, 1)
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
        grid[1, 0, 1]

    with pytest.raises(IndexError):
        grid[1, 0, 1, 0, 0]

    # Slices return new TensorGrid's
    assert grid == grid[...]

    sub_grid = TensorGrid(vec1_sub, vec2_sub, vec3_sub, vec4_sub)
    assert grid[1, 0, 1:3, 0] == sub_grid
    assert grid[-1, :1, 1:3, :1] == sub_grid
    assert grid[1, 0, ..., 1:3, 0] == sub_grid

    sub_grid = TensorGrid(vec1_sub, vec2, vec3, vec4)
    assert grid[1, :, :, :] == sub_grid
    assert grid[1, ...] == sub_grid

    sub_grid = TensorGrid(vec1, vec2, vec3, vec4_sub)
    assert grid[:, :, :, 0] == sub_grid
    assert grid[..., 0] == sub_grid

    sub_grid = TensorGrid(vec1_sub, vec2, vec3, vec4_sub)
    assert grid[1, :, :, 0] == sub_grid
    assert grid[1, ..., 0] == sub_grid
    assert grid[1, :, :, ..., 0] == sub_grid

    with pytest.raises(IndexError):
        grid[1, ..., ..., 0]

    with pytest.raises(IndexError):
        grid[1, :, 1]

    with pytest.raises(IndexError):
        grid[1, 0, 1:2, 0, :]

    with pytest.raises(IndexError):
        grid[1, 0, np.newaxis, 1, 0]

    with pytest.raises(IndexError):
        grid[1, 0, None, 1, 0]

    # One-dimensional grid
    grid = TensorGrid(vec3)
    assert grid == grid[...]

    sub_grid = TensorGrid(vec3_sub)
    assert grid[1:3] == sub_grid


def test_cell_sizes():
    vec1 = np.array([1, 3])
    vec2 = np.array([-1, 0, 1])
    vec3 = np.array([2, 4, 5, 10])
    scalar = 0.5

    cs1, cs2, cs3 = [np.diff(vec) for vec in (vec1, vec2, vec3)]
    csscal = 0

    # Grid as set
    grid = TensorGrid(vec1, vec2, vec3, as_midp=False)
    assert all_equal(grid.cell_sizes(), (cs1, cs2, cs3))

    grid = TensorGrid(vec1, scalar, vec3, as_midp=False)
    assert all_equal(grid.cell_sizes(), (cs1, csscal, cs3))

    # Grid as tesselation
    cs1 = (2, 2)
    cs2 = (1, 1, 1)
    cs3 = (2, 1.5, 3, 5)

    grid = TensorGrid(vec1, vec2, vec3, as_midp=True)
    assert all_equal(grid.cell_sizes(), (cs1, cs2, cs3))

    grid = TensorGrid(vec1, scalar, vec3, as_midp=True)
    assert all_equal(grid.cell_sizes(), (cs1, csscal, cs3))


def test_convex_hull():
    vec1 = (1, 3)
    vec2 = (-1, 0, 1)
    vec3 = (2, 4, 5, 10)
    scalar = 0.5

    # Grid as set
    grid = TensorGrid(vec1, vec2, vec3, as_midp=False)
    begin = (vec1[0], vec2[0], vec3[0])
    end = (vec1[-1], vec2[-1], vec3[-1])
    chull = odl.IntervalProd(begin, end)
    assert grid.convex_hull() == chull

    # With degenerate axis
    grid = TensorGrid(vec1, vec2, scalar, as_midp=False)
    begin = (vec1[0], vec2[0], scalar)
    end = (vec1[-1], vec2[-1], scalar)
    chull = odl.IntervalProd(begin, end)
    assert grid.convex_hull() == chull

    # Grid as tesselation
    grid = TensorGrid(vec1, vec2, vec3, as_midp=True)
    cs1 = (2, 2)
    cs2 = (1, 1, 1)
    cs3 = (2, 1.5, 3, 5)
    begin = (vec1[0] - cs1[0] / 2.,
             vec2[0] - cs2[0] / 2.,
             vec3[0] - cs3[0] / 2.)
    end = (vec1[-1] + cs1[-1] / 2.,
           vec2[-1] + cs2[-1] / 2.,
           vec3[-1] + cs3[-1] / 2.)
    chull = odl.IntervalProd(begin, end)
    assert grid.convex_hull() == chull

    # With degenerate axis
    grid = TensorGrid(vec1, vec2, scalar, as_midp=True)
    begin = (vec1[0] - cs1[0] / 2., vec2[0] - cs2[0] / 2., scalar)
    end = (vec1[-1] + cs1[-1] / 2., vec2[-1] + cs2[-1] / 2., scalar)
    chull = odl.IntervalProd(begin, end)
    assert grid.convex_hull() == chull


def test_regular_grid_init():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    # Check correct initialization of coord_vectors
    grid = RegularGrid(minpt, maxpt, shape)
    vec1 = (0.75, 1.25)
    vec2 = (0,)
    vec3 = (-5, -2, 1)
    assert all_equal(grid.coord_vectors, (vec1, vec2, vec3))

    # Check different error scenarios
    nonpos_shape1 = (2, 0, 3)
    nonpos_shape2 = (-2, 1, 3)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt, nonpos_shape1)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt, nonpos_shape2)

    minpt_with_nan = (0.75, 0, np.nan)
    minpt_with_inf = (0.75, 0, np.inf)
    maxpt_with_nan = (1.25, np.nan, 1)
    maxpt_with_inf = (1.25, np.inf, 1)
    shape_with_nan = (2, np.nan, 3)
    shape_with_inf = (2, np.inf, 3)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt_with_nan, maxpt, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt_with_inf, maxpt, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt_with_nan, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt_with_inf, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt, shape_with_nan)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt, shape_with_inf)

    maxpt_smaller_minpt1 = (0.7, 0, 1)
    maxpt_smaller_minpt2 = (1.25, -1, 1)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt_smaller_minpt1, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt_smaller_minpt2, shape)

    too_short_minpt = (0.75, 0)
    too_long_minpt = (0.75, 0, -5, 2)
    too_short_maxpt = (0, 1)
    too_long_maxpt = (1.25, 0, 1, 25)
    too_short_shape = (2, 3)
    too_long_shape = (2, 1, 4, 3)

    with pytest.raises(ValueError):
        grid = RegularGrid(too_short_minpt, maxpt, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(too_long_minpt, maxpt, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, too_short_maxpt, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, too_long_maxpt, shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt, too_short_shape)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt, too_long_shape)

    maxpt_eq_minpt_at_shape_larger_than_1 = (0.75, 0, 1)

    with pytest.raises(ValueError):
        grid = RegularGrid(minpt, maxpt_eq_minpt_at_shape_larger_than_1,
                           shape)

    # Overrides for exact min and max
    exact_min = np.array((0, 1, 0), dtype=float)
    exact_max = np.array((1, 1, 3), dtype=float)
    shape = np.array((3, 1, 7), dtype=int)
    shift = (exact_max - exact_min) / (2 * shape)

    minpt = exact_min + shift
    maxpt = exact_max - shift

    grid = RegularGrid(minpt, maxpt, shape, as_midp=True,
                       _exact_min=exact_min, _exact_max=exact_max)
    assert all_equal(grid.min(), exact_min)
    assert all_equal(grid.max(), exact_max)


def test_center():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    center = (1, 0, -2)

    grid = RegularGrid(minpt, maxpt, shape)
    assert all_equal(grid.center, center)


def test_stride():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    stride = (0.5, 1, 3)

    grid = RegularGrid(minpt, maxpt, shape)
    assert all_equal(grid.stride, stride)


def test_cell_volume():
    minpt = (0.75, 0, -5)
    maxpt = (1.25, 0, 1)
    shape = (2, 1, 3)

    grid = RegularGrid(minpt, maxpt, shape)
    assert grid.cell_volume == 1.5


def test_regular_is_subgrid():
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
    assert grid.is_subgrid(fuzzy_sup_grid, tol=0.015)
    assert not grid.is_subgrid(fuzzy_sup_grid, tol=0.005)

    fuzzy_sup_grid = RegularGrid(minpt_fuzzy_sup2, maxpt_fuzzy_sup2,
                                 shape_sup)
    assert grid.is_subgrid(fuzzy_sup_grid, tol=0.015)
    assert not grid.is_subgrid(fuzzy_sup_grid, tol=0.005)


def test_regular_getitem():
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
        grid[1, 0, 1]

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

    with pytest.raises(IndexError):
        grid[1, ..., ..., 0]

    with pytest.raises(IndexError):
        grid[1, :, 1]

    with pytest.raises(IndexError):
        grid[1, 0, 1:2, 0, :]

    with pytest.raises(IndexError):
        grid[1, 0, np.newaxis, 1, 0]

    with pytest.raises(IndexError):
        grid[1, 0, None, 1, 0]

    # One-dimensional grid
    grid = RegularGrid(1, 5, 5)
    assert grid == grid[...]

    sub_grid = RegularGrid(1, 5, 3)
    assert grid[::2], sub_grid


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
