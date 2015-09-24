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
import unittest
import numpy as np

# ODL imports
from odl.discr.grid import TensorGrid, RegularGrid
from odl.sets.domain import IntervalProd
from odl.util.testutils import ODLTestCase


class TensorGridTestInit(ODLTestCase):
    def test_init(self):
        sorted1 = np.arange(2, 6)
        sorted2 = np.arange(-4, 5, 2)
        sorted3 = np.linspace(-1, 2, 50)
        scalar = 0.5

        # Just test if the code runs
        grid = TensorGrid(sorted1)
        grid = TensorGrid(sorted1, sorted2)
        grid = TensorGrid(sorted1, sorted1)
        grid = TensorGrid(sorted1, sorted2, sorted3)
        grid = TensorGrid(sorted2, scalar, sorted1)

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

        with self.assertRaises(ValueError):
            grid = TensorGrid()

        with self.assertRaises(ValueError):
            grid = TensorGrid(sorted1, unsorted, sorted2)

        with self.assertRaises(ValueError):
            grid = TensorGrid(sorted1, with_dups, sorted2)

        with self.assertRaises(ValueError):
            grid = TensorGrid(sorted1, unsorted_with_dups, sorted2)

        with self.assertRaises(ValueError):
            grid = TensorGrid(sorted1, with_nan, sorted2)

        with self.assertRaises(ValueError):
            grid = TensorGrid(sorted1, with_inf, sorted2)

        with self.assertRaises(ValueError):
            grid = TensorGrid(sorted1, empty, sorted2)


class TensorGridTestAttributes(ODLTestCase):
    def test_ndim(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)
        vec3 = np.linspace(-2, 2, 50)

        grid1 = TensorGrid(vec1)
        grid2 = TensorGrid(vec1, vec2, vec3)

        self.assertEquals(grid1.ndim, 1)
        self.assertEquals(grid2.ndim, 3)

    def test_shape(self):
        vec1 = np.arange(2, 6)
        vec2 = np.linspace(-2, 2, 50)
        scalar = 0.5

        grid1 = TensorGrid(vec1)
        grid2 = TensorGrid(vec1, vec2)
        grid3 = TensorGrid(scalar, vec2)

        self.assertEquals(grid1.shape, (4,))
        self.assertEquals(grid2.shape, (4, 50))
        self.assertEquals(grid3.shape, (1, 50))

    def test_ntotal(self):
        vec1 = np.arange(2, 6)
        vec2 = np.linspace(-2, 2, 50)
        scalar = 0.5

        grid1 = TensorGrid(vec1)
        grid2 = TensorGrid(vec1, vec2)
        grid3 = TensorGrid(scalar, vec2)

        self.assertEquals(grid1.ntotal, 4)
        self.assertEquals(grid2.ntotal, 200)
        self.assertEquals(grid3.ntotal, 50)

    def test_minpt_maxpt(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)
        vec3 = np.arange(-1, 1)
        scalar = 0.5

        grid = TensorGrid(vec1, vec2, vec3)
        self.assertAllEquals(grid.min_pt, (2, -4, -1))
        self.assertAllEquals(grid.max_pt, (5, 4, 0))

        grid = TensorGrid(vec1, scalar, vec2, scalar)
        self.assertAllEquals(grid.min_pt, (2, 0.5, -4, 0.5))
        self.assertAllEquals(grid.max_pt, (5, 0.5, 4, 0.5))


class TensorGridTestMethods(ODLTestCase):
    def test_element(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)

        grid = TensorGrid(vec1, vec2)
        some_pt = grid.element()
        self.assertIn(some_pt, grid)

    def test_min_max(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)
        vec3 = np.arange(-1, 1)
        scalar = 0.5

        grid = TensorGrid(vec1, vec2, vec3, as_midp=False)
        self.assertAllEquals(grid.min(), (2, -4, -1))
        self.assertAllEquals(grid.max(), (5, 4, 0))

        grid = TensorGrid(vec1, scalar, vec2, scalar, as_midp=False)
        self.assertAllEquals(grid.min(), (2, 0.5, -4, 0.5))
        self.assertAllEquals(grid.max(), (5, 0.5, 4, 0.5))

        grid = TensorGrid(vec1, vec2, vec3, as_midp=True)
        self.assertAllEquals(grid.min(), (1.5, -5, -1.5))
        self.assertAllEquals(grid.max(), (5.5, 5, 0.5))

        grid = TensorGrid(vec1, scalar, vec2, scalar, as_midp=True)
        self.assertAllEquals(grid.min(), (1.5, 0.5, -5, 0.5))
        self.assertAllEquals(grid.max(), (5.5, 0.5, 5, 0.5))

    def test_equals(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)

        grid1 = TensorGrid(vec1)
        grid2 = TensorGrid(vec1, vec2)
        grid2_again = TensorGrid(vec1, vec2)
        grid2_rev = TensorGrid(vec2, vec1)

        self.assertTrue(grid1 == grid1)
        self.assertFalse(grid1 != grid1)
        self.assertTrue(grid2 == grid2)
        self.assertFalse(grid2 != grid2)
        self.assertTrue(grid2 == grid2_again)
        self.assertFalse(grid1 == grid2)
        self.assertFalse(grid2 == grid2_rev)
        self.assertFalse(grid2 == (vec1, vec2))

        # Fuzzy check
        grid1 = TensorGrid(vec1, vec2)
        grid2 = TensorGrid(vec1 + (0.1, 0.05, 0, -0.1),
                           vec2 + (0.1, 0.05, 0, -0.1, -0.1))
        self.assertTrue(grid1.approx_equals(grid2, tol=0.15))
        self.assertTrue(grid2.approx_equals(grid1, tol=0.15))

        grid2 = TensorGrid(vec1 + (0.11, 0.05, 0, -0.1),
                           vec2 + (0.1, 0.05, 0, -0.1, -0.1))
        self.assertFalse(grid1.approx_equals(grid2, tol=0.1))
        grid2 = TensorGrid(vec1 + (0.1, 0.05, 0, -0.1),
                           vec2 + (0.1, 0.05, 0, -0.11, -0.1))
        self.assertFalse(grid1.approx_equals(grid2, tol=0.1))

    def test_contains(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)

        grid = TensorGrid(vec1, vec2)

        point_list = []
        for x in vec1:
            for y in vec2:
                point_list.append((x, y))

        self.assertTrue(all(p in grid for p in point_list))

        self.assertFalse((0, 0) in grid)
        self.assertTrue((0, 0) not in grid)
        self.assertTrue((2, 0, 0) not in grid)

        # Fuzzy check
        self.assertTrue(grid.approx_contains((2.1, -2.1), tol=0.15))
        self.assertFalse(grid.approx_contains((2.2, -2.1), tol=0.15))

        # 1d points
        grid = TensorGrid(vec1)
        self.assertTrue(3 in grid)
        self.assertFalse(7 in grid)

    def test_is_subgrid(self):
        vec1 = np.arange(2, 6)
        vec1_sup = np.arange(2, 8)
        vec2 = np.arange(-4, 5, 2)
        vec2_sup = np.arange(-6, 7, 2)
        vec2_sub = np.arange(-4, 3, 2)
        scalar = 0.5

        grid = TensorGrid(vec1, vec2)
        self.assertTrue(grid.is_subgrid(grid))

        sup_grid = TensorGrid(vec1_sup, vec2_sup)
        self.assertTrue(grid.is_subgrid(sup_grid))
        self.assertFalse(sup_grid.is_subgrid(grid))

        not_sup_grid = TensorGrid(vec1_sup, vec2_sub)
        self.assertFalse(grid.is_subgrid(not_sup_grid))
        self.assertFalse(not_sup_grid.is_subgrid(grid))

        # Fuzzy check
        fuzzy_vec1_sup = vec1_sup + (0.1, 0.05, 0, -0.1, 0, 0.1)
        fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.1, 0.05)
        fuzzy_sup_grid = TensorGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
        self.assertTrue(grid.is_subgrid(fuzzy_sup_grid, tol=0.15))

        fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.11, 0.05)
        fuzzy_sup_grid = TensorGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
        self.assertFalse(grid.is_subgrid(fuzzy_sup_grid, tol=0.1))

        # Changes in the non-overlapping part don't matter
        fuzzy_vec2_sup = vec2_sup + (0.1, 0.05, 0, -0.1, 0, 0.05, 0.11)
        fuzzy_sup_grid = TensorGrid(fuzzy_vec1_sup, fuzzy_vec2_sup)
        self.assertTrue(grid.is_subgrid(fuzzy_sup_grid, tol=0.15))

        # With degenerate axis
        grid = TensorGrid(vec1, scalar, vec2)
        sup_grid = TensorGrid(vec1_sup, scalar, vec2_sup)
        self.assertTrue(grid.is_subgrid(sup_grid))

        fuzzy_sup_grid = TensorGrid(vec1, scalar+0.1, vec2)
        self.assertTrue(grid.is_subgrid(fuzzy_sup_grid, tol=0.15))

    def test_points(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)
        scalar = 0.5

        # C ordering
        points = []
        for x1 in vec1:
            for x2 in vec2:
                points.append(np.array((x1, x2), dtype=float))

        grid = TensorGrid(vec1, vec2)
        self.assertAllEquals(points, grid.points())
        self.assertAllEquals(points, grid.points(order='C'))
        self.assertAllEquals(grid.min_pt, grid.points()[0])
        self.assertAllEquals(grid.max_pt, grid.points()[-1])

        # F ordering
        points = []
        for x2 in vec2:
            for x1 in vec1:
                points.append(np.array((x1, x2), dtype=float))

        grid = TensorGrid(vec1, vec2)
        self.assertAllEquals(points, grid.points(order='F'))

        # Degenerate axis 1
        points = []
        for x1 in vec1:
            for x2 in vec2:
                points.append(np.array((scalar, x1, x2), dtype=float))

        grid = TensorGrid(scalar, vec1, vec2)
        self.assertAllEquals(points, grid.points())

        # Degenerate axis 2
        points = []
        for x1 in vec1:
            for x2 in vec2:
                points.append(np.array((x1, scalar, x2), dtype=float))

        grid = TensorGrid(vec1, scalar, vec2)
        self.assertAllEquals(points, grid.points())

        # Degenerate axis 3
        points = []
        for x1 in vec1:
            for x2 in vec2:
                points.append(np.array((x1, x2, scalar), dtype=float))

        grid = TensorGrid(vec1, vec2, scalar)
        self.assertAllEquals(points, grid.points())

    def test_corners(self):
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
        self.assertAllEquals(corners, grid.corners())
        self.assertAllEquals(corners, grid.corners(order='C'))

        # minpt and maxpt should appear at the beginning and the end, resp.
        self.assertAllEquals(grid.min_pt, grid.corners()[0])
        self.assertAllEquals(grid.max_pt, grid.corners()[-1])

        # F ordering
        corners = []
        for x3 in minmax3:
            for x2 in minmax2:
                for x1 in minmax1:
                    corners.append(np.array((x1, x2, x3), dtype=float))

        self.assertAllEquals(corners, grid.corners(order='F'))

        # Degenerate axis 1
        corners = []
        for x2 in minmax2:
            for x3 in minmax3:
                corners.append(np.array((scalar, x2, x3), dtype=float))

        grid = TensorGrid(scalar, vec2, vec3)
        self.assertAllEquals(corners, grid.corners())

        # Degenerate axis 2
        corners = []
        for x1 in minmax1:
            for x3 in minmax3:
                corners.append(np.array((x1, scalar, x3), dtype=float))

        grid = TensorGrid(vec1, scalar, vec3)
        self.assertAllEquals(corners, grid.corners())

        # Degenerate axis 3
        corners = []
        for x1 in minmax1:
            for x2 in minmax2:
                corners.append(np.array((x1, x2, scalar), dtype=float))

        grid = TensorGrid(vec1, vec2, scalar)
        self.assertAllEquals(corners, grid.corners())

        # All degenerate
        corners = [(scalar, scalar)]
        grid = TensorGrid(scalar, scalar)
        self.assertAllEquals(corners, grid.corners())

    def test_meshgrid(self):
        vec1 = (0, 1)
        vec2 = (-1, 0, 1)
        vec3 = (2, 3, 4, 5)

        # Sparse meshgrid
        mgx = np.array(vec1)[:, np.newaxis, np.newaxis]
        mgy = np.array(vec2)[np.newaxis, :, np.newaxis]
        mgz = np.array(vec3)[np.newaxis, np.newaxis, :]

        grid = TensorGrid(vec1, vec2, vec3)
        xx, yy, zz = grid.meshgrid()
        self.assertAllEquals(mgx, xx)
        self.assertAllEquals(mgy, yy)
        self.assertAllEquals(mgz, zz)

        xx, yy, zz = grid.meshgrid(sparse=True)
        self.assertAllEquals(mgx, xx)
        self.assertAllEquals(mgy, yy)
        self.assertAllEquals(mgz, zz)

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
        self.assertAllEquals(mgx, xx)
        self.assertAllEquals(mgy, yy)
        self.assertAllEquals(mgz, zz)

        self.assertTupleEqual(xx.shape, (2, 3, 4))

    def test_getitem(self):
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
        self.assertAllEquals(grid[1, 0, 1, 0], (1.0, -1.0, 3.0, 1.0))

        with self.assertRaises(IndexError):
            grid[1, 0, 1]

        with self.assertRaises(IndexError):
            grid[1, 0, 1, 0, 0]

        # Slices return new TensorGrid's
        self.assertEquals(grid, grid[...])

        sub_grid = TensorGrid(vec1_sub, vec2_sub, vec3_sub, vec4_sub)
        self.assertEquals(grid[1, 0, 1:3, 0], sub_grid)
        self.assertEquals(grid[-1, :1, 1:3, :1], sub_grid)
        self.assertEquals(grid[1, 0, ..., 1:3, 0], sub_grid)

        sub_grid = TensorGrid(vec1_sub, vec2, vec3, vec4)
        self.assertEquals(grid[1, :, :, :], sub_grid)
        self.assertEquals(grid[1, ...], sub_grid)

        sub_grid = TensorGrid(vec1, vec2, vec3, vec4_sub)
        self.assertEquals(grid[:, :, :, 0], sub_grid)
        self.assertEquals(grid[..., 0], sub_grid)

        sub_grid = TensorGrid(vec1_sub, vec2, vec3, vec4_sub)
        self.assertEquals(grid[1, :, :, 0], sub_grid)
        self.assertEquals(grid[1, ..., 0], sub_grid)
        self.assertEquals(grid[1, :, :, ..., 0], sub_grid)

        with self.assertRaises(IndexError):
            grid[1, ..., ..., 0]

        with self.assertRaises(IndexError):
            grid[1, :, 1]

        with self.assertRaises(IndexError):
            grid[1, 0, 1:2, 0, :]

        with self.assertRaises(IndexError):
            grid[1, 0, np.newaxis, 1, 0]

        with self.assertRaises(IndexError):
            grid[1, 0, None, 1, 0]

        # One-dimensional grid
        grid = TensorGrid(vec3)
        self.assertEquals(grid, grid[...])

        sub_grid = TensorGrid(vec3_sub)
        self.assertEquals(grid[1:3], sub_grid)

    def test_cell_sizes(self):
        vec1 = np.array([1, 3])
        vec2 = np.array([-1, 0, 1])
        vec3 = np.array([2, 4, 5, 10])
        scalar = 0.5

        cs1, cs2, cs3 = [np.diff(vec) for vec in (vec1, vec2, vec3)]
        csscal = 0

        # Grid as set
        grid = TensorGrid(vec1, vec2, vec3, as_midp=False)
        self.assertAllEquals(grid.cell_sizes(), (cs1, cs2, cs3))

        grid = TensorGrid(vec1, scalar, vec3, as_midp=False)
        self.assertAllEquals(grid.cell_sizes(), (cs1, csscal, cs3))

        # Grid as tesselation
        cs1 = (2, 2)
        cs2 = (1, 1, 1)
        cs3 = (2, 1.5, 3, 5)

        grid = TensorGrid(vec1, vec2, vec3, as_midp=True)
        self.assertAllEquals(grid.cell_sizes(), (cs1, cs2, cs3))

        grid = TensorGrid(vec1, scalar, vec3, as_midp=True)
        self.assertAllEquals(grid.cell_sizes(), (cs1, csscal, cs3))

    def test_convex_hull(self):
        vec1 = (1, 3)
        vec2 = (-1, 0, 1)
        vec3 = (2, 4, 5, 10)
        scalar = 0.5

        # Grid as set
        grid = TensorGrid(vec1, vec2, vec3, as_midp=False)
        begin = (vec1[0], vec2[0], vec3[0])
        end = (vec1[-1], vec2[-1], vec3[-1])
        chull = IntervalProd(begin, end)
        self.assertEquals(grid.convex_hull(), chull)

        # With degenerate axis
        grid = TensorGrid(vec1, vec2, scalar, as_midp=False)
        begin = (vec1[0], vec2[0], scalar)
        end = (vec1[-1], vec2[-1], scalar)
        chull = IntervalProd(begin, end)
        self.assertEquals(grid.convex_hull(), chull)

        # Grid as tesselation
        grid = TensorGrid(vec1, vec2, vec3, as_midp=True)
        cs1 = (2, 2)
        cs2 = (1, 1, 1)
        cs3 = (2, 1.5, 3, 5)
        begin = (vec1[0] - cs1[0]/2., vec2[0] - cs2[0]/2., vec3[0] - cs3[0]/2.)
        end = (vec1[-1] + cs1[-1]/2., vec2[-1] + cs2[-1]/2.,
               vec3[-1] + cs3[-1]/2.)
        chull = IntervalProd(begin, end)
        self.assertEquals(grid.convex_hull(), chull)

        # With degenerate axis
        grid = TensorGrid(vec1, vec2, scalar, as_midp=True)
        begin = (vec1[0] - cs1[0]/2., vec2[0] - cs2[0]/2., scalar)
        end = (vec1[-1] + cs1[-1]/2., vec2[-1] + cs2[-1]/2., scalar)
        chull = IntervalProd(begin, end)
        self.assertEquals(grid.convex_hull(), chull)

    def test_repr(self):
        vec1 = (0, 1)
        long_vec = np.arange(10)
        scalar = 0.5

        # Grid as set
        grid = TensorGrid(vec1, scalar)
        repr_string = 'TensorGrid([0.0, 1.0], [0.5])'
        self.assertEquals(repr(grid), repr_string)

        grid = TensorGrid(scalar, long_vec)
        repr_string = 'TensorGrid([0.5], [0.0, 1.0, 2.0, ..., 7.0, 8.0, 9.0])'
        self.assertEquals(repr(grid), repr_string)

        # Grid as tesselation
        grid = TensorGrid(vec1, scalar, as_midp=True)
        repr_string = 'TensorGrid([0.0, 1.0], [0.5], as_midp=True)'
        self.assertEquals(repr(grid), repr_string)

        grid = TensorGrid(scalar, long_vec, as_midp=True)
        repr_string = ('TensorGrid([0.5], [0.0, 1.0, 2.0, ..., 7.0, 8.0, 9.0],'
                       ' as_midp=True)')
        self.assertEquals(repr(grid), repr_string)

    def test_str(self):
        vec1 = (0, 1)
        vec2 = (-1, 0, 2)
        long_vec = np.arange(10)
        scalar = 0.5

        # Grid as set
        grid = TensorGrid(vec1, scalar, vec2)
        grid_string = 'grid [0.0, 1.0] x [0.5] x [-1.0, 0.0, 2.0]'
        self.assertEquals(str(grid), grid_string)

        grid = TensorGrid(scalar, long_vec)
        grid_string = 'grid [0.5] x [0.0, 1.0, 2.0, ..., 7.0, 8.0, 9.0]'
        self.assertEquals(str(grid), grid_string)

        # Grid as tesselation
        grid = TensorGrid(vec1, scalar, vec2, as_midp=True)
        grid_string = 'midp grid [0.0, 1.0] x [0.5] x [-1.0, 0.0, 2.0]'
        self.assertEquals(str(grid), grid_string)

        grid = TensorGrid(scalar, long_vec, as_midp=True)
        grid_string = ('midp grid [0.5] x [0.0, 1.0, 2.0, ..., '
                       '7.0, 8.0, 9.0]')
        self.assertEquals(str(grid), grid_string)


class RegularGridTestInit(ODLTestCase):
    def test_init(self):
        minpt = (0.75, 0, -5)
        maxpt = (1.25, 0, 1)
        shape = (2, 1, 3)

        # Check correct initialization of coord_vectors
        grid = RegularGrid(minpt, maxpt, shape)
        vec1 = (0.75, 1.25)
        vec2 = (0,)
        vec3 = (-5, -2, 1)
        self.assertAllEquals(grid.coord_vectors, (vec1, vec2, vec3))

        # Check different error scenarios
        nonpos_shape1 = (2, 0, 3)
        nonpos_shape2 = (-2, 1, 3)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt, nonpos_shape1)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt, nonpos_shape2)

        minpt_with_nan = (0.75, 0, np.nan)
        minpt_with_inf = (0.75, 0, np.inf)
        maxpt_with_nan = (1.25, np.nan, 1)
        maxpt_with_inf = (1.25, np.inf, 1)
        shape_with_nan = (2, np.nan, 3)
        shape_with_inf = (2, np.inf, 3)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt_with_nan, maxpt, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt_with_inf, maxpt, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt_with_nan, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt_with_inf, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt, shape_with_nan)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt, shape_with_inf)

        maxpt_smaller_minpt1 = (0.7, 0, 1)
        maxpt_smaller_minpt2 = (1.25, -1, 1)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt_smaller_minpt1, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt_smaller_minpt2, shape)

        too_short_minpt = (0.75, 0)
        too_long_minpt = (0.75, 0, -5, 2)
        too_short_maxpt = (0, 1)
        too_long_maxpt = (1.25, 0, 1, 25)
        too_short_shape = (2, 3)
        too_long_shape = (2, 1, 4, 3)

        with self.assertRaises(ValueError):
            grid = RegularGrid(too_short_minpt, maxpt, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(too_long_minpt, maxpt, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, too_short_maxpt, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, too_long_maxpt, shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt, too_short_shape)

        with self.assertRaises(ValueError):
            grid = RegularGrid(minpt, maxpt, too_long_shape)

        maxpt_eq_minpt_at_shape_larger_than_1 = (0.75, 0, 1)

        with self.assertRaises(ValueError):
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
        self.assertAllEquals(grid.min(), exact_min)
        self.assertAllEquals(grid.max(), exact_max)


class RegularGridTestAttributes(ODLTestCase):
    def test_center(self):
        minpt = (0.75, 0, -5)
        maxpt = (1.25, 0, 1)
        shape = (2, 1, 3)

        center = (1, 0, -2)

        grid = RegularGrid(minpt, maxpt, shape)
        self.assertAllEquals(grid.center, center)

    def test_stride(self):
        minpt = (0.75, 0, -5)
        maxpt = (1.25, 0, 1)
        shape = (2, 1, 3)

        stride = (0.5, 1, 3)

        grid = RegularGrid(minpt, maxpt, shape)
        self.assertAllEquals(grid.stride, stride)


class RegularGridTestMethods(ODLTestCase):
    def test_is_subgrid(self):
        minpt = (0.75, 0, -5)
        maxpt = (1.25, 0, 1)
        shape = (2, 1, 5)

        # Optimized cases
        grid = RegularGrid(minpt, maxpt, shape)
        self.assertTrue(grid.is_subgrid(grid))

        smaller_shape = (1, 1, 5)
        not_sup_grid = RegularGrid(minpt, maxpt, smaller_shape)
        self.assertFalse(grid.is_subgrid(not_sup_grid))

        larger_minpt = (0.85, 0, -4)
        not_sup_grid = RegularGrid(larger_minpt, maxpt, shape)
        self.assertFalse(grid.is_subgrid(not_sup_grid))

        smaller_maxpt = (1.15, 0, 0)
        not_sup_grid = RegularGrid(minpt, smaller_maxpt, shape)
        self.assertFalse(grid.is_subgrid(not_sup_grid))

        # Real checks
        minpt_sup1 = (-0.25, -2, -5)
        maxpt_sup1 = (1.25, 2, 1)
        shape_sup1 = (4, 3, 9)
        sup_grid = RegularGrid(minpt_sup1, maxpt_sup1, shape_sup1)
        self.assertTrue(grid.is_subgrid(sup_grid))
        self.assertFalse(sup_grid.is_subgrid(grid))

        minpt_sup2 = (0.5, 0, -5)
        maxpt_sup2 = (1.5, 0, 1)
        shape_sup2 = (5, 1, 9)
        sup_grid = RegularGrid(minpt_sup2, maxpt_sup2, shape_sup2)
        self.assertTrue(grid.is_subgrid(sup_grid))
        self.assertFalse(sup_grid.is_subgrid(grid))

        shape_not_sup1 = (4, 3, 10)
        not_sup_grid = RegularGrid(minpt_sup1, maxpt_sup1, shape_not_sup1)
        self.assertFalse(grid.is_subgrid(not_sup_grid))
        self.assertFalse(not_sup_grid.is_subgrid(grid))

        minpt_not_sup1 = (-0.25, -2.5, -5)
        not_sup_grid = RegularGrid(minpt_not_sup1, maxpt_sup1, shape_sup1)
        self.assertFalse(grid.is_subgrid(not_sup_grid))
        self.assertFalse(not_sup_grid.is_subgrid(grid))

        maxpt_not_sup1 = (1.35, 2.0001, 1)
        not_sup_grid = RegularGrid(minpt_sup1, maxpt_not_sup1, shape_sup1)
        self.assertFalse(grid.is_subgrid(not_sup_grid))
        self.assertFalse(not_sup_grid.is_subgrid(grid))

        # Should also work for TensorGrid's
        vec1_sup = (0.75, 1, 1.25, 7)
        vec2_sup = (0,)
        vec3_sup = (-5, -3.5, -3, -2, -0.5, 0, 1, 9.5)

        tensor_sup_grid = TensorGrid(vec1_sup, vec2_sup, vec3_sup)
        self.assertTrue(grid.is_subgrid(tensor_sup_grid))

        vec1_not_sup = (1, 1.25, 7)
        vec2_not_sup = (0,)
        vec3_not_sup = (-4, -2, 1)

        tensor_not_sup_grid = TensorGrid(vec1_not_sup, vec2_not_sup,
                                         vec3_not_sup)
        self.assertFalse(grid.is_subgrid(tensor_not_sup_grid))

        # Fuzzy check
        shape_sup = (4, 3, 9)

        minpt_fuzzy_sup1 = (-0.24, -2, -5.01)
        minpt_fuzzy_sup2 = (-0.24, -2, -5)
        maxpt_fuzzy_sup1 = (1.24, 2, 1)
        maxpt_fuzzy_sup2 = (1.25, 2, 1.01)

        fuzzy_sup_grid = RegularGrid(minpt_fuzzy_sup1, maxpt_fuzzy_sup1,
                                     shape_sup)
        self.assertTrue(grid.is_subgrid(fuzzy_sup_grid, tol=0.015))
        self.assertFalse(grid.is_subgrid(fuzzy_sup_grid, tol=0.005))

        fuzzy_sup_grid = RegularGrid(minpt_fuzzy_sup2, maxpt_fuzzy_sup2,
                                     shape_sup)
        self.assertTrue(grid.is_subgrid(fuzzy_sup_grid, tol=0.015))
        self.assertFalse(grid.is_subgrid(fuzzy_sup_grid, tol=0.005))

        # TODO: make modules for automated randomized tests
        # Some more randomized tests
#        for _ in range(50):
#            tol = 0.01
#            center_fuzzy_sup = center + tol * np.random.uniform(-1, 1, size=3)
#            shape_fuzzy_sup = (6, 2, 5)
#            # Approximately same stride
#            stride1_fuzzy_sup = stride + tol * np.random.uniform(-1, 1, size=3)
#            # Approximately 1/3 stride
#            stride2_fuzzy_sup = (stride +
#                                 3*tol * np.random.uniform(-1, 1, size=3)) / 3
#
#            fuzzy_sup_grid1 = RegularGrid(shape_fuzzy_sup, center_fuzzy_sup,
#                                          stride1_fuzzy_sup)
#            fuzzy_sup_grid2 = RegularGrid(shape_fuzzy_sup, center_fuzzy_sup,
#                                          stride2_fuzzy_sup)
#            fuzzy_sup_tensor_grid1 = TensorGrid(*fuzzy_sup_grid1.coord_vectors)
#            fuzzy_sup_tensor_grid2 = TensorGrid(*fuzzy_sup_grid2.coord_vectors)
#
#            # Test against element-by-element comparison for various levels
#            # of tolerance (includes ridiculously large tolerance)
#            for fac in range(1, 51, 2):
#                self.assertEquals(
#                    grid.is_subgrid(fuzzy_sup_grid1, tol=fac*tol),
#                    grid.is_subgrid(fuzzy_sup_tensor_grid1, tol=fac*tol))
#                self.assertEquals(
#                    grid.is_subgrid(fuzzy_sup_grid2, tol=fac*tol),
#                    grid.is_subgrid(fuzzy_sup_tensor_grid2, tol=fac*tol))

    def test_getitem(self):
        minpt = (0.75, 0, -5, 4)
        maxpt = (1.25, 0, 1, 13)
        shape = (2, 1, 5, 4)

        grid = RegularGrid(minpt, maxpt, shape)

        # Single indices yield points as an array
        indices = [1, 0, 1, 1]
        values = [vec[i] for i, vec in zip(indices, grid.coord_vectors)]
        self.assertAllEquals(grid[1, 0, 1, 1], values)

        indices = [0, 0, 4, 3]
        values = [vec[i] for i, vec in zip(indices, grid.coord_vectors)]
        self.assertAllEquals(grid[0, 0, 4, 3], values)

        with self.assertRaises(IndexError):
            grid[1, 0, 1]

        with self.assertRaises(IndexError):
            grid[1, 0, 1, 2, 0]

        with self.assertRaises(IndexError):
            grid[1, 1, 6, 2]

        with self.assertRaises(IndexError):
            grid[1, 0, 4, 6]

        # Slices return new RegularGrid's
        self.assertEquals(grid, grid[...])

        # Use TensorGrid implementation as reference here
        tensor_grid = TensorGrid(*grid.coord_vectors)

        test_slice = np.s_[1, :, ::2, ::3]
        self.assertAllEquals(grid[test_slice].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[1:2, :, ::2, ::3].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[1:2, :, ::2, ..., ::3].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)

        test_slice = np.s_[0:1, :, :, 2:4]
        self.assertAllEquals(grid[test_slice].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[:1, :, :, 2:].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[:-1, ..., 2:].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)

        test_slice = np.s_[:, 0, :, :]
        self.assertAllEquals(grid[test_slice].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[:, 0, ...].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[0:2, :, ...].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[...].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)

        test_slice = np.s_[:, :, 0::2, :]
        self.assertAllEquals(grid[test_slice].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[..., 0::2, :].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)

        test_slice = np.s_[..., 1, :]
        self.assertAllEquals(grid[test_slice].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)
        self.assertAllEquals(grid[:, :, 1, :].coord_vectors,
                             tensor_grid[test_slice].coord_vectors)

        with self.assertRaises(IndexError):
            grid[1:1, :, 0, 0]

        with self.assertRaises(IndexError):
            grid[1, ..., ..., 0]

        with self.assertRaises(IndexError):
            grid[1, :, 1]

        with self.assertRaises(IndexError):
            grid[1, 0, 1:2, 0, :]

        with self.assertRaises(IndexError):
            grid[1, 0, np.newaxis, 1, 0]

        with self.assertRaises(IndexError):
            grid[1, 0, None, 1, 0]

        # One-dimensional grid
        grid = RegularGrid(1, 5, 5)
        self.assertEquals(grid, grid[...])

        sub_grid = RegularGrid(1, 5, 3)
        self.assertEquals(grid[::2], sub_grid)

    def test_repr(self):
        minpt = (0.75, 0)
        maxpt = (1.25, 0)
        shape = (2, 1)

        grid = RegularGrid(minpt, maxpt, shape)
        repr_string = 'RegularGrid([0.75, 0.0], [1.25, 0.0], [2, 1])'
        self.assertEquals(repr(grid), repr_string)

        grid = RegularGrid(minpt, maxpt, shape, as_midp=True)
        repr_string = ('RegularGrid([0.75, 0.0], [1.25, 0.0], [2, 1], '
                       'as_midp=True)')
        self.assertEquals(repr(grid), repr_string)

    def test_str(self):
        minpt = (0, 0)
        maxpt = (1, 0)
        shape = (5, 1)

        grid = RegularGrid(minpt, maxpt, shape)
        str_string = 'regular grid [0.0, 0.25, ..., 1.0] x [0.0]'
        self.assertEquals(str(grid), str_string)

        grid = RegularGrid(minpt, maxpt, shape, as_midp=True)
        str_string = 'midp regular grid [0.0, 0.25, ..., 1.0] x [0.0]'
        self.assertEquals(str(grid), str_string)


if __name__ == '__main__':
    unittest.main(exit=False)
