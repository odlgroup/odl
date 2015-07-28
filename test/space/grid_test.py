# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import
from future import standard_library

# External module imports
import unittest
import numpy as np

# RL imports
from RL.space.grid import TensorGrid
from RL.space.set import IntervalProd
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()


class TensorGridTestInit(RLTestCase):
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
            set_ = TensorGrid()

        with self.assertRaises(ValueError):
            set_ = TensorGrid(sorted1, unsorted, sorted2)

        with self.assertRaises(ValueError):
            set_ = TensorGrid(sorted1, with_dups, sorted2)

        with self.assertRaises(ValueError):
            set_ = TensorGrid(sorted1, unsorted_with_dups, sorted2)

        with self.assertRaises(ValueError):
            set_ = TensorGrid(sorted1, with_nan, sorted2)

        with self.assertRaises(ValueError):
            set_ = TensorGrid(sorted1, with_inf, sorted2)

        with self.assertRaises(ValueError):
            set_ = TensorGrid(sorted1, empty, sorted2)


class TensorGridTestAttributes(RLTestCase):
    def test_dim(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)
        vec3 = np.linspace(-2, 2, 50)

        grid1 = TensorGrid(vec1)
        grid2 = TensorGrid(vec1, vec2, vec3)

        self.assertEquals(grid1.dim, 1)
        self.assertEquals(grid2.dim, 3)

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

    def test_min_max(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)
        vec3 = np.arange(-1, 1)
        scalar = 0.5

        grid = TensorGrid(vec1, vec2, vec3)
        self.assertAllAlmostEquals(grid.min, (2, -4, -1), delta=0)
        self.assertAllAlmostEquals(grid.max, (5, 4, 0), delta=0)

        grid = TensorGrid(vec1, scalar, vec2, scalar)
        self.assertAllAlmostEquals(grid.min, (2, 0.5, -4, 0.5), delta=0)
        self.assertAllAlmostEquals(grid.max, (5, 0.5, 4, 0.5), delta=0)


class TensorGridTestMethods(RLTestCase):
    def test_equals(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)

        grid1 = TensorGrid(vec1)
        grid2 = TensorGrid(vec1, vec2)
        grid2_again = TensorGrid(vec1, vec2)
        grid2_rev = TensorGrid(vec2, vec1)

        self.assertTrue(grid1.equals(grid1))
        self.assertTrue(grid2.equals(grid2))
        self.assertTrue(grid2.equals(grid2_again))
        self.assertFalse(grid1.equals(grid2))
        self.assertFalse(grid2.equals(grid2_rev))
        self.assertFalse(grid2.equals((vec1, vec2)))

        # Test operators
        self.assertTrue(grid1 == grid1)
        self.assertFalse(grid1 != grid1)
        self.assertTrue(grid2 == grid2)
        self.assertFalse(grid2 != grid2)
        self.assertTrue(grid2 == grid2_again)
        self.assertFalse(grid2_again != grid2)
        self.assertFalse(grid2 == grid2_rev)
        self.assertTrue(grid2_rev != grid2)
        self.assertFalse(grid2 == (vec1, vec2))
        self.assertTrue(grid2_rev != (vec1, vec2))

        # Fuzzy check
        grid1 = TensorGrid(vec1, vec2)
        grid2 = TensorGrid(vec1 + (0.1, 0.05, 0, -0.1),
                           vec2 + (0.1, 0.05, 0, -0.1, -0.1))
        self.assertTrue(grid1.equals(grid2, tol=0.15))
        self.assertTrue(grid2.equals(grid1, tol=0.15))

        grid2 = TensorGrid(vec1 + (0.11, 0.05, 0, -0.1),
                           vec2 + (0.1, 0.05, 0, -0.1, -0.1))
        self.assertFalse(grid1.equals(grid2, tol=0.1))
        grid2 = TensorGrid(vec1 + (0.1, 0.05, 0, -0.1),
                           vec2 + (0.1, 0.05, 0, -0.11, -0.1))
        self.assertFalse(grid1.equals(grid2, tol=0.1))

    def test_contains(self):
        vec1 = np.arange(2, 6)
        vec2 = np.arange(-4, 5, 2)

        grid = TensorGrid(vec1, vec2)

        point_list = []
        for x in vec1:
            for y in vec2:
                point_list.append((x, y))

        self.assertTrue(all(grid.contains(p) for p in point_list))
        self.assertTrue(all(p in grid for p in point_list))

        self.assertFalse(grid.contains((0, 0)))
        self.assertFalse((0, 0) in grid)
        self.assertTrue((0, 0) not in grid)
        self.assertTrue((2, 0, 0) not in grid)

        # Fuzzy check
        self.assertTrue(grid.contains((2.1, -2.1), tol=0.15))
        self.assertFalse(grid.contains((2.2, -2.1), tol=0.15))

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
        self.assertAllAlmostEquals(points, grid.points(), delta=0)
        self.assertAllAlmostEquals(points, grid.points(order='C'), delta=0)
        self.assertAllAlmostEquals(grid.min, grid.points()[0], delta=0)
        self.assertAllAlmostEquals(grid.max, grid.points()[-1], delta=0)

        # F ordering
        points = []
        for x2 in vec2:
            for x1 in vec1:
                points.append(np.array((x1, x2), dtype=float))

        grid = TensorGrid(vec1, vec2)
        self.assertAllAlmostEquals(points, grid.points(order='F'), delta=0)

        # Degenerate axis 1
        points = []
        for x1 in vec1:
            for x2 in vec2:
                points.append(np.array((scalar, x1, x2), dtype=float))

        grid = TensorGrid(scalar, vec1, vec2)
        self.assertAllAlmostEquals(points, grid.points(), delta=0)

        # Degenerate axis 2
        points = []
        for x1 in vec1:
            for x2 in vec2:
                points.append(np.array((x1, scalar, x2), dtype=float))

        grid = TensorGrid(vec1, scalar, vec2)
        self.assertAllAlmostEquals(points, grid.points(), delta=0)

        # Degenerate axis 3
        points = []
        for x1 in vec1:
            for x2 in vec2:
                points.append(np.array((x1, x2, scalar), dtype=float))

        grid = TensorGrid(vec1, vec2, scalar)
        self.assertAllAlmostEquals(points, grid.points(), delta=0)

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
        self.assertAllAlmostEquals(corners, grid.corners(), delta=0)
        self.assertAllAlmostEquals(corners, grid.corners(order='C'), delta=0)

        # Min and max should appear at the beginning and the end, resp.
        self.assertAllAlmostEquals(grid.min, grid.corners()[0], delta=0)
        self.assertAllAlmostEquals(grid.max, grid.corners()[-1], delta=0)

        # F ordering
        corners = []
        for x3 in minmax3:
            for x2 in minmax2:
                for x1 in minmax1:
                    corners.append(np.array((x1, x2, x3), dtype=float))

        self.assertAllAlmostEquals(corners, grid.corners(order='F'), delta=0)

        # Degenerate axis 1
        corners = []
        for x2 in minmax2:
            for x3 in minmax3:
                corners.append(np.array((scalar, x2, x3), dtype=float))

        grid = TensorGrid(scalar, vec2, vec3)
        self.assertAllAlmostEquals(corners, grid.corners(), delta=0)

        # Degenerate axis 2
        corners = []
        for x1 in minmax1:
            for x3 in minmax3:
                corners.append(np.array((x1, scalar, x3), dtype=float))

        grid = TensorGrid(vec1, scalar, vec3)
        self.assertAllAlmostEquals(corners, grid.corners(), delta=0)

        # Degenerate axis 3
        corners = []
        for x1 in minmax1:
            for x2 in minmax2:
                corners.append(np.array((x1, x2, scalar), dtype=float))

        grid = TensorGrid(vec1, vec2, scalar)
        self.assertAllAlmostEquals(corners, grid.corners(), delta=0)

        # All degenerate
        corners = [(scalar, scalar)]
        grid = TensorGrid(scalar, scalar)
        self.assertAllAlmostEquals(corners, grid.corners(), delta=0)

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
        self.assertAllAlmostEquals(mgx, xx, delta=0)
        self.assertAllAlmostEquals(mgy, yy, delta=0)
        self.assertAllAlmostEquals(mgz, zz, delta=0)

        xx, yy, zz = grid.meshgrid(sparse=True)
        self.assertAllAlmostEquals(mgx, xx, delta=0)
        self.assertAllAlmostEquals(mgy, yy, delta=0)
        self.assertAllAlmostEquals(mgz, zz, delta=0)

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
        self.assertAllAlmostEquals(mgx, xx, delta=0)
        self.assertAllAlmostEquals(mgy, yy, delta=0)
        self.assertAllAlmostEquals(mgz, zz, delta=0)

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
        self.assertAllAlmostEquals(grid[1, 0, 1, 0], (1.0, -1.0, 3.0, 1.0),
                                   delta=0)

        # Slices return new TensorGrid's
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

    def test_convex_hull(self):
        vec1 = (0, 1)
        vec2 = (-1, 0, 1)
        vec3 = (2, 3, 4, 5)
        scalar = 0.5

        grid = TensorGrid(vec1, vec2, vec3)
        begin = (vec1[0], vec2[0], vec3[0])
        end = (vec1[-1], vec2[-1], vec3[-1])
        chull = IntervalProd(begin, end)
        self.assertEquals(grid.convex_hull(), chull)

        # With degenerate axis
        grid = TensorGrid(vec1, vec2, scalar)
        begin = (vec1[0], vec2[0], scalar)
        end = (vec1[-1], vec2[-1], scalar)
        chull = IntervalProd(begin, end)
        self.assertEquals(grid.convex_hull(), chull)


if __name__ == '__main__':
    unittest.main(exit=False)
