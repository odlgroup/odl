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
import math
from numpy import float64

# RL imports
from RL.operator.operator import *
from RL.space.space import *
from RL.space.euclidean import RN
from RL.utility.testutils import RLTestCase, skip_all_tests, Timer

try:
    from RL.space.cuda import *
except ImportError:
    RLTestCase = skip_all_tests("Missing RLcpp")

import numpy as np

standard_library.install_aliases()

class TestInit(RLTestCase):
    def test_empty(self):
        r3 = CudaRN(3)
        x = r3.element()
        # Nothing to test, simply check that code runs

    def test_zero(self):
        r3 = CudaRN(3)
        self.assertAllAlmostEquals(r3.zero(), [0, 0, 0])

    def test_list_init(self):
        r3 = CudaRN(3)
        x = r3.element([1, 2, 3])
        self.assertAllAlmostEquals(x, [1, 2, 3])

    def test_ndarray_init(self):
        r3 = CudaRN(3)

        x0 = np.array([1., 2., 3.])
        x = r3.element(x0)
        self.assertAllAlmostEquals(x, x0)

        x0 = np.array([1, 2, 3], dtype=float64)
        x = r3.element(x0)
        self.assertAllAlmostEquals(x, x0)

        x0 = np.array([1, 2, 3], dtype=int)
        x = r3.element(x0)
        self.assertAllAlmostEquals(x, x0)

class TestAccessors(RLTestCase):
    def test_getitem(self):
        r3 = CudaRN(3)
        y = [1, 2, 3]
        x = r3.element(y)

        for index in [0, 1, 2, -1, -2, -3]:
            self.assertAlmostEquals(x[index], y[index])

    def test_iterator(self):
        r3 = CudaRN(3)
        y = [1, 2, 3]
        x = r3.element(y)

        self.assertAlmostEquals([a for a in x], [b for b in y])

    def test_getitem_index_error(self):
        r3 = CudaRN(3)
        x = r3.element([1, 2, 3])

        with self.assertRaises(IndexError):
            result = x[-4]

        with self.assertRaises(IndexError):
            result = x[3]

    def test_setitem(self):
        r3 = CudaRN(3)
        x = r3.element([42, 42, 42])

        for index in [0, 1, 2, -1, -2, -3]:
            x[index] = index
            self.assertAlmostEquals(x[index], index)

    def test_setitem_index_error(self):
        r3 = CudaRN(3)
        x = r3.element([1, 2, 3])

        with self.assertRaises(IndexError):
            x[-4] = 0

        with self.assertRaises(IndexError):
            x[3] = 0

    def _test_getslice(self, slice):
        # Validate get against python list behaviour
        r6 = CudaRN(6)
        y = [0, 1, 2, 3, 4, 5]
        x = r6.element(y)

        self.assertAllAlmostEquals(x[slice], y[slice])

    def test_getslice(self):
        # Tests getting all combinations of slices
        steps = [None, -2, -1, 1, 2]
        starts = [None, -1, -3, 0, 2, 5]
        ends = [None, -1, -3, 0, 2, 5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self._test_getslice(slice(start, end, step))

    def testGetSliceExceptions(self):
        r3 = CudaRN(3)
        xd = r3.element([1, 2, 3])

        # Bad slice
        with self.assertRaises(IndexError):
            result = xd[10:13]

    def _test_setslice(self, slice):
        # Validate set against python list behaviour
        r6 = CudaRN(6)
        z = [7, 8, 9, 10, 11, 10]
        y = [0, 1, 2, 3, 4, 5]
        x = r6.element(y)

        x[slice] = z[slice]
        y[slice] = z[slice]
        self.assertAllAlmostEquals(x, y)

    def test_setslice(self):
        # Tests a range of combination of slices
        steps = [None, -2, -1, 1, 2]
        starts = [None, -1, -3, 0, 2, 5]
        ends = [None, -1, -3, 0, 2, 5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self._test_setslice(slice(start, end, step))

    def test_setslice_index_error(self):
        r3 = CudaRN(3)
        xd = r3.element([1, 2, 3])

        # Bad slice
        with self.assertRaises(IndexError):
            xd[10:13] = [1, 2, 3]

        # Bad size of rhs
        with self.assertRaises(IndexError):
            xd[:] = []

        with self.assertRaises(IndexError):
            xd[:] = [1, 2]

        with self.assertRaises(IndexError):
            xd[:] = [1, 2, 3, 4]


class TestMethods(RLTestCase):
    def test_norm(self):
        r3 = CudaRN(3)
        xd = r3.element([1, 2, 3])

        correct_norm_squared = 1**2 + 2**2 + 3**2
        correct_norm = math.sqrt(correct_norm_squared)

        # Space function
        self.assertAlmostEquals(r3.norm(xd), correct_norm)

        # Member function
        self.assertAlmostEquals(xd.norm(), correct_norm)

    def test_inner(self):
        r3 = CudaRN(3)
        xd = r3.element([1, 2, 3])
        yd = r3.element([5, 3, 9])

        correct_inner = 1*5 + 2*3 + 3*9

        # Space function
        self.assertAlmostEquals(r3.inner(xd, yd), correct_inner)

        # Member function
        self.assertAlmostEquals(xd.inner(yd), correct_inner)

    def vectors(self, rn):
        # Generate numpy arrays
        x_arr = np.random.rand(rn.n)
        y_arr = np.random.rand(rn.n)
        z_arr = np.random.rand(rn.n)

        # Make rn vectors
        x, y, z = rn.element(x_arr), rn.element(y_arr), rn.element(z_arr)

        return x_arr, y_arr, z_arr, x, y, z

    def _test_lincomb(self, a, b, n=100):
        # Validates lincomb against the result on host with randomized
        # data and given a,b
        rn = CudaRN(n)

        # Unaliased arguments
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * x_arr + b * y_arr
        rn.lincomb(z, a, x, b, y)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

        # First argument aliased with output
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * z_arr + b * y_arr
        rn.lincomb(z, a, z, b, y)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

        # Second argument aliased with output
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * x_arr + b * z_arr
        rn.lincomb(z, a, x, b, z)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

        # Both arguments aliased with each other
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * x_arr + b * x_arr
        rn.lincomb(z, a, x, b, x)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

        # All aliased
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * z_arr + b * z_arr
        rn.lincomb(z, a, z, b, z)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

    def test_lincomb(self):
        scalar_values = [0, 1, -1, 3.41]
        for a in scalar_values:
            for b in scalar_values:
                self._test_lincomb(a, b)

    def _test_member_lincomb(self, a, n=100):
        # Validates vector member lincomb against the result on host with
        # randomized data
        n = 100

        # Generate vectors
        y_host = np.random.rand(n)
        x_host = np.random.rand(n)

        r3 = CudaRN(n)
        y_device = r3.element(y_host)
        x_device = r3.element(x_host)

        # Host side calculation
        y_host[:] = a*x_host

        # Device side calculation
        y_device.lincomb(a, x_device)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(y_device, y_host, places=5)

    def test_member_lincomb(self):
        scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
        for a in scalar_values:
            self._test_member_lincomb(a)

    def test_multiply(self):
        # Validates multiply against the result on host with randomized data
        n = 100
        y_host = np.random.rand(n)
        x_host = np.random.rand(n)

        r3 = CudaRN(n)
        y_device = r3.element(y_host)
        x_device = r3.element(x_host)

        # Host side calculation
        y_host[:] = x_host*y_host

        # Device side calculation
        r3.multiply(x_device, y_device)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(y_device, y_host, places=5)

    def test_member_multiply(self):
        # Validates vector member multiply against the result on host
        # with randomized data
        n = 100
        y_host = np.random.rand(n)
        x_host = np.random.rand(n)

        r3 = CudaRN(n)
        y_device = r3.element(y_host)
        x_device = r3.element(x_host)

        # Host side calculation
        y_host[:] = x_host*y_host

        # Device side calculation
        y_device.multiply(x_device)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(y_device, y_host, places=5)

class TestConvenience(RLTestCase):
    def test_addition(self):
        r3 = CudaRN(3)
        xd = r3.element([1, 2, 3])
        yd = r3.element([5, 3, 7])

        self.assertAllAlmostEquals(xd + yd, [6, 5, 10])

    def test_scalar_mult(self):
        r3 = CudaRN(3)
        xd = r3.element([1, 2, 3])
        C = 5

        self.assertAllAlmostEquals(C*xd, [5, 10, 15])

    def test_incompatible_operations(self):
        r3 = CudaRN(3)
        R3h = RN(3)
        xA = r3.zero()
        xB = R3h.zero()

        with self.assertRaises(TypeError):
            xA += xB

        with self.assertRaises(TypeError):
            xA -= xB

        with self.assertRaises(TypeError):
            z = xA+xB

        with self.assertRaises(TypeError):
            z = xA-xB

class TestPointer(RLTestCase):
    def test_get_ptr(self):
        r3 = CudaRN(3)
        x = r3.element([1, 2, 3])
        y = r3.element(RLcpp.PyCuda.vectorFromPointer(x.data_ptr, 3))
        self.assertAllAlmostEquals(x, y)
        self.assertEquals(x.data_ptr, y.data_ptr)

if __name__ == '__main__':
    unittest.main(exit=False)
