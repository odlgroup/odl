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
from RL.space.cuda import *
from RL.space.euclidean import RN
from RL.utility.testutils import RLTestCase, Timer

import numpy as np

standard_library.install_aliases()


class TestInit(RLTestCase):
    def testEmpty(self):
        R = CudaRN(3)
        x = R.element()
        # Nothing to test, simply check that code runs

    def testZero(self):
        R3d = CudaRN(3)
        self.assertAllAlmostEquals(R3d.zero(), [0, 0, 0])

    def testListInit(self):
        R = CudaRN(3)
        x = R.element([1, 2, 3])
        self.assertAllAlmostEquals(x, [1, 2, 3])

    def testNPInit(self):
        R = CudaRN(3)

        x0 = np.array([1., 2., 3.])
        x = R.element(x0)
        self.assertAllAlmostEquals(x, x0)

        x0 = np.array([1, 2, 3], dtype=float64)
        x = R.element(x0)
        self.assertAllAlmostEquals(x, x0)

        x0 = np.array([1, 2, 3], dtype=int)
        x = R.element(x0)
        self.assertAllAlmostEquals(x, x0)


class TestAccessors(RLTestCase):
    def testGetItem(self):
        R3d = CudaRN(3)
        y = [1, 2, 3]
        x = R3d.element(y)

        for index in [0, 1, 2, -1, -2, -3]:
            self.assertAlmostEquals(x[index], y[index])

    def testIterator(self):
        R3d = CudaRN(3)
        y = [1, 2, 3]
        x = R3d.element(y)

        self.assertAlmostEquals([a for a in x], [b for b in y])

    def testGetExceptions(self):
        R3d = CudaRN(3)
        x = R3d.element([1, 2, 3])

        with self.assertRaises(IndexError):
            result = x[-4]

        with self.assertRaises(IndexError):
            result = x[3]

    def testSetItem(self):
        R3d = CudaRN(3)
        x = R3d.element([42,42,42])

        for index in [0, 1, 2, -1, -2, -3]:
            x[index] = index
            self.assertAlmostEquals(x[index], index)

    def testSetItemOutOfBounds(self):
        R3d = CudaRN(3)
        x = R3d.element([1, 2, 3])

        with self.assertRaises(IndexError):
            x[-4] = 0

        with self.assertRaises(IndexError):
            x[3] = 0

    def doGetSliceCase(self, slice):
        # Validate get against python list behaviour
        R6d = CudaRN(6)
        y = [0, 1, 2, 3, 4, 5]
        x = R6d.element(y)

        self.assertAllAlmostEquals(x[slice], y[slice])

    def testGetSlice(self):
        # Tests getting all combinations of slices
        steps = [None, -2, -1, 1, 2]
        starts = [None, -1, -3, 0, 2, 5]
        ends = [None, -1, -3, 0, 2, 5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self.doGetSliceCase(slice(start, end, step))

    def testGetSliceExceptions(self):
        R3d = CudaRN(3)
        xd = R3d.element([1, 2, 3])

        #Bad slice
        with self.assertRaises(IndexError):
            result = xd[10:13]

    def doSetSliceCase(self, slice):
        # Validate set against python list behaviour
        R6d = CudaRN(6)
        z = [7, 8, 9, 10, 11, 10]
        y = [0, 1, 2, 3, 4, 5]
        x = R6d.element(y)

        x[slice] = z[slice]
        y[slice] = z[slice]
        self.assertAllAlmostEquals(x, y)

    def testSetSlice(self):
        # Tests a range of combination of slices
        steps = [None, -2, -1, 1, 2]
        starts = [None, -1, -3, 0, 2, 5]
        ends = [None, -1, -3, 0, 2, 5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self.doSetSliceCase(slice(start, end, step))

    def testSetSliceExceptions(self):
        R3d = CudaRN(3)
        xd = R3d.element([1, 2, 3])

        #Bad slice
        with self.assertRaises(IndexError):
            xd[10:13] = [1, 2, 3]

        #Bad size of rhs
        with self.assertRaises(IndexError):
            xd[:] = []

        with self.assertRaises(IndexError):
            xd[:] = [1, 2]

        with self.assertRaises(IndexError):
            xd[:] = [1, 2, 3, 4]


class TestFunctions(RLTestCase):
    def testNorm(self):
        R3d = CudaRN(3)
        xd = R3d.element([1, 2, 3])

        correct_norm_squared = 1**2 + 2**2 + 3**2
        correct_norm = math.sqrt(correct_norm_squared)

        #Space function
        self.assertAlmostEquals(R3d.norm(xd), correct_norm)

        #Member function
        self.assertAlmostEquals(xd.norm(), correct_norm)

    def testInner(self):
        R3d = CudaRN(3)
        xd = R3d.element([1, 2, 3])
        yd = R3d.element([5, 3, 9])

        correct_inner = 1*5 + 2*3 + 3*9

        #Space function
        self.assertAlmostEquals(R3d.inner(xd,yd), correct_inner)

        #Member function
        self.assertAlmostEquals(xd.inner(yd), correct_inner)

    def elements(self, rn):
        # Generate numpy vectors
        x, y, z = np.random.rand(rn.n), np.random.rand(rn.n), np.random.rand(rn.n)

        # Make rn vectors
        xVec, yVec, zVec = rn.element(x), rn.element(y), rn.element(z)

        return x, y, z, xVec, yVec, zVec

    def dolincombTest(self, a, b, n=100):
        # Validates lincomb against the result on host with randomized
        # data and given a,b
        rn = CudaRN(n)

        # Unaliased arguments
        x, y, z, xVec, yVec, zVec = self.elements(rn)

        z[:] = a*x + b*y
        rn.lincomb(zVec, a, xVec, b, yVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z], places=4)

        # First argument aliased with output
        x, y, z, xVec, yVec, zVec = self.elements(rn)

        z[:] = a*z + b*y
        rn.lincomb(zVec, a, zVec, b, yVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z], places=4)

        # Second argument aliased with output
        x, y, z, xVec, yVec, zVec = self.elements(rn)

        z[:] = a*x + b*z
        rn.lincomb(zVec, a, xVec, b, zVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z], places=4)

        # Both arguments aliased with each other
        x, y, z, xVec, yVec, zVec = self.elements(rn)

        z[:] = a*x + b*x
        rn.lincomb(zVec, a, xVec, b, xVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z], places=4)

        # All aliased
        x, y, z, xVec, yVec, zVec = self.elements(rn)
        z[:] = a*z + b*z
        rn.lincomb(zVec, a, zVec, b, zVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z], places=4)

    def testlincomb(self):
        scalar_values = [0, 1, -1, 3.41]
        for a in scalar_values:
            for b in scalar_values:
                self.dolincombTest(a, b)

    def dolincombMemberTest(self, a, n=100):
        # Validates vector member lincomb against the result on host with
        # randomized data
        n = 100

        # Generate vectors
        yHost = np.random.rand(n)
        xHost = np.random.rand(n)

        R3d = CudaRN(n)
        yDevice = R3d.element(yHost)
        xDevice = R3d.element(xHost)

        # Host side calculation
        yHost[:] = a*xHost

        # Device side calculation
        yDevice.lincomb(a, xDevice)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(yDevice, yHost, places=5)

    def testMemberlincomb(self):
        scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
        for a in scalar_values:
            self.dolincombMemberTest(a)

    def testMultiply(self):
        # Validates multiply against the result on host with randomized data
        n = 100
        yHost = np.random.rand(n)
        xHost = np.random.rand(n)

        R3d = CudaRN(n)
        yDevice = R3d.element(yHost)
        xDevice = R3d.element(xHost)

        # Host side calculation
        yHost[:] = xHost*yHost

        # Device side calculation
        R3d.multiply(xDevice, yDevice)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(yDevice, yHost, places=5)

    def testMemberMultiply(self):
        # Validates vector member multiply against the result on host
        # with randomized data
        n = 100
        yHost = np.random.rand(n)
        xHost = np.random.rand(n)

        R3d = CudaRN(n)
        yDevice = R3d.element(yHost)
        xDevice = R3d.element(xHost)

        # Host side calculation
        yHost[:] = xHost*yHost

        # Device side calculation
        yDevice.multiply(xDevice)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(yDevice, yHost, places=5)


class TestConvenience(RLTestCase):
    def testAddition(self):
        R3d = CudaRN(3)
        xd = R3d.element([1, 2, 3])
        yd = R3d.element([5, 3, 7])

        self.assertAllAlmostEquals(xd + yd, [6, 5, 10])

    def testScalarMult(self):
        R3d = CudaRN(3)
        xd = R3d.element([1, 2, 3])
        C = 5

        self.assertAllAlmostEquals(C*xd, [5, 10, 15])

    def testIncompatibleOperations(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        xA = R3d.zero()
        xB = R3h.zero()

        with self.assertRaises(TypeError):
            xA += xB

        with self.assertRaises(TypeError):
            xA -= xB

        with self.assertRaises(TypeError):
            z = xA+xB

        with self.assertRaises(TypeError):
            z = xA-xB

if __name__ == '__main__':
    unittest.main(exit=False)
