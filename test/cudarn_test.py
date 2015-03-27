# -*- coding: utf-8 -*-
"""
simple_test_astra.py -- a simple test script

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import numpy as np
from RL.operator.operatorAlternative import *
from RL.space.space import *
from RL.space.defaultSpaces import *
from RL.space.CudaSpace import *
from testutils import RLTestCase, Timer

class TestInit(RLTestCase):
    def testZero(self):
        R = CudaRN(3)
        x = R.zero()
        self.assertAllAlmostEquals(x, [0]*3)

    def testEmpty(self):
        R = CudaRN(3)
        x = R.empty()
        #Nothing to test, simply check that code runs

    def testNPInit(self):
        R = CudaRN(3)
        x = R.makeVector([1, 2, 3])
        self.assertAllAlmostEquals(x, [1, 2, 3])

class TestAccessors(RLTestCase):
    def testGetItem(self):
        R3d = CudaRN(3)
        y = [1, 2, 3]
        x = R3d.makeVector(y)
        
        self.assertAlmostEquals(x[0], y[0])
        self.assertAlmostEquals(x[1], y[1])
        self.assertAlmostEquals(x[2], y[2])
        self.assertAlmostEquals(x[-1], y[-1])
        self.assertAlmostEquals(x[-2], y[-2])
        self.assertAlmostEquals(x[-3], y[-3])
        
    def testGetItemOutOfBounds(self):        
        R3d = CudaRN(3)
        x = R3d.makeVector([1, 2, 3])

        with self.assertRaises(IndexError):
            result = x[-4]

        with self.assertRaises(IndexError):
            result = x[3]

    def testSetItem(self):
        R3d = CudaRN(3)
        x = R3d.makeVector([1, 2, 3])
        x[1] = 5

        self.assertAlmostEquals(x[1], 5)

    def testSetItemOutOfBounds(self):
        R3d = CudaRN(3)
        x = R3d.makeVector([1, 2, 3])

        with self.assertRaises(IndexError):
            x[-4] = 0

        with self.assertRaises(IndexError):
            x[3] = 0

    def testGetSlice(self):
        R5d = CudaRN(6)
        y = [1, 2, 3, 4, 5, 6]
        x = R5d.makeVector(y)

        self.assertAllAlmostEquals(x[1:1], y[1:1])
        self.assertAllAlmostEquals(x[2:1], y[2:1])
        self.assertAllAlmostEquals(x[1:2], y[1:2])
        self.assertAllAlmostEquals(x[::3], y[::3])
        self.assertAllAlmostEquals(x[:], y[:])
        self.assertAllAlmostEquals(x[::-1], y[::-1])
        self.assertAllAlmostEquals(x[:1:-1], y[:1:-1])
        self.assertAllAlmostEquals(x[::-2], y[::-2])
        self.assertAllAlmostEquals(x[:1:-3], y[:1:-3])

    def testSetSlice(self):
        R3d = CudaRN(3)  
        x0 = R3d.makeVector([1, 2, 3])
        
        x = x0.copy()
        x[1:3] = [5, 6]
        self.assertAllAlmostEquals(x, [1, 5, 6])

        x = x0.copy()
        x[:] = [7, 5, 6]
        self.assertAllAlmostEquals(x, [7, 5, 6])
        
        x = x0.copy()
        x[0::2] = [7, 6]
        self.assertAllAlmostEquals(x, [7, 2, 6])

        x = x0.copy()
        x[::-1] = [7, 6, 1]
        self.assertAllAlmostEquals(x, [1, 6, 7])
        
class TestLincomb(RLTestCase):
    def doLincombTest(self, a, b, n=100):
        #Validates lincomb against the result on host with randomized data and given a,b
        
        #Generate vectors
        yHost = np.random.rand(n)
        xHost = np.random.rand(n)        
        
        R3d = CudaRN(n)
        yDevice = R3d.makeVector(yHost)
        xDevice = R3d.makeVector(xHost)
        
        #Host side calculation
        yHost[:] = a*xHost + b*yHost
        
        #Device side calculation
        R3d.linComb(a,xDevice,b,yDevice)
        
        self.assertAllAlmostEquals(yDevice,yHost, places=5) #Cuda only uses floats, so require 5 places

    def testAllCases(self):

        # a = 0
        self.doLincombTest(0,0)
        self.doLincombTest(0,1)
        self.doLincombTest(0,-1)
        self.doLincombTest(0,3.252)

        # a = 1
        self.doLincombTest(1,0)
        self.doLincombTest(1,1)
        self.doLincombTest(1,-1)
        self.doLincombTest(1,6.4324)

        # a = -1
        self.doLincombTest(-1,0)
        self.doLincombTest(-1,1)
        self.doLincombTest(-1,-1)
        self.doLincombTest(-1,1.324)

        # a = arbitrary
        self.doLincombTest(5.234,0)
        self.doLincombTest(-3.32,1)
        self.doLincombTest(9.123,-1)
        self.doLincombTest(-1.23,6.4324)


class TestConvenience(RLTestCase):
    def testZero(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        self.assertAllAlmostEquals(R3d.zero(), R3h.zero())

    def testAddition(self):
        R3d = CudaRN(3)
        xd = R3d.makeVector([1, 2, 3])
        yd = R3d.makeVector([5, 3, 7])

        self.assertAllAlmostEquals(xd + yd, [6, 5, 10])

    def testScalarMult(self):
        R3d = CudaRN(3)
        xd = R3d.makeVector([1, 2, 3])
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
