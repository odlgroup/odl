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
import math
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

    def testZero(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        self.assertAllAlmostEquals(R3d.zero(), R3h.zero())

    def testListInit(self):
        R = CudaRN(3)
        x = R.makeVector([1, 2, 3])
        self.assertAllAlmostEquals(x, [1, 2, 3])

    def testNPInit(self):
        R = CudaRN(3)
        
        x = R.makeVector(np.array([1.,2.,3.]))
        self.assertAllAlmostEquals(x, [1, 2, 3])

        x = R.makeVector(np.array([1,2,3],dtype = float))
        self.assertAllAlmostEquals(x, [1, 2, 3])

        x = R.makeVector(np.array([1,2,3],dtype = int))
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

    def testIterator(self):
        R3d = CudaRN(3)
        y = [1, 2, 3]
        x = R3d.makeVector(y)
        
        self.assertAlmostEquals([a for a in x],[b for b in y])
        
    def testGetExceptions(self):        
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

    def doGetCase(self,slice):
        #Validate get against python list behaviour
        R6d = CudaRN(6)
        y = [0, 1, 2, 3, 4, 5]
        x = R6d.makeVector(y)
        
        self.assertAllAlmostEquals(x[slice], y[slice])

    def testGetSlice(self):
        #Tests getting all combinations of slices
        steps = [None,-2,-1,1,2]
        starts = [None,-1,-3,0,2,5]
        ends = [None,-1,-3,0,2,5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self.doGetCase(slice(start,end,step))

    def doSetCase(self,slice):
        #Validate set against python list behaviour
        R6d = CudaRN(6)
        z = [7, 8, 9, 10, 11, 10]
        y = [0, 1, 2, 3, 4, 5]
        x = R6d.makeVector(y)

        x[slice] = z[slice]
        y[slice] = z[slice]
        self.assertAllAlmostEquals(x, y)

    def testSetSlice(self):
        #Tests a range of combination of slices
        steps = [None,-2,-1,1,2]
        starts = [None,-1,-3,0,2,5]
        ends = [None,-1,-3,0,2,5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self.doSetCase(slice(start,end,step))

class TestFunctions(RLTestCase):
    def testNorm(self):
        R3d = CudaRN(3)
        xd = R3d.makeVector([1, 2, 3])

        correct_norm_squared = 1**2 + 2**2 + 3**2
        correct_norm = math.sqrt(correct_norm_squared)
        
        self.assertAlmostEquals(R3d.normSq(xd),correct_norm_squared)
        self.assertAlmostEquals(R3d.norm(xd),correct_norm)
        self.assertAlmostEquals(xd.normSq(),correct_norm_squared)
        self.assertAlmostEquals(xd.norm(),correct_norm)

    def makeVectors(self, rn):
        #Generate numpy vectors
        y = np.random.rand(rn.dimension)
        x = np.random.rand(rn.dimension)  
        z = np.zeros(rn.dimension)       
        
        #Make rn vectors
        yVec = rn.makeVector(y)
        xVec = rn.makeVector(x)
        zVec = rn.makeVector(z)
        return x,y,z, xVec,yVec,zVec

    def doLincombTest(self, a, b, n=100):
        #Validates lincomb against the result on host with randomized data and given a,b
        
        rn = CudaRN(n)

        #Unaliased data
        x,y,z, xVec,yVec,zVec = self.makeVectors(rn)

        z[:] = a*x + b*y
        rn.linComb(zVec, a, xVec, b, yVec)
        self.assertAllAlmostEquals([xVec,yVec,zVec], [x,y,z], places=4)

        #One aliased
        x,y,z, xVec,yVec,zVec = self.makeVectors(rn)

        z[:] = a*z + b*y
        rn.linComb(zVec, a, zVec, b, yVec)
        self.assertAllAlmostEquals([xVec,yVec,zVec], [x,y,z], places=4)

        #One aliased
        x,y,z,xVec,yVec,zVec = self.makeVectors(rn)

        z[:] = a*z + b*y
        rn.linComb(zVec, a, zVec, b, yVec)
        self.assertAllAlmostEquals([xVec,yVec,zVec], [x,y,z], places=4)

        #All aliased
        x,y,z, xVec,yVec,zVec = self.makeVectors(rn)
        z[:] = a*z + b*z
        rn.linComb(zVec, a, zVec, b, zVec)
        self.assertAllAlmostEquals([xVec,yVec,zVec], [x,y,z], places=4)

    def testLinComb(self):
        scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
        for a in scalar_values:
            for b in scalar_values:
                self.doLincombTest(a, b)

    def doLinCombMemberTest(self, a, n=100):
        #Validates vector member lincomb against the result on host with randomized data
        n = 100
        
        #Generate vectors
        yHost = np.random.rand(n)
        xHost = np.random.rand(n)        
        
        R3d = CudaRN(n)
        yDevice = R3d.makeVector(yHost)
        xDevice = R3d.makeVector(xHost)
        
        #Host side calculation
        yHost[:] = a*xHost
        
        #Device side calculation
        yDevice.linComb(a, xDevice)
        
        self.assertAllAlmostEquals(yDevice, yHost, places=5) #Cuda only uses floats, so require 5 places

    def testMemberLinComb(self):
        scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
        for a in scalar_values:
            self.doLinCombMemberTest(a)

    def testMultiply(self):
        #Validates multiply against the result on host with randomized data
        n = 100
        yHost = np.random.rand(n)
        xHost = np.random.rand(n)        
        
        R3d = CudaRN(n)
        yDevice = R3d.makeVector(yHost)
        xDevice = R3d.makeVector(xHost)

        #Host side calculation
        yHost[:] = xHost*yHost
        
        #Device side calculation
        R3d.multiply(xDevice, yDevice)
        
        self.assertAllAlmostEquals(yDevice, yHost, places=5) #Cuda only uses floats, so require 5 places

    def testMemberMultiply(self):
        #Validates vector member multiply against the result on host with randomized data        
        n = 100
        yHost = np.random.rand(n)
        xHost = np.random.rand(n)        
        
        R3d = CudaRN(n)
        yDevice = R3d.makeVector(yHost)
        xDevice = R3d.makeVector(xHost)

        #Host side calculation
        yHost[:] = xHost*yHost
        
        #Device side calculation
        yDevice.multiply(xDevice)
        
        self.assertAllAlmostEquals(yDevice, yHost, places=5) #Cuda only uses floats, so require 5 places

class TestConvenience(RLTestCase):
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
