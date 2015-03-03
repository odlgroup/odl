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
from __future__ import division,print_function, unicode_literals,absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
from math import pi

import numpy as np
from RL.operator.operatorAlternative import *
from RL.operator.space import *
import SimRec2DPy as SR

class RealsTest(unittest.TestCase):
    def testAddition(self):
        R = Reals()
        x = R.makeVector(1.0)
        y = R.makeVector(2.0)
        R.linearComb(2,x,3,y)

        z = R.makeVector(8.0)

        self.assertAlmostEqual(y,z)

class RNTest(unittest.TestCase):
    def testAddition(self):
        r3 = RN(3)
        x = r3.makeVector([1.,2.,3.])
        y = r3.makeVector([3.,5.,7.])
        z = r3.makeVector([4.,7.,10.])
        r3.linearComb(1,x,1,y)
        self.assertTrue(np.allclose(y,z))

    def testMultiply(self):
        r3 = RN(3)

        A = np.random.rand(3,3)
        Aop = r3.MultiplyOp(A)
        x = np.random.rand(3)

        self.assertTrue(np.allclose(Aop(x),np.dot(A,x)))

    def testAdjoint(self):
        r3 = RN(3)

        A = r3.makeVector(np.random.rand(3,3))
        Aop = r3.MultiplyOp(A)
        x = r3.makeVector(np.random.rand(3))
        y = r3.makeVector(np.random.rand(3))

        self.assertAlmostEqual(r3.inner(Aop(x),y),r3.inner(x,Aop.applyAdjoint(y)))


class ProductTest(unittest.TestCase):
    def testRxR(self):
        A = Reals()
        B = Reals()
        C = ProductSpace(A,B)

        v1 = A.makeVector(1.0)
        v2 = B.makeVector(2.0)
        v3 = C.makeVector(v1,v2)

        self.assertAlmostEquals(v1,v3[0])
        self.assertAlmostEquals(v2,v3[1])
        self.assertTrue(C.dimension == 2)

    def testConstructR1xR2(self):
        r1 = RN(1)
        r2 = RN(2)
        S = ProductSpace(r1,r2)

        v1 = r1.makeVector([1.0])
        v2 = r2.makeVector([2.0,3.0])
        v = S.makeVector(v1,v2)
        
        self.assertTrue(S.dimension == 3)
        self.assertAlmostEquals(v1[0],v[0][0])
        self.assertAlmostEquals(v2[0],v[1][0])
        self.assertAlmostEquals(v2[1],v[1][1])
        self.assertAlmostEquals(S.normSquared(v),r1.normSquared(v1)+r2.normSquared(v2))

    def testArbitraryProduct(self):
        s1 = Reals()
        s2 = Reals()
        s3 = Reals()
        S = ProductSpace(s1,s2,s3)

        v1 = s1.makeVector(1.0)
        v2 = s2.makeVector(2.0)
        v3 = s3.makeVector(3.0)
        v = S.makeVector(v1,v2,v3)
        
        self.assertTrue(S.dimension == 3)
        self.assertAlmostEquals(v1,v[0])
        self.assertAlmostEquals(v2,v[1])
        self.assertAlmostEquals(v3,v[2])
        self.assertAlmostEquals(S.normSquared(v),s1.normSquared(v1)+s2.normSquared(v2)+s3.normSquared(v3))

class L2Test(unittest.TestCase):
    def testInit(self):
        I = Interval(0,pi)
        d = LinspaceDiscretization(I,1000)
        space = L2(d)

        s = space.sin()
        
        self.assertAlmostEqual(space.normSquared(s),pi/2,2)

if __name__ == '__main__':
    unittest.main(exit = False)
