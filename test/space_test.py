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
from itertools import chain
from collections import Iterable

import numpy as np
from RL.operator.operatorAlternative import *
from RL.operator.space import *
import SimRec2DPy as SR


#Todo move
class RLTestCase(unittest.TestCase):
    def assertAlmostEqual(self,f1,f2):
        unittest.TestCase.assertAlmostEqual(self,float(f1),float(f2))

    def assertAllAlmostEquals(self,iter1,iter2):
        for [i1,i2] in zip(iter1,iter2):
            if isinstance(i1, Iterable):
                self.assertAllAlmostEquals(i1,i2)
            else:
                self.assertAlmostEqual(i1,i2)

class RealsTest(RLTestCase):
    def testAddition(self):
        R = Reals()
        x = R.makeVector(1.0)
        y = R.makeVector(4.0)
        z = R.makeVector(5.0)

        self.assertAlmostEqual(x+y,z)

class RNTest(RLTestCase):
    def testLinComb(self):
        R3 = RN(3)
        x = R3.makeVector([1.,2.,3.])
        y = R3.makeVector([3.,5.,7.])
        z = R3.makeVector([4.,7.,10.])
        R3.linComb(1,x,1,y)
        
        self.assertAllAlmostEquals(y,z)

    def testAddition(self):
        R3 = RN(3)
        x = R3.makeVector([1.,2.,3.])
        y = R3.makeVector([3.,5.,7.])
        z = R3.makeVector([4.,7.,10.])
        
        self.assertAllAlmostEquals(x+y,z)

class ProductTest(RLTestCase):
    def testRxR(self):
        R = Reals()
        R2 = ProductSpace(R,R)
        self.assertTrue(R2.dimension == 2)

        v1 = R.makeVector(1.0)
        v2 = R.makeVector(2.0)
        v = R2.makeVector(v1,v2)
        u = R2.makeVector(1.0,2.0)
        
        self.assertAllAlmostEquals([v1,v2],v)
        self.assertAllAlmostEquals([v1,v2],u)

    def testAdd(self):
        R = Reals()
        R2 = ProductSpace(R,R)

        u = R2.makeVector(1.0,4.0)
        v = R2.makeVector(3.0,7.0)

        sum = u+v
        expected = R2.makeVector(u[0]+v[0],u[1]+v[1])
        
        self.assertAllAlmostEquals(sum,expected)

    def testLinComb(self):
        R = Reals()
        R2 = ProductSpace(R,R)

        u = R2.makeVector(1.0,4.0)
        v = R2.makeVector(3.0,7.0)
        a = 4.0
        b = 2.0
        expected = R2.makeVector(a*u[0]+b*v[0],a*u[1]+b*v[1])

        R2.linComb(a,u,b,v)
        
        self.assertAllAlmostEquals(v,expected)

    def testConstructR1xR2(self):
        R1 = RN(1)
        R2 = RN(2)
        S = ProductSpace(R1,R2)
        self.assertTrue(S.dimension == 3)

        v1 = R1.makeVector([1.0])
        v2 = R2.makeVector([2.0,3.0])
        v = S.makeVector(v1,v2)
        
        self.assertAllAlmostEquals([v1,v2],v)
        self.assertAlmostEqual(v.normSquared(),v1.normSquared()+v2.normSquared())

    def testArbitraryProduct(self):
        R = Reals()
        R3 = ProductSpace(R,R,R)
        self.assertTrue(R3.dimension == 3)

        v1 = R.makeVector(1.0)
        v2 = R.makeVector(2.0)
        v3 = R.makeVector(3.0)
        v = R3.makeVector(1.0,2.0,3.0)
        
        self.assertAllAlmostEquals([v1,v2,v3],v)
        self.assertAlmostEqual(v.normSquared(),v1.normSquared()+v2.normSquared()+v3.normSquared())

class L2Test(RLTestCase):
    def testR(self):
        I = Interval(0,pi)
        d = UniformDiscretization(I,1000)
        m = BorelMeasure()
        measureSpace = DiscreteMeaureSpace(d,m)
        space = L2(measureSpace)

        class SinFunction(L2.Vector):
            def apply(self,rhs):
                return sin(rhs)

        s = SinFunction(space)
        
        self.assertAlmostEqual(s.normSquared(),pi/2)

if __name__ == '__main__':
    unittest.main(exit = False)
