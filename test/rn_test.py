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

from RL.operator.operatorAlternative import *
from RL.space.space import *
from RL.space.defaultSpaces import *
from RL.space.functionSpaces import *
from testutils import RLTestCase

class RealsTest(RLTestCase):
    def testAddition(self):
        R = Reals()
        x = R.makeVector(1)
        y = R.makeVector(4)

        self.assertAlmostEqual(x+y, 5)

class ProductTest(RLTestCase):
    def testRxR(self):
        R = Reals()
        R2 = ProductSpace(R, R)
        self.assertTrue(R2.dimension == 2)

        v1 = R.makeVector(1)
        v2 = R.makeVector(2)
        v = R2.makeVector(v1, v2)
        u = R2.makeVector(1, 2)
        
        self.assertAllAlmostEquals([v1, v2], v)
        self.assertAllAlmostEquals([v1, v2], u)

    def testAdd(self):
        R = Reals()
        R2 = ProductSpace(R, R)

        u = R2.makeVector(1, 4)
        v = R2.makeVector(3, 7)

        sum = u+v
        expected = R2.makeVector(u[0]+v[0], u[1]+v[1])
        
        self.assertAllAlmostEquals(sum, expected)

    def testLinComb(self):
        R = Reals()
        R2 = ProductSpace(R, R)

        u = R2.makeVector(1, 4)
        v = R2.makeVector(3, 7)
        a = 4
        b = 2
        expected = R2.makeVector(a*u[0] + b*v[0], a*u[1] + b*v[1])

        R2.linComb(a, u, b, v)
        
        self.assertAllAlmostEquals(v, expected)

    def testConstructR1xR2(self):
        R3 = EuclidianSpace(3)
        R2 = EuclidianSpace(2)
        S = ProductSpace(R3, R2)
        self.assertTrue(S.dimension == 2)

        v1 = R3.makeVector([1, 5, 7])
        v2 = R2.makeVector([2, 3])
        v = S.makeVector(v1, v2)

        self.assertAllAlmostEquals([v1, v2], v)
        self.assertAlmostEqual(v.normSq(), v1.normSq()+v2.normSq())

    def testArbitraryProduct(self):
        R = Reals()
        R3 = ProductSpace(R, R, R)
        self.assertTrue(R3.dimension == 3)

        v1 = R.makeVector(1)
        v2 = R.makeVector(2)
        v3 = R.makeVector(3)
        v = R3.makeVector(1, 2, 3)
        
        self.assertAllAlmostEquals([v1, v2, v3], v)
        self.assertAlmostEqual(v.normSq(), v1.normSq()+v2.normSq()+v3.normSq())


if __name__ == '__main__':
    unittest.main(exit=False)
