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

class BasicOperator(unittest.TestCase):
    def testMultiply(self):
        A = Reals.MultiplyOp(3)
        x = 2
        self.assertAlmostEqual(A(x),3*x)

    def testAddition(self):
        A = Reals.AddOp(3)
        x = 2
        self.assertAlmostEqual(A(x),3+x)

class OperatorArithmetic(unittest.TestCase):
    def testAddition(self):
        A = Reals.MultiplyOp(3)
        B = Reals.MultiplyOp(5)
        x = 2
        self.assertAlmostEqual((A+B)(2),3*x+5*x)

    def testComposition(self):
        A = Reals.MultiplyOp(3)
        B = Reals.MultiplyOp(5)
        x = 2
        self.assertAlmostEqual((A*B)(2),5*3*x)

    def testComplex(self):
        A = Reals.MultiplyOp(3)
        B = Reals.MultiplyOp(5)
        C = Reals.AddOp(5)
        x = 2
        self.assertAlmostEqual((A*(B+C))(2),3*(5*x+5+x))

class RNTest(unittest.TestCase):
    def testMultiply(self):
        r3 = RN(3)

        A = np.random.rand(3,3)
        Aop = r3.MultiplyOp(A)
        x = np.random.rand(3)

        self.assertTrue(np.allclose(Aop(x),np.dot(A,x)))

    def testAdjoint(self):
        r3 = RN(3)

        A = np.random.rand(3,3)
        Aop = r3.MultiplyOp(A)
        x = np.random.rand(3)
        y = np.random.rand(3)

        self.assertAlmostEqual(r3.inner(Aop(x),y),r3.inner(x,Aop.applyAdjoint(y)))


class ProductTest(unittest.TestCase):
    def testInit(self):
        A = Reals()
        B = Reals()
        C = ProductSpace(A,B)

        self.assertTrue(C.dimension() == 2)

class L2Test(unittest.TestCase):
    def testInit(self):
        d = linspaceDiscretization(0,pi,1000)
        space = L2(d)

        s = space.Sin()
        
        self.assertAlmostEqual(space.squaredNorm(s),pi/2,2)

if __name__ == '__main__':
    unittest.main(exit = False)
