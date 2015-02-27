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

import numpy as np
from RL.operator.operatorAlternative import *

class BasicOperator(unittest.TestCase):
    def testMultiply(self):
        A = Reals.MultiplyOp(3)
        x = 2
        self.failUnless(A(x) == 3*x)

    def testAddition(self):
        A = Reals.AddOp(3)
        x = 2
        self.failUnless(A(x) == 3+x)

class OperatorArithmetic(unittest.TestCase):
    def testAddition(self):
        A = Reals.MultiplyOp(3)
        B = Reals.MultiplyOp(5)
        x = 2
        self.failUnless((A+B)(2) == 3*x+5*x)

    def testComposition(self):
        A = Reals.MultiplyOp(3)
        B = Reals.MultiplyOp(5)
        x = 2
        self.failUnless((A*B)(2) == 5*3*x)

    def testComplex(self):
        A = Reals.MultiplyOp(3)
        B = Reals.MultiplyOp(5)
        C = Reals.AddOp(5)
        x = 2
        self.failUnless((A*(B+C))(2) == 3*(5*x+5+x))

class R3Test(unittest.TestCase):
    def testMultiply(self):
        A = np.random.rand(3,3)
        Aop = R3.MultiplyOp(A)
        x = np.random.rand(3)

        self.failUnless((Aop(x) == np.dot(A,x)).all())

    def testAdjoint(self):
        A = np.random.rand(3,3)
        Aop = R3.MultiplyOp(A)
        x = np.random.rand(3)
        y = np.random.rand(3)

        self.failUnless(R3.inner(Aop(x),y) == R3.inner(x,Aop.T(y)))

def main():
    unittest.main(exit = False)

if __name__ == '__main__':
    main()
