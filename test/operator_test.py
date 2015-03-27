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
from testutils import RLTestCase

import numpy as np
import RL.operator.operatorAlternative as OP
from RL.space.space import *
from RL.space.defaultSpaces import *


class MultiplyOp(OP.LinearOperator):
    """Multiply with matrix
    """

    def __init__(self, space, matrix):
        self.space = space
        self.matrix = matrix

    def applyImpl(self, rhs, out):
        out.values[:] = np.dot(self.matrix, rhs.values)

    def applyAdjointImpl(self, rhs, out):
        out.values[:] = np.dot(self.matrix.T, rhs.values)

    @property
    def domain(self):           
        return self.space

    @property
    def range(self):            
        return self.space

class TestRN(RLTestCase):   
    def testMultiply(self):
        r3 = EuclidianSpace(3)

        A = np.random.rand(3, 3)
        x = np.random.rand(3)
        Aop = MultiplyOp(r3, A)
        xvec = r3.makeVector(x)

        self.assertAllAlmostEquals(Aop(xvec), np.dot(A,x))

    def testAdjoint(self):
        r3 = EuclidianSpace(3)

        A = MultiplyOp(r3, np.random.rand(3,3))
        x = r3.makeVector(np.random.rand(3))
        y = r3.makeVector(np.random.rand(3))

        self.assertAlmostEqual(A(x).inner(y), x.inner(A.T(y)))

    def testCompose(self):
        r3 = EuclidianSpace(3)

        A = np.random.rand(3, 3)
        B = np.random.rand(3, 3)
        x = np.random.rand(3)

        Aop = MultiplyOp(r3, A)
        Bop = MultiplyOp(r3, B)
        xvec = r3.makeVector(x)

        C = OP.OperatorComposition(Aop, Bop)

        self.assertAllAlmostEquals(C(xvec), np.dot(A,np.dot(B,x)))

    def testTypechecking(self):
        r3 = EuclidianSpace(3)
        r4 = EuclidianSpace(4)

        Aop = MultiplyOp(r3, np.random.rand(3, 3))
        r3Vec = r3.zero()
        r4Vec = r4.zero()
        
        #Test that "correct" usage works
        Aop.apply(r3Vec, r3Vec)
        Aop.applyAdjoint(r3Vec, r3Vec)

        #Test that erroneous usage raises TypeError
        with self.assertRaises(TypeError):  
            Aop(r4Vec)

        with self.assertRaises(TypeError):  
            Aop.T(r4Vec)

        with self.assertRaises(TypeError):  
            Aop.apply(r3Vec, r4Vec)

        with self.assertRaises(TypeError):  
            Aop.applyAdjoint(r3Vec, r4Vec)

        with self.assertRaises(TypeError):  
            Aop.apply(r4Vec, r4Vec)

        with self.assertRaises(TypeError):  
            Aop.applyAdjoint(r4Vec, r4Vec)


if __name__ == '__main__':
    unittest.main(exit=False)
