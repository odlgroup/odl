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
import RL.operator.operator as OP
from RL.space.space import *
from RL.space.defaultSpaces import *


class MultiplyOp(OP.LinearOperator):
    """Multiply with matrix
    """

    def __init__(self, matrix, domain = None, range = None):
        self._domain = EuclidianSpace(matrix.shape[1]) if domain is None else domain
        self._range = EuclidianSpace(matrix.shape[0]) if range is None else range
        self.matrix = matrix

    def applyImpl(self, rhs, out):
        out.values[:] = np.dot(self.matrix, rhs.values)

    def applyAdjointImpl(self, rhs, out):
        out.values[:] = np.dot(self.matrix.T, rhs.values)

    @property
    def domain(self):           
        return self._domain

    @property
    def range(self):            
        return self._range

class TestRN(RLTestCase):   
    def testSquareMultiplyOp(self):
        #Verify that the multiply op does indeed work as expected
        r3 = EuclidianSpace(3)

        A = np.random.rand(3, 3)
        x = np.random.rand(3)
        Aop = MultiplyOp(A)
        xvec = r3.makeVector(x)

        self.assertAllAlmostEquals(Aop(xvec), np.dot(A,x))


    def testNonSquareMultiplyOp(self):
        #Verify that the multiply op does indeed work as expected
        r3 = EuclidianSpace(3)

        A = np.random.rand(4, 3)
        x = np.random.rand(3)
        Aop = MultiplyOp(A)
        xvec = r3.makeVector(x)

        self.assertAllAlmostEquals(Aop(xvec), np.dot(A,x))


    def testAdjoint(self):
        r3 = EuclidianSpace(3)

        A = np.random.rand(3, 3)
        x = np.random.rand(3)
        Aop = MultiplyOp(A)
        xvec = r3.makeVector(x)

        self.assertAllAlmostEquals(Aop.T(xvec), np.dot(A.T,x))


    def testAdd(self):
        r3 = EuclidianSpace(3)

        A = np.random.rand(3, 3)
        B = np.random.rand(3, 3)
        x = np.random.rand(3)

        Aop = MultiplyOp(A)
        Bop = MultiplyOp(B)
        xvec = r3.makeVector(x)

        #Explicit instantiation
        C = OP.LinearOperatorSum(Aop, Bop)
        
        self.assertAllAlmostEquals(C(xvec), np.dot(A,x) + np.dot(B,x))
        self.assertAllAlmostEquals(C.T(xvec), np.dot(A.T,x) + np.dot(B.T,x))

        #Using operator overloading
        COverloading = Aop + Bop
        
        self.assertAllAlmostEquals(COverloading(xvec), np.dot(A,x) + np.dot(B,x))

    def testScale(self):
        r3 = EuclidianSpace(3)

        A = np.random.rand(3, 3)
        B = np.random.rand(3, 3)
        x = np.random.rand(3)

        Aop = MultiplyOp(A)
        xvec = r3.makeVector(x)

        #Test a range of scalars (scalar multiplication could implement optimizations for (-1, 0, 1).
        scalars = [-1.432, -1, 0, 1, 3.14]
        for scale in scalars:
            C = OP.LinearOperatorScalarMultiplication(Aop, scale)
        
            self.assertAllAlmostEquals(C(xvec), scale * np.dot(A,x))
            self.assertAllAlmostEquals(C.T(xvec), scale * np.dot(A.T,x))

            #Using operator overloading
            COverloadingLeft = scale * Aop
            COverloadingRight = Aop * scale
        
            self.assertAllAlmostEquals(COverloadingLeft(xvec), scale * np.dot(A,x))
            self.assertAllAlmostEquals(COverloadingRight(xvec), np.dot(A, scale * x))


    def testCompose(self):
        r3 = EuclidianSpace(3)

        A = np.random.rand(3, 3)
        B = np.random.rand(3, 3)
        x = np.random.rand(3)

        Aop = MultiplyOp(A)
        Bop = MultiplyOp(B)
        xvec = r3.makeVector(x)

        C = OP.LinearOperatorComposition(Aop, Bop)

        self.assertAllAlmostEquals(C(xvec), np.dot(A,np.dot(B,x)))
        self.assertAllAlmostEquals(C.T(xvec), np.dot(B.T,np.dot(A.T,x)))

    def testTypechecking(self):
        r3 = EuclidianSpace(3)
        r4 = EuclidianSpace(4)

        Aop = MultiplyOp(np.random.rand(3, 3))
        r3Vec1 = r3.zero()        
        r3Vec2 = r3.zero()
        r4Vec1 = r4.zero()
        r4Vec2 = r4.zero()
        
        #Verify that correct usage works
        Aop.apply(r3Vec1, r3Vec2)
        Aop.applyAdjoint(r3Vec1, r3Vec2)

        #Test that erroneous usage raises TypeError
        with self.assertRaises(TypeError):  
            Aop(r4Vec1)

        with self.assertRaises(TypeError):  
            Aop.T(r4Vec1)

        with self.assertRaises(TypeError):  
            Aop.apply(r3Vec1, r4Vec1)

        with self.assertRaises(TypeError):  
            Aop.applyAdjoint(r3Vec1, r4Vec1)

        with self.assertRaises(TypeError):  
            Aop.apply(r4Vec1, r3Vec1)

        with self.assertRaises(TypeError):  
            Aop.applyAdjoint(r4Vec1, r3Vec1)

        with self.assertRaises(TypeError):  
            Aop.apply(r4Vec1, r4Vec2)

        with self.assertRaises(TypeError):  
            Aop.applyAdjoint(r4Vec1, r4Vec2)

        #Check test against aliased values
        with self.assertRaises(ValueError):  
            Aop.apply(r3Vec1, r3Vec1)
            
        with self.assertRaises(ValueError):  
            Aop.applyAdjoint(r3Vec1, r3Vec1)


if __name__ == '__main__':
    unittest.main(exit=False)
