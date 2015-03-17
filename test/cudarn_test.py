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
from RL.operator.defaultSpaces import *
from RL.operator.CudaSpace import *
import SimRec2DPy as SR
from testutils import RLTestCase

class TestInit(RLTestCase):
    def testZero(self):
        R = CudaRN(3)
        x = R.zero()

    def testEmpty(self):
        R = CudaRN(3)
        x = R.empty()

    def testNPInit(self):
        R = CudaRN(3)
        x = R.makeVector(np.array([1.0,2.0,3.0]))

class TestAccessors(RLTestCase):
    def testGetItem(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        x = R3d.makeVector(np.array([1.0,2.0,3.0]))

        self.assertAlmostEquals(x[1],2.0)

    def testSetItem(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        x = R3d.makeVector(np.array([1.0,2.0,3.0]))
        x[1]=5.0

        self.assertAlmostEquals(x[1],5.0)

    def testGetSlice(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        x = R3d.makeVector(np.array([1.0,2.0,3.0]))

        self.assertAllAlmostEquals(x[1:2],[2.0,3.0])

    def testSetSlice(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        x = R3d.makeVector(np.array([1.0,2.0,3.0]))
        x[1:2] = np.array([5.0,6.0])

        self.assertAllAlmostEquals(x,[1.0,5.0,6.0])

class TestRNInteractions(RLTestCase):
    def testZero(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        self.assertAllAlmostEquals(R3d.zero(),R3h.zero())

    def testAddition(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        xd = R3d.makeVector(np.array([1.0,2.0,3.0]))
        yd = R3d.makeVector(np.array([5.0,3.0,7.0]))
        xh = R3h.makeVector(np.array([1.0,2.0,3.0]))
        yh = R3h.makeVector(np.array([5.0,3.0,7.0]))

        self.assertAllAlmostEquals(xd + yd,xh + yh)

    def testScalarMult(self):
        R3d = CudaRN(3)
        R3h = RN(3)
        xd = R3d.makeVector(np.array([1.0,2.0,3.0]))
        xh = R3h.makeVector(np.array([1.0,2.0,3.0]))
        C = 5

        self.assertAllAlmostEquals(C*xd,C*xh)

if __name__ == '__main__':
    unittest.main(exit = False)
