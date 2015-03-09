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

class RNTest(unittest.TestCase):
    def testMultiply(self):
        r3 = RN(3)

        A = np.random.rand(3,3)
        x = np.random.rand(3)
        Aop = r3.MakeMultiplyOp(A)
        xvec = r3.makeVector(x)

        self.assertTrue(np.allclose(Aop(xvec),np.dot(A,x)))

    def testAdjoint(self):
        r3 = RN(3)

        A = r3.MakeMultiplyOp(np.random.rand(3,3))
        x = r3.makeVector(np.random.rand(3))
        y = r3.makeVector(np.random.rand(3))

        self.assertAlmostEqual(A(x).inner(y),x.inner(A.T(y)))

if __name__ == '__main__':
    unittest.main(exit = False)