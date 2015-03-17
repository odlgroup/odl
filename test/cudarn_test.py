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
    def testAddition(self):
        R = CudaRN(3)
        x = R.makeVector(np.array([1.0,2.0,3.0]))
        y = R.makeVector(np.array([1.0,2.0,3.0]))
        z = R.empty()
        print(z)

        print(x+y)

class TestRNInteractions(RLTestCase):
    def testAddition(self):
        R3c = CudaRN(3)
        R3 = RN(3)
        x = R3c.makeVector(np.array([1.0,2.0,3.0]))
        y = R3c.makeVector(np.array([1.0,2.0,3.0]))

        z = x.asRNVector(R3)

        print(z)

        print(x+y)

if __name__ == '__main__':
    unittest.main(exit = False)
