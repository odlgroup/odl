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
from math import pi

import numpy as np
from RL.operator.operatorAlternative import *
import RL.space.space as space
import RL.space.defaultSpaces as ds
import RL.space.defaultDiscretizations as dd
import RL.space.functionSpaces as fs
import RL.space.set as sets
from testutils import RLTestCase


class L2Test(RLTestCase):
    def testInterval(self):
        I = sets.Interval(0, pi)
        l2 = fs.L2(I)
        rn = ds.EuclidianSpace(10)
        d = dd.makeUniformDiscretization(l2, rn)

        l2sin = l2.makeVector(np.sin)
        sind = d.makeVector(l2sin)

        self.assertAlmostEqual(sind.normSq(), pi/2)

    def testSquare(self):
        I = sets.Square((0, 0), (pi, pi))
        l2 = fs.L2(I)
        n = 10
        m = 10
        rn = ds.EuclidianSpace(n*m)
        d =  dd.makePixelDiscretization(l2, rn, n, m)

        l2sin = l2.makeVector(lambda point: np.sin(point[0]) * np.sin(point[1]))
        sind = d.makeVector(l2sin)

        self.assertAlmostEqual(sind.normSq(), pi**2 / 4)

if __name__ == '__main__':
    unittest.main(exit=False)
