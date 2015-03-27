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
from RL.space.space import *
from RL.space.defaultSpaces import *
from RL.space.functionSpaces import *
from testutils import RLTestCase


class L2Test(RLTestCase):
    def testR(self):
        I = Interval(0, pi)
        space = L2(I)
        d = UniformDiscretization(space, 10)

        l2sin = space.makeVector(np.sin)
        sind = d.makeVector(l2sin)

        self.assertAlmostEqual(sind.normSq(), pi/2, places=10)

if __name__ == '__main__':
    unittest.main(exit=False)
