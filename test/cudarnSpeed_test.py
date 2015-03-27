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

import numpy as np
from RL.operator.operatorAlternative import *
from RL.space.space import *
from RL.space.defaultSpaces import *
from RL.space.CudaSpace import *
from testutils import RLTestCase, Timer

class TestCompareWithHost(RLTestCase):
    def makeRandomVectors(self, n):
        deviceSpace = CudaRN(n)
        hostSpace = RN(n)
        x = np.random.rand(n)
        y = np.random.rand(n)

        xDevice = deviceSpace.makeVector(x)
        yDevice = deviceSpace.makeVector(y)
        xHost = hostSpace.makeVector(x)
        yHost = hostSpace.makeVector(y)
        return xDevice, yDevice, xHost, yHost

    def testAdd(self):
        xDevice, yDevice, xHost, yHost = self.makeRandomVectors(10**7)

        with Timer('CUDA x+y'):
            for _ in range(10):
                zDevice = xDevice+yDevice

        with Timer('CPU x+y'):
            for _ in range(10):
                zHost = xHost+yHost

        #Only check equality on some elements for efficiency
        self.assertAllAlmostEquals(zDevice[:100], zHost[:100], places=5)

    def testIncAdd(self):
        xDevice, yDevice, xHost, yHost = self.makeRandomVectors(10**7)

        with Timer('CUDA x+=y'):
            for _ in range(10):
                xDevice += yDevice

        with Timer('CPU x+=y'):
            for _ in range(10):
                xHost += yHost

        #Only check equality on some elements for efficiency
        self.assertAllAlmostEquals(xDevice[:100], xHost[:100], places=5)

    def testNoOp(self):
        xDevice, yDevice, xHost, yHost = self.makeRandomVectors(10**7)

        with Timer('CUDA x = x'):
            for _ in range(10):
                xDevice.linComb(0,yDevice)

        with Timer('CPU x = x'):
            for _ in range(10):
                xHost.linComb(0,yHost)

        #Only check equality on some elements for efficiency
        self.assertAllAlmostEquals(xDevice[:100], xHost[:100], places=5)


if __name__ == '__main__':
    unittest.main(exit=False)
