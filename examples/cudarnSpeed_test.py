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

n=10**7
iterations = 100
deviceSpace = CudaRN(n)
hostSpace = RN(n)
x = np.random.rand(n)
y = np.random.rand(n)

xDevice = deviceSpace.makeVector(x)
yDevice = deviceSpace.makeVector(y)
xHost = hostSpace.makeVector(x)
yHost = hostSpace.makeVector(y)


def doTest(function, message):
    with Timer('+GPU ' + message):
        for _ in range(iterations):
            function(xDevice, yDevice)

    with Timer('-CPU ' + message):
        for _ in range(iterations):
            function(xHost, yHost)

#Lincomb tests
doTest(lambda x,y: x.space.linComb(0,x,0,y), "y = 0")
doTest(lambda x,y: x.space.linComb(0,x,1,y), "y = y")
doTest(lambda x,y: x.space.linComb(0,x,2,y), "y = b*y")

doTest(lambda x,y: x.space.linComb(1,x,0,y), "y = x")
doTest(lambda x,y: x.space.linComb(1,x,1,y), "y += x")
doTest(lambda x,y: x.space.linComb(1,x,2,y), "y = x + b*y")

doTest(lambda x,y: x.space.linComb(2,x,0,y), "y = b*x")
doTest(lambda x,y: x.space.linComb(2,x,1,y), "y += b*x")
doTest(lambda x,y: x.space.linComb(2,x,2,y), "y = a*x + b*y")

#Test optimization for aliased vectors
doTest(lambda x,y: x.space.linComb(2,y,2,y), "y = (a+b)*y")

#Non lincomb tests
doTest(lambda x,y: x+y, "z = x + y")
doTest(lambda x,y: y.assign(x), "y.assign(x)")