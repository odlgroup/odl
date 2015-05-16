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
from RL.operator.operator import *
from RL.space.space import *
from RL.space.euclidean import *
from RL.space.cuda import *
from RL.utility.testutils import Timer

n=10**7
iterations = 100
deviceSpace = CudaRN(n)
hostSpace = RN(n)
x = np.random.rand(n)
y = np.random.rand(n)
z = np.empty(n)

xDevice = deviceSpace.element(x)
yDevice = deviceSpace.element(y)
zDevice = deviceSpace.element(z)
xHost = hostSpace.element(x)
yHost = hostSpace.element(y)
zHost = hostSpace.element(z)


def doTest(function, message):
    with Timer('+GPU ' + message):
        for _ in range(iterations):
            function(zDevice, xDevice, yDevice)

    with Timer('-CPU ' + message):
        for _ in range(iterations):
            function(zHost, xHost, yHost)

#Lincomb tests
doTest(lambda z,x,y: x.space.linComb(z,0,x,0,y), "z = 0")
doTest(lambda z,x,y: x.space.linComb(z,0,x,1,y), "z = y")
doTest(lambda z,x,y: x.space.linComb(z,0,x,2,y), "z = b*y")

doTest(lambda z,x,y: x.space.linComb(z,1,x,0,y), "z = x")
doTest(lambda z,x,y: x.space.linComb(z,1,x,1,y), "z = x + y")
doTest(lambda z,x,y: x.space.linComb(z,1,x,2,y), "z = x + b*y")

doTest(lambda z,x,y: x.space.linComb(z,2,x,0,y), "z = a*x")
doTest(lambda z,x,y: x.space.linComb(z,2,x,1,y), "z = a*x + y")
doTest(lambda z,x,y: x.space.linComb(z,2,x,2,y), "z = a*x + b*y")

#Test optimization for 1 aliased vector
doTest(lambda z,x,y: x.space.linComb(z,1,z,0,y), "z = z")
doTest(lambda z,x,y: x.space.linComb(z,1,z,1,y), "z += y")
doTest(lambda z,x,y: x.space.linComb(z,1,z,2,y), "z += b*y")

doTest(lambda z,x,y: x.space.linComb(z,2,z,0,y), "z = a*z")
doTest(lambda z,x,y: x.space.linComb(z,2,z,1,y), "z = a*z + y")
doTest(lambda z,x,y: x.space.linComb(z,2,z,2,y), "z = a*z + b*y")

#Test optimization for 2 aliased vectors
doTest(lambda z,x,y: x.space.linComb(z,2,z,2,z), "z = (a+b)*z")

#Non lincomb tests
doTest(lambda z,x,y: x+y, "z = x + y")
doTest(lambda z,x,y: y.assign(x), "y.assign(x)")