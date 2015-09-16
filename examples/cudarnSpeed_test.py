# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from odl.operator.operator import *
from odl.sets.space import *
from odl.space.cartesian import *
from odl.space.cuda import *
from odl.util.testutils import Timer

n = 10**7
iterations = 100
deviceSpace = CudaRn(n)
hostSpace = Rn(n)
x = np.random.rand(n)
y = np.random.rand(n)
z = np.empty(n)

xDevice = deviceSpace.element(x)
yDevice = deviceSpace.element(y)
zDevice = deviceSpace.element(z)
xHost = hostSpace.element(x)
yHost = hostSpace.element(y)
zHost = hostSpace.element(z)


def run_test(function, message):
    with Timer('+GPU ' + message):
        for _ in range(iterations):
            function(zDevice, xDevice, yDevice)

    with Timer('-CPU ' + message):
        for _ in range(iterations):
            function(zHost, xHost, yHost)

# lincomb tests
run_test(lambda z, x, y: x.space.lincomb(z, 0, x, 0, y), "z = 0")
run_test(lambda z, x, y: x.space.lincomb(z, 0, x, 1, y), "z = y")
run_test(lambda z, x, y: x.space.lincomb(z, 0, x, 2, y), "z = b*y")

run_test(lambda z, x, y: x.space.lincomb(z, 1, x, 0, y), "z = x")
run_test(lambda z, x, y: x.space.lincomb(z, 1, x, 1, y), "z = x + y")
run_test(lambda z, x, y: x.space.lincomb(z, 1, x, 2, y), "z = x + b*y")

run_test(lambda z, x, y: x.space.lincomb(z, 2, x, 0, y), "z = a*x")
run_test(lambda z, x, y: x.space.lincomb(z, 2, x, 1, y), "z = a*x + y")
run_test(lambda z, x, y: x.space.lincomb(z, 2, x, 2, y), "z = a*x + b*y")

# Test optimization for 1 aliased vector
run_test(lambda z, x, y: x.space.lincomb(z, 1, z, 0, y), "z = z")
run_test(lambda z, x, y: x.space.lincomb(z, 1, z, 1, y), "z += y")
run_test(lambda z, x, y: x.space.lincomb(z, 1, z, 2, y), "z += b*y")

run_test(lambda z, x, y: x.space.lincomb(z, 2, z, 0, y), "z = a*z")
run_test(lambda z, x, y: x.space.lincomb(z, 2, z, 1, y), "z = a*z + y")
run_test(lambda z, x, y: x.space.lincomb(z, 2, z, 2, y), "z = a*z + b*y")

# Test optimization for 2 aliased vectors
run_test(lambda z, x, y: x.space.lincomb(z, 2, z, 2, z), "z = (a+b)*z")

# Non lincomb tests
run_test(lambda z, x, y: x+y, "z = x + y")
run_test(lambda z, x, y: y.assign(x), "y.assign(x)")
