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

"""Speed comparison between CPU and GPU spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range

import numpy as np
import odl
from odl.util.testutils import Timer

n = 10**7
iterations = 100
device_space = odl.CudaRn(n)
host_space = odl.Rn(n)
x = np.random.rand(n)
y = np.random.rand(n)
z = np.empty(n)

x_dev = device_space.element(x)
y_dev = device_space.element(y)
z_dev = device_space.element(z)
x_host = host_space.element(x)
y_host = host_space.element(y)
z_host = host_space.element(z)


def run_test(function, message):
    with Timer('+GPU ' + message):
        for _ in range(iterations):
            function(z_dev, x_dev, y_dev)

    with Timer('-CPU ' + message):
        for _ in range(iterations):
            function(z_host, x_host, y_host)

# lincomb tests
run_test(lambda z, x, y: x.space.lincomb(0, x, 0, y, z), "z = 0")
run_test(lambda z, x, y: x.space.lincomb(0, x, 1, y, z), "z = y")
run_test(lambda z, x, y: x.space.lincomb(0, x, 2, y, z), "z = b*y")

run_test(lambda z, x, y: x.space.lincomb(1, x, 0, y, z), "z = x")
run_test(lambda z, x, y: x.space.lincomb(1, x, 1, y, z), "z = x + y")
run_test(lambda z, x, y: x.space.lincomb(1, x, 2, y, z), "z = x + b*y")

run_test(lambda z, x, y: x.space.lincomb(2, x, 0, y, z), "z = a*x")
run_test(lambda z, x, y: x.space.lincomb(2, x, 1, y, z), "z = a*x + y")
run_test(lambda z, x, y: x.space.lincomb(2, x, 2, y, z), "z = a*x + b*y")

# Test optimization for 1 aliased vector
run_test(lambda z, x, y: x.space.lincomb(1, z, 0, y, z), "z = z")
run_test(lambda z, x, y: x.space.lincomb(1, z, 1, y, z), "z += y")
run_test(lambda z, x, y: x.space.lincomb(1, z, 2, y, z), "z += b*y")

run_test(lambda z, x, y: x.space.lincomb(2, z, 0, y, z), "z = a*z")
run_test(lambda z, x, y: x.space.lincomb(2, z, 1, y, z), "z = a*z + y")
run_test(lambda z, x, y: x.space.lincomb(2, z, 2, y, z), "z = a*z + b*y")

# Test optimization for 2 aliased vectors
run_test(lambda z, x, y: x.space.lincomb(2, z, 2, z, z), "z = (a+b)*z")

# Non lincomb tests
run_test(lambda z, x, y: x+y, "z = x + y")
run_test(lambda z, x, y: y.assign(x), "y.assign(x)")
