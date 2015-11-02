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
from builtins import super

# External module imports
import numpy as np
from numbers import Integral

# ODL imports
import odl
from odl.util.testutils import Timer


"""An example of a very simple space, the space Rn, as well as benchmarks
with an optimized version.
"""


class SimpleRn(odl.space.base_ntuples.FnBase):
    """The real space R^n, non-optimized implmentation."""

    def __init__(self, size):
        super().__init__(size, np.float)

    def zero(self):
        return self.element(np.zeros(self.size))

    def one(self):
        return self.element(np.ones(self.size))

    def _lincomb(self, a, x1, b, x2, out):
        out.data[:] = a * x1.data + b * x2.data

    def _inner(self, x1, x2):
        return float(np.vdot(x1.data, x2.data))

    def _multiply(self, x1, x2, out):
        out.data[:] = x1.data * x2.data

    def _divide(self, x1, x2, out):
        out.data[:] = x1.data / x2.data

    def element(self, *args, **kwargs):
        if not args and not kwargs:
            return self.element(np.empty(self.size))
        if isinstance(args[0], np.ndarray):
            if args[0].shape == (self.size,):
                return SimpleRn.Vector(self, args[0])
            else:
                raise ValueError('input array {} is of shape {}, expected '
                                 'shape ({},).'.format(args[0], args[0].shape,
                                                       self.dim,))
        else:
            return self.element(np.array(
                *args, **kwargs).astype(np.float64, copy=False))
        return self.element(np.empty(self.dim, dtype=np.float64))

    class Vector(odl.space.base_ntuples.FnBase.Vector):
        def __init__(self, space, data):
            super().__init__(space)
            self.data = data

        def __getitem__(self, index):
            return self.data.__getitem__(index)

        def __setitem__(self, index, value):
            return self.data.__setitem__(index, value)

        def asarray(self, *args):
            return self.data(*args)

r5 = SimpleRn(5)
odl.diagnostics.SpaceTest(r5).run_tests()

# Do some tests to compare
n = 10**7
iterations = 10

#Perform some benchmarks with Rn adn CudaRn

optX = odl.Rn(n)
simpleX = SimpleRn(n)

x, y, z = np.random.rand(n), np.random.rand(n), np.random.rand(n)
ox, oy, oz = (optX.element(x.copy()), optX.element(y.copy()),
              optX.element(z.copy()))
sx, sy, sz = (simpleX.element(x.copy()), simpleX.element(y.copy()),
              simpleX.element(z.copy()))
if odl.CUDA_AVAILABLE:
    cuX = odl.CudaRn(n)
    cx, cy, cz = (cuX.element(x.copy()), cuX.element(y.copy()),
                  cuX.element(z.copy()))

print(" lincomb:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        simpleX.lincomb(2.13, sx, 3.14, sy, out=sz)
print("result: {}".format(sz[1:5]))

with Timer("Rn"):
    for _ in range(iterations):
        optX.lincomb(2.13, ox, 3.14, oy, out=oz)
print("result: {}".format(oz[1:5]))

if odl.CUDA_AVAILABLE:
    with Timer("CudaRn"):
        for _ in range(iterations):
            cuX.lincomb(2.13, cx, 3.14, cy, out=cz)
    print("result: {}".format(cz[1:5]))


print("\n Norm:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        result = sz.norm()
print("result: {}".format(result))

with Timer("Rn"):
    for _ in range(iterations):
        result = oz.norm()
print("result: {}".format(result))

if odl.CUDA_AVAILABLE:
    with Timer("CudaRn"):
        for _ in range(iterations):
            result = cz.norm()
    print("result: {}".format(result))


print("\n Inner:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        result = sz.inner(sx)
print("result: {}".format(result))

with Timer("Rn"):
    for _ in range(iterations):
        result = oz.inner(ox)
print("result: {}".format(result))

if odl.CUDA_AVAILABLE:
    with Timer("CudaRn"):
        for _ in range(iterations):
            result = cz.inner(cx)
    print("result: {}".format(result))
