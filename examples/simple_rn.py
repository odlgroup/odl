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

# ODL imports
from odl.sets.space import *
from odl.space.cuda import *
from odl.sets.set import *
from odl.space.cartesian import Rn
from odl.util.testutils import Timer


"""An example of a very simple space, the space Rn, as well as benchmarks
with an optimized version
"""


class SimpleRn(LinearSpace):
    """The real space R^n, unoptimized implmentation
    """

    def __init__(self, dim):
        if not isinstance(n, Integral) or dim < 1:
            raise TypeError('dimension {!r} not a positive integer.'
                            ''.format(dim))
        self._dim = dim
        self._field = RealNumbers()

    def _lincomb(self, z, a, x, b, y):
        z.data[:] = a * x.data + b * y.data

    def _inner(self, x, y):
        return float(np.vdot(x.data, y.data))

    def _multiply(self, z, x, y):
        z.data[:] = x.data * y.data

    def element(self, *args, **kwargs):
        if not args and not kwargs:
            return self.element(np.empty(self.dim))
        if isinstance(args[0], np.ndarray):
            if args[0].shape == (self.dim,):
                return SimpleRn.Vector(self, args[0])
            else:
                raise ValueError('input array {} is of shape {}, expected '
                                 'shape ({},).'.format(args[0], args[0].shape,
                                                       self.dim,))
        else:
            return self.element(np.array(
                *args, **kwargs).astype(np.float64, copy=False))
        return self.element(np.empty(self.dim, dtype=np.float64))

    @property
    def field(self):
        return self._field

    @property
    def dim(self):
        """ The number of dimensions of this space
        """
        return self._dim

    def equals(self, other):
        return type(self) == type(other) and self.dim == other.dim

    class Vector(HilbertSpace.Vector, Algebra.Vector):
        def __init__(self, space, data):
            super().__init__(space)
            self.data = data

        def __len__(self):
            return self.space._n

        def __getitem__(self, index):
            return self.data.__getitem__(index)

        def __setitem__(self, index, value):
            return self.data.__setitem__(index, value)


# Do some tests to compare
n = 10**7
iterations = 10


optX = Rn(n)
simpleX = SimpleRn(n)
cuX = CudaRn(n)

x, y, z = np.random.rand(n), np.random.rand(n), np.random.rand(n)
ox, oy, oz = (optX.element(x.copy()), optX.element(y.copy()),
              optX.element(z.copy()))
sx, sy, sz = (simpleX.element(x.copy()), simpleX.element(y.copy()),
              simpleX.element(z.copy()))
cx, cy, cz = (cuX.element(x.copy()), cuX.element(y.copy()),
              cuX.element(z.copy()))


print(" lincomb:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        simpleX.lincomb(sz, 2.13, sx, 3.14, sy)
print("result: {}".format(sz[1:5]))

with Timer("Rn"):
    for _ in range(iterations):
        optX.lincomb(oz, 2.13, ox, 3.14, oy)
print("result: {}".format(oz[1:5]))

with Timer("CudaRn"):
    for _ in range(iterations):
        cuX.lincomb(cz, 2.13, cx, 3.14, cy)
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

with Timer("CudaRn"):
    for _ in range(iterations):
        result = cz.inner(cx)
print("result: {}".format(result))
