""" An example of a very simple space, the space Rn, as well as benchmarks
with an optimized version
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super
from future import standard_library

# External module imports
import numpy as np

# RL imports
from RL.space.space import *
from RL.space.cuda import *
from RL.space.set import *
from RL.utility.utility import errfmt
from RL.space.euclidean import EuclidRn
from RL.utility.testutils import Timer

standard_library.install_aliases()


class SimpleRn(HilbertSpace, Algebra):
    """The real space R^n, unoptimized implmentation
    """

    def __init__(self, dim):
        if not isinstance(n, Integral) or dim < 1:
            raise TypeError(errfmt('''
            dim ({}) has to be a positive integer'''.format(dim)))
        self._dim = dim
        self._field = RealNumbers()

    def _lincomb(self, z, a, x, b, y):
        z.data[:] = a * x.data + b * y.data

    def _inner(self, x, y):
        return float(np.vdot(x.data, y.data))

    def _multiply(self, x, y):
        y.data[:] = x.data * y.data

    def element(self, *args, **kwargs):
        if not args and not kwargs:
            return self.element(np.empty(self.dim))
        if isinstance(args[0], np.ndarray):
            if args[0].shape == (self.dim,):
                return SimpleRn.Vector(self, args[0])
            else:
                raise ValueError(errfmt('''
                Input numpy array ({}) is of shape {}, expected shape shape {}
                '''.format(args[0], args[0].shape, (self.dim,))))
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
        return isinstance(other, SimpleRn) and self.dim == other.dim

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


optX = EuclidRn(n)
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

with Timer("EuclidRn"):
    for _ in range(iterations):
        optX.lincomb(oz, 2.13, ox, 3.14, oy)
print("result: {}".format(oz[1:5]))

with Timer("CudaRN"):
    for _ in range(iterations):
        cuX.lincomb(cz, 2.13, cx, 3.14, cy)
print("result: {}".format(cz[1:5]))


print("\n Norm:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        result = sz.norm()
print("result: {}".format(result))

with Timer("EuclidRn"):
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

with Timer("EuclidRn"):
    for _ in range(iterations):
        result = oz.inner(ox)
print("result: {}".format(result))

with Timer("CudaRn"):
    for _ in range(iterations):
        result = cz.inner(cx)
print("result: {}".format(result))
