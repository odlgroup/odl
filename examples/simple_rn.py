""" An example of a very simple space, the space Rn, as well as benchmarks with an optimized version
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super
from future import standard_library

# External module imports
import numpy as np
from numpy import float64

# RL imports
from RL.space.space import *
from RL.space.set import *
from RL.utility.utility import errfmt
from RL.space.euclidean import EuclidRn
from RL.utility.testutils import Timer

standard_library.install_aliases()


class SimpleRn(HilbertSpace, Algebra):
    """The real space R^n, unoptimized implmentation
    """

    def __init__(self, n):
        if not isinstance(n, Integral) or n < 1:
            raise TypeError('n ({}) has to be a positive integer'.format(n))
        self._n = n
        self._field = RealNumbers()

    def _lincomb(self, z, a, x, b, y):
        # Implement y = a*x + b*y using optimized BLAS rutines
        z.data[:] = a * x.data + b * y.data

    def _inner(self, x, y):
        return float(np.vdot(x.data, y.data))

    def _multiply(self, x, y):
        y.data[:] = x.data * y.data

    def element(self, *args, **kwargs):
        if not args and not kwargs:
            return self.element(np.empty(self.n))
        if isinstance(args[0], np.ndarray):
            if args[0].shape == (self._n,):
                return SimpleRn.Vector(self, args[0])
            else:
                raise ValueError(errfmt('''
                Input numpy array ({}) is of shape {}, expected shape shape {}
                '''.format(args[0], args[0].shape, (self.n,))))
        else:
            return self.makeVector(np.array(*args,
                                            **kwargs).astype(float64,
                                                             copy=False))
        return self.makeVector(np.empty(self._n, dtype=float64))

    @property
    def field(self):
        return self._field

    @property
    def n(self):
        """ The number of dimensions of this space
        """
        return self._n

    def equals(self, other):
        return isinstance(other, SimpleRn) and self._n == other._n

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

x, y, z = np.random.rand(n), np.random.rand(n), np.random.rand(n)
ox, oy, oz = (optX.element(x.copy()), optX.element(y.copy()),
              optX.element(z.copy()))
sx, sy, sz = (simpleX.element(x.copy()), simpleX.element(y.copy()),
              simpleX.element(z.copy()))


print(" lincomb:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        simpleX.lincomb(sz, 2.13, sx, 3.14, sy)
print("result: {}".format(sz[1:5]))

with Timer("EuclidRn"):
    for _ in range(iterations):
        optX.lincomb(oz, 2.13, ox, 3.14, oy)
print("result: {}".format(oz[1:5]))


print("\n Norm:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        result = sz.norm()
print("result: {}".format(result))

with Timer("EuclidRn"):
    for _ in range(iterations):
        result = oz.norm()
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
