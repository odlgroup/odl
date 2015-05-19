""" An example of a very simple space, the space RN, as well as benchmarks with an optimized version
"""

# Imports for common Python 2/3 codebase
from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
try:
    from builtins import str, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, super
from future import standard_library

# External module imports
import numpy as np
from numpy import float64
from scipy.lib.blas import get_blas_funcs

# RL imports
from RL.space.space import *
from RL.space.set import *
from RL.utility.utility import errfmt
from RL.space.euclidean import EuclideanSpace
from RL.utility.testutils import Timer

standard_library.install_aliases()


class SimpleRN(HilbertSpace, Algebra):
    """The real space R^n, unoptimized implmentation
    """

    def __init__(self, n):
        if not isinstance(n, Integral) or n < 1:
            raise TypeError('n ({}) has to be a positive integer'.format(n))
        self._n = n
        self._field = RealNumbers()

    def linCombImpl(self, z, a, x, b, y):
        # Implement y = a*x + b*y using optimized BLAS rutines

        z.values[:] = a*x.values + b*y.values

    def innerImpl(self, x, y):
        return float(np.vdot(x.values, y.values))

    def multiplyImpl(self, x, y):
        y.values[:] = x.values*y.values

    def empty(self):
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
        return isinstance(other, SimpleRN) and self._n == other._n

    def makeVector(self, *args, **kwargs):
        if isinstance(args[0], np.ndarray):
            if args[0].shape == (self._n,):
                return SimpleRN.Vector(self, args[0])
            else:
                raise ValueError(errfmt('''
                Input numpy array ({}) is of shape {}, expected shape shape {}
                '''.format(args[0], args[0].shape, (self.n,))))
        else:
            return self.makeVector(np.array(*args,
                                            **kwargs).astype(float64,
                                                             copy=False))

    class Vector(HilbertSpace.Vector, Algebra.Vector):
        def __init__(self, space, values):
            super().__init__(space)
            self.values = values

        def __len__(self):
            return self.space._n

        def __getitem__(self, index):
            return self.values.__getitem__(index)

        def __setitem__(self, index, value):
            return self.values.__setitem__(index, value)


#Do some tests to compare
n = 10**7
iterations = 10

x = np.random.rand(n)
y = np.random.rand(n)
z = np.random.rand(n)

optX = EuclideanSpace(n)
simpleX = SimpleRN(n)

ox, oy, oz = optX.makeVector(x.copy()), optX.makeVector(y.copy()), optX.makeVector(z.copy())
sx, sy, sz = simpleX.makeVector(x.copy()), simpleX.makeVector(y.copy()), simpleX.makeVector(z.copy())


print(" Lincomb:")
with Timer("SimpleRN"):
    for _ in range(iterations):
        simpleX.linComb(sz, 2.13, sx, 3.14, sy)
print("result: {}".format(sz[1:5]))

with Timer("EuclideanSpace"):
    for _ in range(iterations):
        optX.linComb(oz, 2.13, ox, 3.14, oy)
print("result: {}".format(oz[1:5]))


print("\n Norm:")
with Timer("SimpleRN"):
    for _ in range(iterations):
        result = sz.norm()
print("result: {}".format(result))

with Timer("EuclideanSpace"):
    for _ in range(iterations):
        result = oz.norm()
print("result: {}".format(result))


print("\n Inner:")
with Timer("SimpleRN"):
    for _ in range(iterations):
        result = sz.inner(sx)
print("result: {}".format(result))

with Timer("EuclideanSpace"):
    for _ in range(iterations):
        result = oz.inner(ox)
print("result: {}".format(result))