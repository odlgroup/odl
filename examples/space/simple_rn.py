"""An example of a very simple space, the space rn.

Including some benchmarks with an optimized version.
"""

import numpy as np
import odl
from odl.space.base_tensors import TensorSpace, Tensor
from odl.util.testutils import Timer


class SimpleRn(TensorSpace):
    """The real space R^n, non-optimized implmentation."""

    def __init__(self, size):
        TensorSpace.__init__(self, size, np.float)

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
                return RnVector(self, args[0])
            else:
                raise ValueError('input array {} is of shape {}, expected '
                                 'shape ({},).'.format(args[0], args[0].shape,
                                                       self.dim,))
        else:
            return self.element(np.array(
                *args, **kwargs).astype(np.float64, copy=False))
        return self.element(np.empty(self.dim, dtype=np.float64))


class RnVector(Tensor):
    def __init__(self, space, data):
        Tensor.__init__(self, space)
        self.data = data

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def __setitem__(self, index, value):
        return self.data.__setitem__(index, value)

    def asarray(self, *args):
        return self.data(*args)


r5 = SimpleRn(5)
# odl.diagnostics.SpaceTest(r5).run_tests()

# Do some tests to compare
n = 10**7
iterations = 10

# Perform some benchmarks with rn
opt_spc = odl.rn(n)
simple_spc = SimpleRn(n)

x, y, z = np.random.rand(n), np.random.rand(n), np.random.rand(n)
ox, oy, oz = (opt_spc.element(x.copy()), opt_spc.element(y.copy()),
              opt_spc.element(z.copy()))
sx, sy, sz = (simple_spc.element(x.copy()), simple_spc.element(y.copy()),
              simple_spc.element(z.copy()))
if 'cuda' in odl.space.entry_points.TENSOR_SPACE_IMPLS:
    cu_spc = odl.rn(n, impl='cuda')
    cx, cy, cz = (cu_spc.element(x.copy()), cu_spc.element(y.copy()),
                  cu_spc.element(z.copy()))

print(" lincomb:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        simple_spc.lincomb(2.13, sx, 3.14, sy, out=sz)
print("result: {}".format(sz[1:5]))

with Timer("odl numpy"):
    for _ in range(iterations):
        opt_spc.lincomb(2.13, ox, 3.14, oy, out=oz)
print("result: {}".format(oz[1:5]))

if 'cuda' in odl.space.entry_points.TENSOR_SPACE_IMPLS:
    with Timer("odl cuda"):
        for _ in range(iterations):
            cu_spc.lincomb(2.13, cx, 3.14, cy, out=cz)
    print("result: {}".format(cz[1:5]))


print("\n Norm:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        result = sz.norm()
print("result: {}".format(result))

with Timer("odl numpy"):
    for _ in range(iterations):
        result = oz.norm()
print("result: {}".format(result))

if 'cuda' in odl.space.entry_points.TENSOR_SPACE_IMPLS:
    with Timer("odl cuda"):
        for _ in range(iterations):
            result = cz.norm()
    print("result: {}".format(result))


print("\n Inner:")
with Timer("SimpleRn"):
    for _ in range(iterations):
        result = sz.inner(sx)
print("result: {}".format(result))

with Timer("odl numpy"):
    for _ in range(iterations):
        result = oz.inner(ox)
print("result: {}".format(result))

if 'cuda' in odl.space.entry_points.TENSOR_SPACE_IMPLS:
    with Timer("odl cuda"):
        for _ in range(iterations):
            result = cz.inner(cx)
    print("result: {}".format(result))
