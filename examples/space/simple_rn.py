"""An example of a very simple space, the space rn.

Including some benchmarks with an optimized version.
"""

import numpy as np
import odl
from odl.space.base_tensors import TensorSpace
from odl.util.testutils import timer


class SimpleRn(TensorSpace):
    """The real space R^n, non-optimized implmentation."""

    def __init__(self, size):
        super(SimpleRn, self).__init__(size, dtype='float64')

    def zero(self):
        return np.zeros(self.size)

    def one(self):
        return np.ones(self.size)

    def _lincomb(self, a, x1, b, x2, out):
        out[:] = a * x1 + b * x2

    def _inner(self, x1, x2):
        return float(np.vdot(x1, x2))

    def _multiply(self, x1, x2, out):
        out[:] = x1 * x2

    def _divide(self, x1, x2, out):
        out[:] = x1 / x2

    def __contains__(self, other):
        return (
            isinstance(other, np.ndarray)
            and other.shape == (self.size,)
            and other.dtype == 'float64'
        )

    def element(self, *args, **kwargs):
        if not args and not kwargs:
            return np.empty(self.size)
        if isinstance(args[0], np.ndarray):
            if args[0].shape == (self.size,):
                return np.asarray(args[0])
            else:
                raise ValueError(
                    'input array has shape {}, expected shape ({},)'
                    ''.format(args[0].shape, self.dim)
                )
        else:
            return np.array(*args, **kwargs).astype('float64', copy=False)


r5 = SimpleRn(5)
# odl.diagnostics.SpaceTest(r5).run_tests()

# Do some tests to compare
n = 10 ** 7
iterations = 10
cuda_supported = 'cuda' in odl.space.entry_points.tensor_space_impl_names()

# Perform some benchmarks with rn
opt_space = odl.rn(n)
simple_space = SimpleRn(n)

x, y, z = np.random.rand(n), np.random.rand(n), np.random.rand(n)
ox, oy, oz = (opt_space.element(a.copy()) for a in (x, y, z))
sx, sy, sz = (simple_space.element(a.copy()) for a in (x, y, z))

if cuda_supported:
    cu_space = odl.rn(n, impl='cuda')
    cx, cy, cz = (cu_space.element(a.copy()) for a in (x, y, z))

print(" lincomb:")
with timer("SimpleRn"):
    for _ in range(iterations):
        simple_space.lincomb(2.13, sx, 3.14, sy, out=sz)
print("result: {}".format(sz[1:5]))

with timer("odl numpy"):
    for _ in range(iterations):
        opt_space.lincomb(2.13, ox, 3.14, oy, out=oz)
print("result: {}".format(oz[1:5]))

if cuda_supported:
    with timer("odl cuda"):
        for _ in range(iterations):
            cu_space.lincomb(2.13, cx, 3.14, cy, out=cz)
    print("result: {}".format(cz[1:5]))


print("\n Norm:")
with timer("SimpleRn"):
    for _ in range(iterations):
        result = simple_space.norm(sz)
print("result: {}".format(result))

with timer("odl numpy"):
    for _ in range(iterations):
        result = opt_space.norm(oz)
print("result: {}".format(result))

if cuda_supported:
    with timer("odl cuda"):
        for _ in range(iterations):
            result = cu_space.norm(cz)
    print("result: {}".format(result))


print("\n Inner:")
with timer("SimpleRn"):
    for _ in range(iterations):
        result = simple_space.inner(sx, sz)
print("result: {}".format(result))

with timer("odl numpy"):
    for _ in range(iterations):
        result = opt_space.inner(ox, oz)
print("result: {}".format(result))

if cuda_supported:
    with timer("odl cuda"):
        for _ in range(iterations):
            result = cu_space.inner(cx, cz)
    print("result: {}".format(result))
