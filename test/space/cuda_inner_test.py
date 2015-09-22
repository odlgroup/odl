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
from builtins import range

# External module imports
import unittest
import numpy as np

# ODL
import odl
from odl.util.testutils import ODLTestCase


@unittest.skipIf(not odl.CUDA_AVAILABLE, "cuda not available")
class CudaConstantWeightedInnerTest(ODLTestCase):
    @staticmethod
    def _vectors(fn):
        # Generate numpy vectors, real or complex
        if isinstance(fn, odl.CudaRn):
            xarr = np.random.rand(fn.size)
            yarr = np.random.rand(fn.size)
        else:
            xarr = np.random.rand(fn.size) + 1j * np.random.rand(fn.size)
            yarr = np.random.rand(fn.size) + 1j * np.random.rand(fn.size)

        # Make CudaFn vectors
        x = fn.element(xarr)
        y = fn.element(yarr)
        return xarr, yarr, x, y

    def test_init(self):
        rn = odl.CudaRn(10)
        constant = 1.5

        # Just test if the code runs
        inner = odl.CudaConstantWeightedInner(constant)

    def test_equals(self):
        constant = 1.5

        inner_const = odl.CudaConstantWeightedInner(constant)
        inner_const2 = odl.CudaConstantWeightedInner(constant)
        inner_const_npy = odl.ConstantWeightedInner(constant)

        self.assertEquals(inner_const, inner_const)
        self.assertEquals(inner_const, inner_const2)
        self.assertEquals(inner_const2, inner_const)

        self.assertNotEquals(inner_const, inner_const_npy)

    def _test_call_real(self, n):
        rn = odl.CudaRn(n)
        xarr, yarr, x, y = self._vectors(rn)

        constant = 1.5
        inner_const = odl.CudaConstantWeightedInner(constant)

        result_const = inner_const(x.data, y.data)
        true_result_const = constant * np.dot(yarr, xarr)

        self.assertAlmostEquals(result_const, true_result_const, places=5)

    def test_call(self):
        for _ in range(20):
            self._test_call_real(10)

    def test_repr(self):
        constant = 1.5
        inner_const = odl.CudaConstantWeightedInner(constant)

        repr_str = 'CudaConstantWeightedInner(1.5)'
        self.assertEquals(repr(inner_const), repr_str)

    def test_str(self):
        constant = 1.5
        inner_const = odl.CudaConstantWeightedInner(constant)

        print_str = '(x, y) --> 1.5 * y^H x'
        self.assertEquals(str(inner_const), print_str)


if __name__ == '__main__':
    unittest.main(exit=False)
