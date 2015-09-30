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

# External module imports
import unittest
import math
import numpy as np
from numpy import float64

# ODL imports
import odl
from odl.util.testutils import ODLTestCase


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available')
class TestInit(ODLTestCase):
    def test_empty(self):
        r3 = odl.CudaRn(3)
        x = r3.element()
        self.assertEqual(x, odl.CudaRn(3).element())
        # Nothing to test, simply check that code runs

    def test_zero(self):
        r3 = odl.CudaRn(3)
        self.assertAllAlmostEquals(r3.zero(), [0, 0, 0])

    def test_list_init(self):
        r3 = odl.CudaRn(3)
        x = r3.element([1, 2, 3])
        self.assertAllAlmostEquals(x, [1, 2, 3])

    def test_ndarray_init(self):
        r3 = odl.CudaRn(3)

        x0 = np.array([1., 2., 3.])
        x = r3.element(x0)
        self.assertAllAlmostEquals(x, x0)

        x0 = np.array([1, 2, 3], dtype=float64)
        x = r3.element(x0)
        self.assertAllAlmostEquals(x, x0)

        x0 = np.array([1, 2, 3], dtype=int)
        x = r3.element(x0)
        self.assertAllAlmostEquals(x, x0)


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available')
class TestAccessors(ODLTestCase):
    def test_getitem(self):
        r3 = odl.CudaRn(3)
        y = [1, 2, 3]
        x = r3.element(y)

        for index in [0, 1, 2, -1, -2, -3]:
            self.assertAlmostEquals(x[index], y[index])

    def test_iterator(self):
        r3 = odl.CudaRn(3)
        y = [1, 2, 3]
        x = r3.element(y)

        self.assertAlmostEquals([a for a in x], [b for b in y])

    def test_getitem_index_error(self):
        r3 = odl.CudaRn(3)
        x = r3.element([1, 2, 3])

        with self.assertRaises(IndexError):
            x[-4]

        with self.assertRaises(IndexError):
            x[3]

    def test_setitem(self):
        r3 = odl.CudaRn(3)
        x = r3.element([42, 42, 42])

        for index in [0, 1, 2, -1, -2, -3]:
            x[index] = index
            self.assertAlmostEquals(x[index], index)

    def test_setitem_index_error(self):
        r3 = odl.CudaRn(3)
        x = r3.element([1, 2, 3])

        with self.assertRaises(IndexError):
            x[-4] = 0

        with self.assertRaises(IndexError):
            x[3] = 0

    def _test_getslice(self, slice):
        # Validate get against python list behaviour
        r6 = odl.CudaRn(6)
        y = [0, 1, 2, 3, 4, 5]
        x = r6.element(y)

        self.assertAllAlmostEquals(x[slice], y[slice])

    def test_getslice(self):
        # Tests getting all combinations of slices
        steps = [None, -2, -1, 1, 2]
        starts = [None, -1, -3, 0, 2, 5]
        ends = [None, -1, -3, 0, 2, 5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self._test_getslice(slice(start, end, step))

    def test_slice_of_slice(self):
        # Verify that creating slices from slices works as expected
        r10 = odl.CudaRn(10)
        xh = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        xd = r10.element(xh)

        yh = xh[1:8:2]
        yd = xd[1:8:2]

        self.assertAllAlmostEquals(yh, yd)

        zh = yh[1::2]
        zd = yd[1::2]

        self.assertAllAlmostEquals(zh, zd)

    def test_slice_is_view(self):
        # Verify that modifications of a view modify the original data
        r10 = odl.CudaRn(10)
        xh = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        xd = r10.element(xh)

        yh = xh[1:8:2]
        yh[:] = [1, 3, 5, 7]

        yd = xd[1:8:2]
        yd[:] = [1, 3, 5, 7]

        self.assertAllAlmostEquals(xh, xd)
        self.assertAllAlmostEquals(yh, yd)

    def test_getslice_index_error(self):
        r3 = odl.CudaRn(3)
        xd = r3.element([1, 2, 3])

        # Bad slice
        with self.assertRaises(IndexError):
            xd[10:13]

    def _test_setslice(self, slice):
        # Validate set against python list behaviour
        r6 = odl.CudaRn(6)
        z = [7, 8, 9, 10, 11, 10]
        y = [0, 1, 2, 3, 4, 5]
        x = r6.element(y)

        x[slice] = z[slice]
        y[slice] = z[slice]
        self.assertAllAlmostEquals(x, y)

    def test_setslice(self):
        # Tests a range of combination of slices
        steps = [None, -2, -1, 1, 2]
        starts = [None, -1, -3, 0, 2, 5]
        ends = [None, -1, -3, 0, 2, 5]

        for start in starts:
            for end in ends:
                for step in steps:
                    self._test_setslice(slice(start, end, step))

    def test_setslice_index_error(self):
        r3 = odl.CudaRn(3)
        xd = r3.element([1, 2, 3])

        # Bad slice
        with self.assertRaises(IndexError):
            xd[10:13] = [1, 2, 3]

        # Bad size of rhs
        with self.assertRaises(IndexError):
            xd[:] = []

        with self.assertRaises(IndexError):
            xd[:] = [1, 2]

        with self.assertRaises(IndexError):
            xd[:] = [1, 2, 3, 4]


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available')
class TestMethods(ODLTestCase):
    def test_norm(self):
        r3 = odl.CudaRn(3)
        xd = r3.element([1, 2, 3])

        correct_norm_squared = 1 ** 2 + 2 ** 2 + 3 ** 2
        correct_norm = math.sqrt(correct_norm_squared)

        # Space function
        self.assertAlmostEquals(r3.norm(xd), correct_norm, places=5)

    def test_inner(self):
        r3 = odl.CudaRn(3)
        xd = r3.element([1, 2, 3])
        yd = r3.element([5, 3, 9])

        correct_inner = 1 * 5 + 2 * 3 + 3 * 9

        # Space function
        self.assertAlmostEquals(r3.inner(xd, yd), correct_inner)

    def vectors(self, rn):
        # Generate numpy arrays
        x_arr = np.random.rand(rn.size)
        y_arr = np.random.rand(rn.size)
        z_arr = np.random.rand(rn.size)

        # Make rn vectors
        x, y, z = rn.element(x_arr), rn.element(y_arr), rn.element(z_arr)

        return x_arr, y_arr, z_arr, x, y, z

    def _test_lincomb(self, a, b, n=100):
        # Validates lincomb against the result on host with randomized
        # data and given a,b
        rn = odl.CudaRn(n)

        # Unaliased arguments
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * x_arr + b * y_arr
        rn.lincomb(z, a, x, b, y)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

        # First argument aliased with output
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * z_arr + b * y_arr
        rn.lincomb(z, a, z, b, y)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

        # Second argument aliased with output
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * x_arr + b * z_arr
        rn.lincomb(z, a, x, b, z)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

        # Both arguments aliased with each other
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * x_arr + b * x_arr
        rn.lincomb(z, a, x, b, x)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

        # All aliased
        x_arr, y_arr, z_arr, x, y, z = self.vectors(rn)

        z_arr[:] = a * z_arr + b * z_arr
        rn.lincomb(z, a, z, b, z)
        self.assertAllAlmostEquals([x, y, z], [x_arr, y_arr, z_arr], places=4)

    def test_lincomb(self):
        scalar_values = [0, 1, -1, 3.41]
        for a in scalar_values:
            for b in scalar_values:
                self._test_lincomb(a, b)

    def _test_member_lincomb(self, a, n=100):
        # Validates vector member lincomb against the result on host with
        # randomized data
        n = 100

        # Generate vectors
        y_host = np.random.rand(n)
        x_host = np.random.rand(n)

        r3 = odl.CudaRn(n)
        y_device = r3.element(y_host)
        x_device = r3.element(x_host)

        # Host side calculation
        y_host[:] = a * x_host

        # Device side calculation
        y_device.lincomb(a, x_device)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(y_device, y_host, places=5)

    def test_member_lincomb(self):
        scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
        for a in scalar_values:
            self._test_member_lincomb(a)

    def test_multiply(self):
        # Validates multiply against the result on host with randomized data
        n = 100
        x_host = np.random.rand(n)
        y_host = np.random.rand(n)
        z_host = np.empty(n)

        r3 = odl.CudaRn(n)
        x_device = r3.element(x_host)
        y_device = r3.element(y_host)
        z_device = r3.element()

        # Host side calculation
        z_host[:] = x_host * y_host

        # Device side calculation
        r3.multiply(z_device, x_device, y_device)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(z_device, z_host, places=5)

        # Assert input was not modified
        self.assertAllAlmostEquals(x_device, x_host, places=5)
        self.assertAllAlmostEquals(y_device, y_host, places=5)

        # Aliased
        z_host[:] = z_host * x_host
        r3.multiply(z_device, z_device, x_device)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(z_device, z_host, places=5)

        # Aliased
        z_host[:] = z_host * z_host
        r3.multiply(z_device, z_device, z_device)

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(z_device, z_host, places=5)

    def test_member_multiply(self):
        # Validates vector member multiply against the result on host
        # with randomized data
        n = 100
        y_host = np.random.rand(n)
        x_host = np.random.rand(n)

        r3 = odl.CudaRn(n)
        y_device = r3.element(y_host)
        x_device = r3.element(x_host)

        # Host side calculation
        y_host[:] = x_host * y_host

        # Device side calculation
        y_device *= x_device

        # Cuda only uses floats, so require 5 places
        self.assertAllAlmostEquals(y_device, y_host, places=5)


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available')
class TestConvenience(ODLTestCase):
    def test_addition(self):
        r3 = odl.CudaRn(3)
        xd = r3.element([1, 2, 3])
        yd = r3.element([5, 3, 7])

        self.assertAllAlmostEquals(xd + yd, [6, 5, 10])

    def test_scalar_mult(self):
        r3 = odl.CudaRn(3)
        xd = r3.element([1, 2, 3])
        C = 5

        self.assertAllAlmostEquals(C * xd, [5, 10, 15])

    def test_incompatible_operations(self):
        r3 = odl.CudaRn(3)
        R3h = odl.Rn(3)
        xA = r3.zero()
        xB = R3h.zero()

        with self.assertRaises(TypeError):
            xA += xB

        with self.assertRaises(TypeError):
            xA -= xB

        with self.assertRaises(TypeError):
            z = xA + xB

        with self.assertRaises(TypeError):
            z = xA - xB


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available')
class TestPointer(ODLTestCase):
    def test_modify(self):
        r3 = odl.CudaRn(3)
        xd = r3.element([1, 2, 3])
        yd = r3.element(data_ptr=xd.data_ptr)

        yd[:] = [5, 6, 7]

        self.assertAllEquals(xd, yd)

    def test_sub_vector(self):
        r6 = odl.CudaRn(6)
        r3 = odl.CudaRn(3)
        xd = r6.element([1, 2, 3, 4, 5, 6])

        yd = r3.element(data_ptr=xd.data_ptr)
        yd[:] = [7, 8, 9]

        self.assertAllEquals([7, 8, 9, 4, 5, 6], xd)

    def test_offset_sub_vector(self):
        r6 = odl.CudaRn(6)
        r3 = odl.CudaRn(3)
        xd = r6.element([1, 2, 3, 4, 5, 6])

        yd = r3.element(data_ptr=xd.data_ptr+3*xd.space.dtype.itemsize)
        yd[:] = [7, 8, 9]

        self.assertAllEquals([1, 2, 3, 7, 8, 9], xd)


@unittest.skipIf(not odl.CUDA_AVAILABLE, "CUDA not available")
class TestDType(ODLTestCase):
    # Simple tests for the various dtypes

    @unittest.skipIf(np.int8 not in odl.CUDA_DTYPES, "int8 not available")
    def do_test_int8(self):
        r3 = odl.CudaFn(3, np.int8)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.int16 not in odl.CUDA_DTYPES, "int16 not available")
    def do_test_int16(self):
        r3 = odl.CudaFn(3, np.int16)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.int32 not in odl.CUDA_DTYPES, "int32 not available")
    def do_test_int32(self):
        r3 = odl.CudaFn(3, np.int32)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.int64 not in odl.CUDA_DTYPES, "int64 not available")
    def do_test_int64(self):
        r3 = odl.CudaFn(3, np.int64)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.uint8 not in odl.CUDA_DTYPES, "uint8 not available")
    def do_test_uint8(self):
        r3 = odl.CudaFn(3, np.uint8)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.uint16 not in odl.CUDA_DTYPES, "uint16 not available")
    def do_test_uint16(self):
        r3 = odl.CudaFn(3, np.uint16)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.uint32 not in odl.CUDA_DTYPES, "uint32 not available")
    def do_test_uint32(self):
        r3 = odl.CudaFn(3, np.uint32)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.uint64 not in odl.CUDA_DTYPES, "uint64 not available")
    def do_test_uint64(self):
        r3 = odl.CudaFn(3, np.uint64)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.float32 not in odl.CUDA_DTYPES,
                     "float32 not available")
    def do_test_float32(self):
        r3 = odl.CudaFn(3, np.float32)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.float64 not in odl.CUDA_DTYPES,
                     "float64 not available")
    def do_test_float64(self):
        r3 = odl.CudaFn(3, np.float64)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.float not in odl.CUDA_DTYPES, "float not available")
    def do_test_float(self):
        r3 = odl.CudaFn(3, np.float)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])

    @unittest.skipIf(np.int not in odl.CUDA_DTYPES, "int not available")
    def do_test_int(self):
        r3 = odl.CudaFn(3, np.int)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        self.assertAllEquals(z, [5, 7, 9])


@unittest.skipIf(not odl.CUDA_AVAILABLE, "CUDA not available")
class TestUFunc(ODLTestCase):
    #Simple tests for the various dtypes

    def test_sin(self):
        r3 = odl.CudaRn(3)
        x_host = [0.1, 0.3, 10.0]
        y_host = np.sin(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.sin(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)

    def test_cos(self):
        r3 = odl.CudaRn(3)
        x_host = [0.1, 0.3, 10.0]
        y_host = np.cos(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.cos(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)

    def test_arcsin(self):
        r3 = odl.CudaRn(3)
        x_host = [0.1, 0.3, 0.5]
        y_host = np.arcsin(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.arcsin(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)

    def test_arccos(self):
        r3 = odl.CudaRn(3)
        x_host = [0.1, 0.3, 0.5]
        y_host = np.arccos(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.arccos(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)

    def test_log(self):
        r3 = odl.CudaRn(3)
        x_host = [0.1, 0.3, 0.5]
        y_host = np.log(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.log(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)

    def test_exp(self):
        r3 = odl.CudaRn(3)
        x_host = [-1.0, 0.0, 1.0]
        y_host = np.exp(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.exp(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)

    def test_abs(self):
        r3 = odl.CudaRn(3)
        x_host = [-1.0, 0.0, 1.0]
        y_host = np.abs(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.abs(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)

    def test_sign(self):
        r3 = odl.CudaRn(3)
        x_host = [-1.0, 0.0, 1.0]
        y_host = np.sign(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.sign(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)

    def test_sqrt(self):
        r3 = odl.CudaRn(3)
        x_host = [0.1, 0.3, 0.5]
        y_host = np.sqrt(x_host)

        x_dev = r3.element(x_host)
        y_dev = odl.space.cu_ntuples.sqrt(x_dev)

        self.assertAllAlmostEquals(y_host, y_dev, places=5)


@unittest.skipIf(not odl.CUDA_AVAILABLE, "CUDA not available")
class CudaConstWeightedInnerProductTest(ODLTestCase):
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
        constant = 1.5

        # Just test if the code runs
        inner = odl.CudaConstWeightedInnerProduct(constant)

    def test_equals(self):
        constant = 1.5

        inner_const = odl.CudaConstWeightedInnerProduct(constant)
        inner_const2 = odl.CudaConstWeightedInnerProduct(constant)
        inner_const_npy = odl.ConstWeightedInnerProduct(constant)

        self.assertEquals(inner_const, inner_const)
        self.assertEquals(inner_const, inner_const2)
        self.assertEquals(inner_const2, inner_const)

        self.assertNotEquals(inner_const, inner_const_npy)

    def _test_call_real(self, n):
        rn = odl.CudaRn(n)
        xarr, yarr, x, y = self._vectors(rn)

        constant = 1.5
        inner_const = odl.CudaConstWeightedInnerProduct(constant)

        result_const = inner_const(x, y)
        true_result_const = constant * np.dot(yarr, xarr)

        self.assertAlmostEquals(result_const, true_result_const, places=5)

    def test_call(self):
        for _ in range(20):
            self._test_call_real(10)

    def test_repr(self):
        constant = 1.5
        inner_const = odl.CudaConstWeightedInnerProduct(constant)

        repr_str = 'CudaConstWeightedInnerProduct(1.5)'
        self.assertEquals(repr(inner_const), repr_str)

    def test_str(self):
        constant = 1.5
        inner_const = odl.CudaConstWeightedInnerProduct(constant)

        print_str = '(x, y) --> 1.5 * y^H x'
        self.assertEquals(str(inner_const), print_str)


if __name__ == '__main__':
    unittest.main(exit=False)
