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
import pytest
import math
import numpy as np
from numpy import float64

# ODL imports
import odl
from odl.space.ntuples import _FnConstWeighting
if odl.CUDA_AVAILABLE:
    from odl.space.cu_ntuples import _CudaFnConstWeighting

from odl.util.testutils import all_almost_equal, almost_equal, skip_if_no_cuda


# TODO:
# * weighted spaces
# * custom dist/norm/inner


def _vectors(fn, n=1):
    # Generate numpy vectors, real or complex
    if isinstance(fn.field, odl.RealNumbers):
        arrs = [np.random.rand(fn.size) for _ in range(n)]
    else:
        arrs = [np.random.rand(fn.size) + 1j * np.random.rand(fn.size) for _ in range(n)]

    # Make Fn vectors
    vecs = [fn.element(arr) for arr in arrs]
    return arrs + vecs
    
@pytest.mark.skipif("np.float32 not in odl.CUDA_DTYPES")
def test_init_cudantuples_f32():
    #verify that the code runs
    r3 = odl.CudaNtuples(3, dtype='float32')
    r3.element()

@skip_if_no_cuda
def test_init_cudantuples_bad_dtype():
    with pytest.raises(TypeError):
        r3 = odl.CudaNtuples(3, dtype=np.ndarray)
    with pytest.raises(TypeError):
        r3 = odl.CudaNtuples(3, dtype=str)
    with pytest.raises(TypeError):
        r3 = odl.CudaNtuples(3, dtype=np.matrix)

@skip_if_no_cuda
def test_element():
    r3 = odl.CudaRn(3)
    x = r3.element()
    assert x in r3

    y = r3.element(inp=[1, 2, 3])
    assert y in r3

    z = r3.element(data_ptr=y.data_ptr)
    assert z in r3

    with pytest.raises(ValueError):
        w = r3.element(inp=[1, 2, 3], data_ptr=y.data_ptr)

@skip_if_no_cuda
def test_zero():
    r3 = odl.CudaRn(3)
    assert all_almost_equal(r3.zero(), [0, 0, 0])

@skip_if_no_cuda
def test_list_init():
    r3 = odl.CudaRn(3)
    x = r3.element([1, 2, 3])
    assert all_almost_equal(x, [1, 2, 3])

@skip_if_no_cuda
def test_ndarray_init():
    r3 = odl.CudaRn(3)

    x0 = np.array([1., 2., 3.])
    x = r3.element(x0)
    assert all_almost_equal(x, x0)

    x0 = np.array([1, 2, 3], dtype=float64)
    x = r3.element(x0)
    assert all_almost_equal(x, x0)

    x0 = np.array([1, 2, 3], dtype=int)
    x = r3.element(x0)
    assert all_almost_equal(x, x0)

@skip_if_no_cuda
def test_getitem():
    r3 = odl.CudaRn(3)
    y = [1, 2, 3]
    x = r3.element(y)

    for index in [0, 1, 2, -1, -2, -3]:
        assert almost_equal(x[index], y[index])

@skip_if_no_cuda
def test_iterator():
    r3 = odl.CudaRn(3)
    y = [1, 2, 3]
    x = r3.element(y)

    assert all_almost_equal([a for a in x], [b for b in y])

@skip_if_no_cuda
def test_getitem_index_error():
    r3 = odl.CudaRn(3)
    x = r3.element([1, 2, 3])

    with  pytest.raises(IndexError):
        x[-4]

    with  pytest.raises(IndexError):
        x[3]

@skip_if_no_cuda
def test_setitem():
    r3 = odl.CudaRn(3)
    x = r3.element([42, 42, 42])

    for index in [0, 1, 2, -1, -2, -3]:
        x[index] = index
        assert almost_equal(x[index], index)

@skip_if_no_cuda
def test_setitem_index_error():
    r3 = odl.CudaRn(3)
    x = r3.element([1, 2, 3])

    with pytest.raises(IndexError):
        x[-4] = 0

    with pytest.raises(IndexError):
        x[3] = 0

@skip_if_no_cuda
def _test_getslice(slice):
    # Validate get against python list behaviour
    r6 = odl.CudaRn(6)
    y = [0, 1, 2, 3, 4, 5]
    x = r6.element(y)

    assert all_almost_equal(x[slice], y[slice])

@skip_if_no_cuda
def test_getslice():
    # Tests getting all combinations of slices
    steps = [None, -2, -1, 1, 2]
    starts = [None, -1, -3, 0, 2, 5]
    ends = [None, -1, -3, 0, 2, 5]

    for start in starts:
        for end in ends:
            for step in steps:
                _test_getslice(slice(start, end, step))

@skip_if_no_cuda
def test_slice_of_slice():
    # Verify that creating slices from slices works as expected
    r10 = odl.CudaRn(10)
    xh = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xd = r10.element(xh)

    yh = xh[1:8:2]
    yd = xd[1:8:2]

    assert all_almost_equal(yh, yd)

    zh = yh[1::2]
    zd = yd[1::2]

    assert all_almost_equal(zh, zd)

@skip_if_no_cuda
def test_slice_is_view():
    # Verify that modifications of a view modify the original data
    r10 = odl.CudaRn(10)
    xh = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    xd = r10.element(xh)

    yh = xh[1:8:2]
    yh[:] = [0, 0, 0, 0]

    yd = xd[1:8:2]
    yd[:] = [0, 0, 0, 0]

    assert all_almost_equal(xh, xd)
    assert all_almost_equal(yh, yd)

@skip_if_no_cuda
def test_getslice_index_error():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])

    # Bad slice
    with  pytest.raises(IndexError):
        xd[10:13]

def _test_setslice(slice):
    # Validate set against python list behaviour
    r6 = odl.CudaRn(6)
    z = [7, 8, 9, 10, 11, 10]
    y = [0, 1, 2, 3, 4, 5]
    x = r6.element(y)

    x[slice] = z[slice]
    y[slice] = z[slice]
    assert all_almost_equal(x, y)

@skip_if_no_cuda
def test_setslice():
    # Tests a range of combination of slices
    steps = [None, -2, -1, 1, 2]
    starts = [None, -1, -3, 0, 2, 5]
    ends = [None, -1, -3, 0, 2, 5]

    for start in starts:
        for end in ends:
            for step in steps:
                _test_setslice(slice(start, end, step))

@skip_if_no_cuda
def test_setslice_index_error():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])

    # Bad slice
    with pytest.raises(IndexError):
        xd[10:13] = [1, 2, 3]

    # Bad size of rhs
    with pytest.raises(IndexError):
        xd[:] = []

    with pytest.raises(IndexError):
        xd[:] = [1, 2]

    with pytest.raises(IndexError):
        xd[:] = [1, 2, 3, 4]

@skip_if_no_cuda
def test_norm():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])

    correct_norm_squared = 1 ** 2 + 2 ** 2 + 3 ** 2
    correct_norm = math.sqrt(correct_norm_squared)

    # Space function
    assert almost_equal(r3.norm(xd), correct_norm, places=5)

@skip_if_no_cuda
def test_inner():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])
    yd = r3.element([5, 3, 9])

    correct_inner = 1 * 5 + 2 * 3 + 3 * 9

    # Space function
    assert almost_equal(r3.inner(xd, yd), correct_inner)


@skip_if_no_cuda
def _test_lincomb(a, b, n=100):
    # Validates lincomb against the result on host with randomized
    # data and given a,b
    rn = odl.CudaRn(n)

    # Unaliased arguments
    x_arr, y_arr, z_arr, x, y, z = _vectors(rn, 3)

    z_arr[:] = a * x_arr + b * y_arr
    rn.lincomb(a, x, b, y, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], places=4)

    # First argument aliased with output
    x_arr, y_arr, z_arr, x, y, z = _vectors(rn, 3)

    z_arr[:] = a * z_arr + b * y_arr
    rn.lincomb(a, z, b, y, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], places=4)

    # Second argument aliased with output
    x_arr, y_arr, z_arr, x, y, z = _vectors(rn, 3)

    z_arr[:] = a * x_arr + b * z_arr
    rn.lincomb(a, x, b, z, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], places=4)

    # Both arguments aliased with each other
    x_arr, y_arr, z_arr, x, y, z = _vectors(rn, 3)

    z_arr[:] = a * x_arr + b * x_arr
    rn.lincomb(a, x, b, x, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], places=4)

    # All aliased
    x_arr, y_arr, z_arr, x, y, z = _vectors(rn, 3)

    z_arr[:] = a * z_arr + b * z_arr
    rn.lincomb(a, z, b, z, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], places=4)

@skip_if_no_cuda
def test_lincomb():
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(a, b)

def _test_member_lincomb(a, n=100):
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
    assert all_almost_equal(y_device, y_host, places=5)

@skip_if_no_cuda
def test_member_lincomb():
    scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
    for a in scalar_values:
        _test_member_lincomb(a)

@skip_if_no_cuda
def test_multiply():
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
    r3.multiply(x_device, y_device, out=z_device)

    # Cuda only uses floats, so require 5 places
    assert all_almost_equal(z_device, z_host, places=5)

    # Assert input was not modified
    assert all_almost_equal(x_device, x_host, places=5)
    assert all_almost_equal(y_device, y_host, places=5)

    # Aliased
    z_host[:] = z_host * x_host
    r3.multiply(z_device, x_device, out=z_device)

    # Cuda only uses floats, so require 5 places
    assert all_almost_equal(z_device, z_host, places=5)

    # Aliased
    z_host[:] = z_host * z_host
    r3.multiply(z_device, z_device, out=z_device)

    # Cuda only uses floats, so require 5 places
    assert all_almost_equal(z_device, z_host, places=5)

@skip_if_no_cuda
def test_member_multiply():
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
    assert all_almost_equal(y_device, y_host, places=5)

@skip_if_no_cuda
def test_addition():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])
    yd = r3.element([5, 3, 7])

    assert all_almost_equal(xd + yd, [6, 5, 10])

@skip_if_no_cuda
def test_scalar_mult():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])
    C = 5

    assert all_almost_equal(C * xd, [5, 10, 15])

@skip_if_no_cuda
def test_incompatible_operations():
    r3 = odl.CudaRn(3)
    R3h = odl.Rn(3)
    xA = r3.zero()
    xB = R3h.zero()

    with pytest.raises(TypeError):
        xA += xB

    with pytest.raises(TypeError):
        xA -= xB

    with pytest.raises(TypeError):
        xA + xB

    with pytest.raises(TypeError):
        xA - xB

@skip_if_no_cuda
def test_transpose():
    r3 = odl.CudaRn(3)
    x = r3.element([1, 2, 3])
    y = r3.element([5, 3, 8])
    
    # Assert linear operator
    assert isinstance(x.T, odl.Operator)
    assert x.T.is_linear

    # Check result
    assert almost_equal(x.T(y), x.inner(y))
    assert all_almost_equal(x.T.adjoint(1.0), x)
    
    # x.T.T returns self    
    assert x.T.T == x

@skip_if_no_cuda
def test_modify():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])
    yd = r3.element(data_ptr=xd.data_ptr)

    yd[:] = [5, 6, 7]

    assert all_almost_equal(xd, yd)

@skip_if_no_cuda
def test_sub_vector():
    r6 = odl.CudaRn(6)
    r3 = odl.CudaRn(3)
    xd = r6.element([1, 2, 3, 4, 5, 6])

    yd = r3.element(data_ptr=xd.data_ptr)
    yd[:] = [7, 8, 9]

    assert all_almost_equal([7, 8, 9, 4, 5, 6], xd)

@skip_if_no_cuda
def test_offset_sub_vector():
    r6 = odl.CudaRn(6)
    r3 = odl.CudaRn(3)
    xd = r6.element([1, 2, 3, 4, 5, 6])

    yd = r3.element(data_ptr=xd.data_ptr+3*xd.space.dtype.itemsize)
    yd[:] = [7, 8, 9]

    assert all_almost_equal([1, 2, 3, 7, 8, 9], xd)

# Simple tests for the various dtypes
@pytest.mark.skipif("np.int8 not in odl.CUDA_DTYPES")
def test_int8():
    r3 = odl.CudaFn(3, np.int8)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.int16 not in odl.CUDA_DTYPES")
def test_int16():
    r3 = odl.CudaFn(3, np.int16)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.int32 not in odl.CUDA_DTYPES")
def test_int32():
    r3 = odl.CudaFn(3, np.int32)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.int64 not in odl.CUDA_DTYPES")
def test_int64():
    r3 = odl.CudaFn(3, np.int64)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.uint8 not in odl.CUDA_DTYPES")
def test_uint8():
    r3 = odl.CudaFn(3, np.uint8)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.uint16 not in odl.CUDA_DTYPES")
def test_uint16():
    r3 = odl.CudaFn(3, np.uint16)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.uint32 not in odl.CUDA_DTYPES")
def test_uint32():
    r3 = odl.CudaFn(3, np.uint32)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.uint64 not in odl.CUDA_DTYPES")
def test_uint64():
    r3 = odl.CudaFn(3, np.uint64)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.float32 not in odl.CUDA_DTYPES")
def test_float32():
    r3 = odl.CudaFn(3, np.float32)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.float64 not in odl.CUDA_DTYPES")
def test_float64():
    r3 = odl.CudaFn(3, np.float64)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.float not in odl.CUDA_DTYPES")
def test_float():
    r3 = odl.CudaFn(3, np.float)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@pytest.mark.skipif("np.int not in odl.CUDA_DTYPES")
def test_int():
    r3 = odl.CudaFn(3, np.int)
    x = r3.element([1, 2, 3])
    y = r3.element([4, 5, 6])
    z = x + y
    assert all_almost_equal(z, [5, 7, 9])

@skip_if_no_cuda
def test_sin():
    r3 = odl.CudaRn(3)
    x_host = [0.1, 0.3, 10.0]
    y_host = np.sin(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.sin(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_cos():
    r3 = odl.CudaRn(3)
    x_host = [0.1, 0.3, 10.0]
    y_host = np.cos(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.cos(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_arcsin():
    r3 = odl.CudaRn(3)
    x_host = [0.1, 0.3, 0.5]
    y_host = np.arcsin(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.arcsin(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_arccos():
    r3 = odl.CudaRn(3)
    x_host = [0.1, 0.3, 0.5]
    y_host = np.arccos(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.arccos(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_log():
    r3 = odl.CudaRn(3)
    x_host = [0.1, 0.3, 0.5]
    y_host = np.log(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.log(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_exp():
    r3 = odl.CudaRn(3)
    x_host = [-1.0, 0.0, 1.0]
    y_host = np.exp(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.exp(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_abs():
    r3 = odl.CudaRn(3)
    x_host = [-1.0, 0.0, 1.0]
    y_host = np.abs(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.abs(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_sign():
    r3 = odl.CudaRn(3)
    x_host = [-1.0, 0.0, 1.0]
    y_host = np.sign(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.sign(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_sqrt():
    r3 = odl.CudaRn(3)
    x_host = [0.1, 0.3, 0.5]
    y_host = np.sqrt(x_host)

    x_dev = r3.element(x_host)
    y_dev = odl.space.cu_ntuples.sqrt(x_dev)

    assert all_almost_equal(y_host, y_dev, places=5)

@skip_if_no_cuda
def test_const_init():
    constant = 1.5

    # Just test if the code runs
    _CudaFnConstWeighting(constant)

@skip_if_no_cuda
def test_const_equals():
    constant = 1.5

    weighting = _CudaFnConstWeighting(constant)
    weighting2 = _CudaFnConstWeighting(constant)
    other_weighting = _CudaFnConstWeighting(2.5)
    weighting_npy = _FnConstWeighting(constant)

    assert weighting == weighting
    assert weighting == weighting2
    assert weighting2 == weighting

    assert weighting != other_weighting
    assert weighting != weighting_npy
    
def _test_const_call_real(n):
    rn = odl.CudaRn(n)
    xarr, yarr, x, y = _vectors(rn, 2)

    constant = 1.5
    weighting = _CudaFnConstWeighting(constant)

    result_const = weighting.inner(x, y)
    true_result_const = constant * np.dot(yarr, xarr)

    assert almost_equal(result_const, true_result_const, places=5)

@skip_if_no_cuda
def test_const_call():
    for n in range(20):
        _test_const_call_real(n)

@skip_if_no_cuda
def test_const_repr():
    constant = 1.5
    weighting = _CudaFnConstWeighting(constant)

    repr_str = '_CudaFnConstWeighting(1.5)'
    assert repr(weighting) == repr_str

@skip_if_no_cuda
def test_const_str():
    constant = 1.5
    weighting = _CudaFnConstWeighting(constant)

    print_str = '_CudaFnConstWeighting: constant = 1.5'
    assert str(weighting) == print_str

if __name__ == '__main__':
    pytest.main(__file__.replace('\\','/') + ' -v')
