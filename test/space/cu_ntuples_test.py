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
from odl.space.ntuples import FnConstWeighting
if odl.CUDA_AVAILABLE:
    from odl.space.cu_ntuples import _CudaFnConstWeighting

from odl.util.testutils import all_almost_equal, almost_equal, skip_if_no_cuda


# TODO:
# * weighted spaces
# * custom dist/norm/inner


def _array(fn):
    # Generate numpy vectors, real or complex or int
    if np.issubdtype(fn.dtype, np.floating):
        return np.random.randn(fn.size).astype(fn.dtype)
    elif np.issubdtype(fn.dtype, np.integer):
        return np.random.randint(0, 10, fn.size).astype(fn.dtype)
    else:
        return (np.random.randn(fn.size) +
                1j * np.random.randn(fn.size)).astype(fn.dtype)


def _element(fn):
    return fn.element(_array(fn))


def _vectors(fn, n=1):
    arrs = [_array(fn) for _ in range(n)]

    # Make Fn vectors
    vecs = [fn.element(arr) for arr in arrs]
    return arrs + vecs


if odl.CUDA_AVAILABLE:
    ids=['CudaRn float32']
    params=[odl.CudaRn(100)]
else:
    ids=['Cuda test']
    params=[None]

@pytest.fixture(scope="module",
                ids=ids,
                params=params)
def fn(request):
    return request.param


@pytest.mark.skipif("np.float32 not in odl.CUDA_DTYPES")
def test_init_cudantuples_f32():
    # verify that the code runs
    r3 = odl.CudaNtuples(3, dtype='float32')
    r3.element()


@skip_if_no_cuda
def test_init_cudantuples_bad_dtype():
    with pytest.raises(TypeError):
        odl.CudaNtuples(3, dtype=np.ndarray)
    with pytest.raises(TypeError):
        odl.CudaNtuples(3, dtype=str)
    with pytest.raises(TypeError):
        odl.CudaNtuples(3, dtype=np.matrix)


@skip_if_no_cuda
def test_element(fn):
    x = fn.element()
    assert x in fn

    y = fn.element(inp=[0]*fn.size)
    assert y in fn

    z = fn.element(data_ptr=y.data_ptr)
    assert z in fn

    with pytest.raises(ValueError):
        fn.element(inp=[0]*fn.size, data_ptr=y.data_ptr)


@skip_if_no_cuda
def test_zero(fn):
    assert all_almost_equal(fn.zero(), [0]*fn.size)


@skip_if_no_cuda
def test_list_init(fn):
    x_list = list(range(fn.size))
    x = fn.element(x_list)
    assert all_almost_equal(x, x_list)


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

    with pytest.raises(IndexError):
        x[-4]

    with pytest.raises(IndexError):
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
    with pytest.raises(IndexError):
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
    assert almost_equal(r3.norm(xd), correct_norm)


@skip_if_no_cuda
def test_inner():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])
    yd = r3.element([5, 3, 9])

    correct_inner = 1 * 5 + 2 * 3 + 3 * 9

    # Space function
    assert almost_equal(r3.inner(xd, yd), correct_inner)


@skip_if_no_cuda
def _test_lincomb(fn, a, b):
    # Validates lincomb against the result on host with randomized
    # data and given a,b

    # Unaliased arguments
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * x_arr + b * y_arr
    fn.lincomb(a, x, b, y, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    # First argument aliased with output
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * z_arr + b * y_arr
    fn.lincomb(a, z, b, y, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    # Second argument aliased with output
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * x_arr + b * z_arr
    fn.lincomb(a, x, b, z, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    # Both arguments aliased with each other
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * x_arr + b * x_arr
    fn.lincomb(a, x, b, x, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    # All aliased
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * z_arr + b * z_arr
    fn.lincomb(a, z, b, z, out=z)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])


@skip_if_no_cuda
def test_lincomb(fn):
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(fn, a, b)


def _test_member_lincomb(fn, a):
    # Validates vector member lincomb against the result on host

    # Generate vectors
    x_host, y_host, x_device, y_device = _vectors(fn, 2)

    # Host side calculation
    y_host[:] = a * x_host

    # Device side calculation
    y_device.lincomb(a, x_device)

    # Cuda only uses floats, so require 5 places
    assert all_almost_equal(y_device, y_host)


@skip_if_no_cuda
def test_member_lincomb(fn):
    scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
    for a in scalar_values:
        _test_member_lincomb(fn, a)


@skip_if_no_cuda
def test_multiply():
    # Validates multiply against the result on host with randomized data
    rn = odl.CudaRn(100)
    x_host, y_host, z_host, x_device, y_device, z_device = _vectors(rn, 3)

    # Host side calculation
    z_host[:] = x_host * y_host

    # Device side calculation
    rn.multiply(x_device, y_device, out=z_device)

    assert all_almost_equal([x_device, y_device, z_device],
                            [x_host, y_host, z_host])

    # Aliased
    z_host[:] = z_host * x_host
    rn.multiply(z_device, x_device, out=z_device)

    assert all_almost_equal([x_device, z_device],
                            [x_host, z_host])

    # Aliased
    z_host[:] = z_host * z_host
    rn.multiply(z_device, z_device, out=z_device)

    assert all_almost_equal(z_device, z_host)


@skip_if_no_cuda
def test_member_multiply():
    # Validates vector member multiply against the result on host
    # with randomized data
    rn = odl.CudaRn(100)
    x_host, y_host, x_device, y_device = _vectors(rn, 2)

    # Host side calculation
    y_host[:] = x_host * y_host

    # Device side calculation
    y_device *= x_device

    # Cuda only uses floats, so require 5 places
    assert all_almost_equal(y_device, y_host)

    
@skip_if_no_cuda
def _test_unary_operator(fn, function):
    """ Verifies that the statement y=function(x) gives equivalent
    results to Numpy.
    """

    x_arr, x = _vectors(fn)

    y_arr = function(x_arr)
    y = function(x)

    assert all_almost_equal([x, y], [x_arr, y_arr])

    
@skip_if_no_cuda
def _test_binary_operator(fn, function):
    """ Verifies that the statement z=function(x,y) gives equivalent
    results to Numpy.
    """

    x_arr, y_arr, x, y = _vectors(fn, 2)

    z_arr = function(x_arr, y_arr)
    z = function(x, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    
@skip_if_no_cuda
def test_operators(fn):
    """ Test of all operator overloads against the corresponding
    Numpy implementation
    """
    # Unary operators
    _test_unary_operator(fn, lambda x: +x)
    _test_unary_operator(fn, lambda x: -x)

    # Scalar multiplication
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def imul(x):
            x *= scalar
        _test_unary_operator(fn, imul)
        _test_unary_operator(fn, lambda x: x*scalar)

    # Scalar division
    for scalar in [-31.2, -1, 1, 2.13]:
        def idiv(x):
            x /= scalar
        _test_unary_operator(fn, idiv)
        _test_unary_operator(fn, lambda x: x/scalar)

    # Incremental operations
    def iadd(x, y):
        x += y

    def isub(x, y):
        x -= y

    def imul(x, y):
        x *= y

    def idiv(x, y):
        x /= y

    _test_binary_operator(fn, iadd)
    _test_binary_operator(fn, isub)
    _test_binary_operator(fn, imul)
    _test_binary_operator(fn, idiv)

    # Incremental operators with aliased inputs
    def iadd_aliased(x):
        x += x

    def isub_aliased(x):
        x -= x

    def imul_aliased(x):
        x *= x

    def idiv_aliased(x):
        x /= x

    _test_unary_operator(fn, iadd_aliased)
    _test_unary_operator(fn, isub_aliased)
    _test_unary_operator(fn, imul_aliased)
    _test_unary_operator(fn, idiv_aliased)

    # Binary operators
    _test_binary_operator(fn, lambda x, y: x + y)
    _test_binary_operator(fn, lambda x, y: x - y)
    _test_binary_operator(fn, lambda x, y: x * y)
    _test_binary_operator(fn, lambda x, y: x / y)

    # Binary with aliased inputs
    _test_unary_operator(fn, lambda x: x + x)
    _test_unary_operator(fn, lambda x: x - x)
    _test_unary_operator(fn, lambda x: x * x)
    _test_unary_operator(fn, lambda x: x / x)


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
def test_copy(fn):
    import copy

    x = _element(fn)
    y = copy.copy(x)

    assert x == y
    assert y is not x

    z = copy.deepcopy(x)

    assert x == z
    assert z is not x


@skip_if_no_cuda
def test_transpose(fn):
    x = _element(fn)
    y = _element(fn)

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

    
@skip_if_no_cuda
def _test_dtype(dtype):
    if dtype not in odl.CUDA_DTYPES:
        with pytest.raises(TypeError):
            r3 = odl.CudaFn(3, dtype)
    else:
        r3 = odl.CudaFn(3, dtype)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        assert all_almost_equal(z, [5, 7, 9])


@skip_if_no_cuda
def test_dtypes():
    for dtype in [np.int8, np.int16, np.int32, np.int64, np.int,
                  np.uint8, np.uint16, np.uint32, np.uint64, np.uint,
                  np.float32, np.float64, np.float,
                  np.complex64, np.complex128, np.complex]:
        yield _test_dtype, dtype

        
@skip_if_no_cuda
def _test_ufunc(ufunc):
    r3 = odl.CudaRn(5)
    x_host = [-1.0, 0, 0.1, 0.3, 10.0]
    y_host = getattr(np, ufunc)(x_host)

    x_dev = r3.element(x_host)
    y_dev = getattr(odl.space.cu_ntuples, ufunc)(x_dev)

    assert all_almost_equal(y_host, y_dev)


@skip_if_no_cuda
def test_ufuncs():
    for ufunc in ['sin', 'cos',
                  'arcsin', 'arccos',
                  'log', 'exp',
                  'abs', 'sign', 'sqrt']:
        yield _test_ufunc, ufunc


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
    weighting_npy = FnConstWeighting(constant)

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

    assert almost_equal(result_const, true_result_const)


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
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))

