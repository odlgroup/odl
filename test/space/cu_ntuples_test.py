# Copyright 2014-2016 The ODL development group
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
import pytest
import numpy as np
from numpy import float64

# ODL imports
import odl
from odl.space.ntuples import FnConstWeighting
from odl.space.cu_ntuples import (
    CudaFnNoWeighting, CudaFnConstWeighting, CudaFnVectorWeighting,
    CudaFnCustomInnerProduct, CudaFnCustomNorm, CudaFnCustomDist)

from odl.util.testutils import all_equal, all_almost_equal, almost_equal

pytestmark = pytest.mark.skipif("not odl.CUDA_AVAILABLE")

# TODO:
# * custom dist/norm/inner


# Helpers to generate data

def _array(fn):
    # Generate numpy vectors, real or complex or int
    if np.issubdtype(fn.dtype, np.floating):
        return np.random.rand(fn.size).astype(fn.dtype, copy=False)
    elif np.issubdtype(fn.dtype, np.integer):
        return np.random.randint(0, 10, fn.size).astype(fn.dtype, copy=False)
    else:
        return (np.random.rand(fn.size) +
                1j * np.random.rand(fn.size)).astype(fn.dtype, copy=False)


def _element(fn):
    return fn.element(_array(fn))


def _vectors(fn, n=1):
    """Create a list of arrays and vectors in `fn`.

    First arrays, then vectors.
    """
    arrs = [_array(fn) for _ in range(n)]

    # Make Fn vectors
    vecs = [fn.element(arr) for arr in arrs]
    return arrs + vecs


def _pos_array(fn):
    """Create an array with positive real entries as weight in `fn`."""
    return np.abs(_array(fn)) + 0.1


def _pos_vector(fn):
    """Create an vector with positive real entries as weight in `fn`."""
    return fn.element(_pos_array(fn))


# Pytest fixtures


if odl.CUDA_AVAILABLE:
    # Simply modify spc_params to modify the fixture
    spc_params = [odl.CudaRn(100)]
else:
    spc_params = []
spc_ids = [' {!r} '.format(spc) for spc in spc_params]
spc_fixture = pytest.fixture(scope="module", ids=spc_ids,
                             params=spc_params)


@spc_fixture
def fn(request):
    return request.param


# Simply modify exp_params to modify the fixture
exp_params = [2.0, 1.0, float('inf'), 0.5, 1.5, 3.0]
exp_ids = [' p = {} '.format(p) for p in exp_params]
exp_fixture = pytest.fixture(scope="module", ids=exp_ids, params=exp_params)


@exp_fixture
def exponent(request):
    return request.param


# Simply modify dtype_params to modify the fixture
dtype_params = odl.CUDA_DTYPES
dtype_ids = [' dtype = {} '.format(t) for t in dtype_params]
dtype_fixture = pytest.fixture(scope="module", ids=dtype_ids,
                               params=dtype_params)


@dtype_fixture
def dtype(request):
    return request.param

# --- CUDA space tests --- #


def test_init_cudantuples(dtype):
    # verify that the code runs
    odl.CudaNtuples(3, dtype=dtype).element()
    odl.CudaFn(3, dtype=dtype).element()


def test_init_exponent(exponent, dtype):
    odl.CudaFn(3, dtype=dtype, exponent=exponent)


def test_init_cudantuples_bad_dtype():
    with pytest.raises(TypeError):
        odl.CudaNtuples(3, dtype=np.ndarray)
    with pytest.raises(TypeError):
        odl.CudaNtuples(3, dtype=str)
    with pytest.raises(TypeError):
        odl.CudaNtuples(3, dtype=np.matrix)


def test_init_spacefuncs(exponent):
    const = 1.5
    weight_vec = _pos_vector(odl.CudaRn(3))
    weight_elem = odl.CudaFn(3, dtype='float32').element(weight_vec)

    f3_none = odl.CudaFn(3, dtype='float32', exponent=exponent)
    f3_const = odl.CudaFn(3, dtype='float32', weight=const, exponent=exponent)
    f3_vec = odl.CudaFn(3, dtype='float32', weight=weight_vec,
                        exponent=exponent)
    f3_elem = odl.CudaFn(3, dtype='float32', weight=weight_elem,
                         exponent=exponent)

    weighting_none = CudaFnNoWeighting(exponent=exponent)
    weighting_const = CudaFnConstWeighting(const, exponent=exponent)
    weighting_vec = CudaFnVectorWeighting(weight_vec, exponent=exponent)
    weighting_elem = CudaFnVectorWeighting(weight_elem, exponent=exponent)

    assert f3_none._space_funcs == weighting_none
    assert f3_const._space_funcs == weighting_const
    assert f3_vec._space_funcs == weighting_vec
    assert f3_elem._space_funcs == weighting_elem


def test_element(fn):
    x = fn.element()
    assert x in fn

    y = fn.element(inp=[0] * fn.size)
    assert y in fn

    z = fn.element(data_ptr=y.data_ptr)
    assert z in fn

    # Rewrap
    z2 = fn.element(z)
    assert z2 in fn

    w = fn.element(inp=np.zeros(fn.size, fn.dtype))
    assert w in fn

    with pytest.raises(ValueError):
        fn.element(inp=[0] * fn.size, data_ptr=y.data_ptr)


def test_zero(fn):
    assert all_almost_equal(fn.zero(), [0] * fn.size)


def test_one(fn):
    assert all_almost_equal(fn.one(), [1] * fn.size)


def test_list_init(fn):
    x_list = list(range(fn.size))
    x = fn.element(x_list)
    assert all_almost_equal(x, x_list)


def test_ndarray_init():
    r3 = odl.CudaRn(3)

    x0 = np.array([1., 2., 3.])
    x = r3.element(x0)
    assert all_equal(x, x0)

    x0 = np.array([1, 2, 3], dtype=float64)
    x = r3.element(x0)
    assert all_equal(x, x0)

    x0 = np.array([1, 2, 3], dtype=int)
    x = r3.element(x0)
    assert all_equal(x, x0)


def test_getitem():
    r3 = odl.CudaRn(3)
    y = [1, 2, 3]
    x = r3.element(y)

    for index in [0, 1, 2, -1, -2, -3]:
        assert x[index] == y[index]


def test_iterator():
    r3 = odl.CudaRn(3)
    y = [1, 2, 3]
    x = r3.element(y)

    assert all_equal([a for a in x], [b for b in y])


def test_getitem_index_error():
    r3 = odl.CudaRn(3)
    x = r3.element([1, 2, 3])

    with pytest.raises(IndexError):
        x[-4]

    with pytest.raises(IndexError):
        x[3]


def test_setitem():
    r3 = odl.CudaRn(3)
    x = r3.element([42, 42, 42])

    for index in [0, 1, 2, -1, -2, -3]:
        x[index] = index
        assert x[index] == index


def test_setitem_index_error():
    r3 = odl.CudaRn(3)
    x = r3.element([1, 2, 3])

    with pytest.raises(IndexError):
        x[-4] = 0

    with pytest.raises(IndexError):
        x[3] = 0


def _test_getslice(slice):
    # Validate get against python list behaviour
    r6 = odl.CudaRn(6)
    y = [0, 1, 2, 3, 4, 5]
    x = r6.element(y)

    assert all_equal(x[slice], y[slice])


def test_getslice():
    # Tests getting all combinations of slices
    steps = [None, -2, -1, 1, 2]
    starts = [None, -1, -3, 0, 2, 5]
    ends = [None, -1, -3, 0, 2, 5]

    for start in starts:
        for end in ends:
            for step in steps:
                _test_getslice(slice(start, end, step))


def test_slice_of_slice():
    # Verify that creating slices from slices works as expected
    r10 = odl.CudaRn(10)
    xh = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xd = r10.element(xh)

    yh = xh[1:8:2]
    yd = xd[1:8:2]

    assert all_equal(yh, yd)

    zh = yh[1::2]
    zd = yd[1::2]

    assert all_equal(zh, zd)


def test_slice_is_view():
    # Verify that modifications of a view modify the original data
    r10 = odl.CudaRn(10)
    xh = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    xd = r10.element(xh)

    yh = xh[1:8:2]
    yh[:] = [0, 0, 0, 0]

    yd = xd[1:8:2]
    yd[:] = [0, 0, 0, 0]

    assert all_equal(xh, xd)
    assert all_equal(yh, yd)


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
    assert all_equal(x, y)


def test_setslice():
    # Tests a range of combination of slices
    steps = [None, -2, -1, 1, 2]
    starts = [None, -1, -3, 0, 2, 5]
    ends = [None, -1, -3, 0, 2, 5]

    for start in starts:
        for end in ends:
            for step in steps:
                _test_setslice(slice(start, end, step))


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


def test_inner():
    r3 = odl.CudaRn(3)
    x = r3.element([1, 2, 3])
    y = r3.element([5, 3, 9])

    correct_inner = 1 * 5 + 2 * 3 + 3 * 9

    # Space function
    assert almost_equal(r3.inner(x, y), correct_inner)

    # Exponent != 2 -> no inner product
    r3 = odl.CudaRn(3, exponent=1)
    x = r3.element([1, 2, 3])
    y = r3.element([5, 3, 9])

    with pytest.raises(NotImplementedError):
        r3.inner(x, y)
    with pytest.raises(NotImplementedError):
        x.inner(y)


def test_norm(exponent):
    r3 = odl.CudaRn(3, exponent=exponent)
    xarr, x = _vectors(r3)

    correct_norm = np.linalg.norm(xarr, ord=exponent)

    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            r3.norm(x)
            x.norm()
    else:
        assert almost_equal(r3.norm(x), correct_norm)
        assert almost_equal(x.norm(), correct_norm)


def test_dist(exponent):
    r3 = odl.CudaRn(3, exponent=exponent)
    xarr, yarr, x, y = _vectors(r3, n=2)

    correct_dist = np.linalg.norm(xarr - yarr, ord=exponent)

    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            r3.dist(x, y)
        with pytest.raises(NotImplementedError):
            x.dist(y)
    else:
        assert almost_equal(r3.dist(x, y), correct_dist)
        assert almost_equal(x.dist(y), correct_dist)


def _test_lincomb(fn, a, b):
    # Validates lincomb against the result on host with randomized
    # data and given a,b

    # Unaliased arguments
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * x_arr + b * y_arr
    fn.lincomb(a, x, b, y, out=z)
    assert all_almost_equal([x, y, z],
                            [x_arr, y_arr, z_arr])

    # First argument aliased with output
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * z_arr + b * y_arr
    fn.lincomb(a, z, b, y, out=z)
    assert all_almost_equal([x, y, z],
                            [x_arr, y_arr, z_arr])

    # Second argument aliased with output
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * x_arr + b * z_arr
    fn.lincomb(a, x, b, z, out=z)
    assert all_almost_equal([x, y, z],
                            [x_arr, y_arr, z_arr])

    # Both arguments aliased with each other
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * x_arr + b * x_arr
    fn.lincomb(a, x, b, x, out=z)
    assert all_almost_equal([x, y, z],
                            [x_arr, y_arr, z_arr])

    # All aliased
    x_arr, y_arr, z_arr, x, y, z = _vectors(fn, 3)

    z_arr[:] = a * z_arr + b * z_arr
    fn.lincomb(a, z, b, z, out=z)
    assert all_almost_equal([x, y, z],
                            [x_arr, y_arr, z_arr])


def test_lincomb(fn):
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(fn, a, b)


def _test_member_lincomb(spc, a):
    # Validates vector member lincomb against the result on host

    # Generate vectors
    x_host, y_host, x_device, y_device = _vectors(spc, 2)

    # Host side calculation
    y_host[:] = a * x_host

    # Device side calculation
    y_device.lincomb(a, x_device)

    # Cuda only uses floats, so require 5 places
    assert all_almost_equal(y_device, y_host)


def test_member_lincomb(fn):
    scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
    for a in scalar_values:
        _test_member_lincomb(fn, a)


def test_multiply(fn):
    # Validates multiply against the result on host with randomized data
    x_host, y_host, z_host, x_device, y_device, z_device = _vectors(fn, 3)

    # Host side calculation
    z_host[:] = x_host * y_host

    # Device side calculation
    fn.multiply(x_device, y_device, out=z_device)

    assert all_almost_equal([x_device, y_device, z_device],
                            [x_host, y_host, z_host])

    # Aliased
    z_host[:] = z_host * x_host
    fn.multiply(z_device, x_device, out=z_device)

    assert all_almost_equal([x_device, z_device],
                            [x_host, z_host])

    # Aliased
    z_host[:] = z_host * z_host
    fn.multiply(z_device, z_device, out=z_device)

    assert all_almost_equal(z_device, z_host)


def test_member_multiply(fn):
    # Validate vector member multiply against the result on host
    # with randomized data
    x_host, y_host, x_device, y_device = _vectors(fn, 2)

    # Host side calculation
    y_host *= x_host

    # Device side calculation
    y_device *= x_device

    # Cuda only uses floats, so require 5 places
    assert all_almost_equal(y_device, y_host)


def _test_unary_operator(spc, function):
    # Verify that the statement y=function(x) gives equivalent
    # results to Numpy.
    x_arr, x = _vectors(spc)

    y_arr = function(x_arr)
    y = function(x)

    assert all_almost_equal([x, y],
                            [x_arr, y_arr])


def _test_binary_operator(spc, function):
    # Verify that the statement z=function(x,y) gives equivalent
    # results to Numpy.
    x_arr, y_arr, x, y = _vectors(spc, 2)

    z_arr = function(x_arr, y_arr)
    z = function(x, y)

    assert all_almost_equal([x, y, z],
                            [x_arr, y_arr, z_arr])


def test_operators(fn):
    # Test of all operator overloads against the corresponding
    # Numpy implementation

    # Unary operators
    _test_unary_operator(fn, lambda x: +x)
    _test_unary_operator(fn, lambda x: -x)

    # Scalar multiplication
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def imul(x):
            x *= scalar
        _test_unary_operator(fn, imul)
        _test_unary_operator(fn, lambda x: x * scalar)

    # Scalar division
    for scalar in [-31.2, -1, 1, 2.13]:
        def idiv(x):
            x /= scalar
        _test_unary_operator(fn, idiv)
        _test_unary_operator(fn, lambda x: x / scalar)

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


def test_copy(fn):
    import copy

    x = _element(fn)
    y = copy.copy(x)

    assert x == y
    assert y is not x

    z = copy.deepcopy(x)

    assert x == z
    assert z is not x


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


def test_modify():
    r3 = odl.CudaRn(3)
    xd = r3.element([1, 2, 3])
    yd = r3.element(data_ptr=xd.data_ptr)

    yd[:] = [5, 6, 7]

    assert all_equal(xd, yd)


def test_sub_vector():
    r6 = odl.CudaRn(6)
    r3 = odl.CudaRn(3)
    xd = r6.element([1, 2, 3, 4, 5, 6])

    yd = r3.element(data_ptr=xd.data_ptr)
    yd[:] = [7, 8, 9]

    assert all_almost_equal([7, 8, 9, 4, 5, 6], xd)


def test_offset_sub_vector():
    r6 = odl.CudaRn(6)
    r3 = odl.CudaRn(3)
    xd = r6.element([1, 2, 3, 4, 5, 6])

    yd = r3.element(data_ptr=xd.data_ptr + 3 * xd.space.dtype.itemsize)
    yd[:] = [7, 8, 9]

    assert all_equal([1, 2, 3, 7, 8, 9], xd)


def _test_dtype(dt):
    if dt not in odl.CUDA_DTYPES:
        with pytest.raises(TypeError):
            r3 = odl.CudaFn(3, dt)
    else:
        r3 = odl.CudaFn(3, dt)
        x = r3.element([1, 2, 3])
        y = r3.element([4, 5, 6])
        z = x + y
        assert all_almost_equal(z, [5, 7, 9])


def test_dtypes():
    for dt in [np.int8, np.int16, np.int32, np.int64, np.int,
               np.uint8, np.uint16, np.uint32, np.uint64, np.uint,
               np.float32, np.float64, np.float,
               np.complex64, np.complex128, np.complex]:
        yield _test_dtype, dt

# --- Weighting tests --- #


def test_const_init(exponent):
    const = 1.5
    CudaFnConstWeighting(const, exponent=exponent)


def test_const_equals(exponent):
    constant = 1.5

    weighting = CudaFnConstWeighting(constant, exponent=exponent)
    weighting2 = CudaFnConstWeighting(constant, exponent=exponent)
    other_weighting = CudaFnConstWeighting(2.5, exponent=exponent)
    wrong_exp = FnConstWeighting(constant, exponent=exponent + 1)

    assert weighting == weighting
    assert weighting == weighting2
    assert weighting2 == weighting

    assert weighting != other_weighting
    assert weighting != wrong_exp


def test_const_inner():
    rn = odl.CudaRn(5)
    xarr, yarr, x, y = _vectors(rn, 2)

    constant = 1.5
    weighting = CudaFnConstWeighting(constant)

    true_inner = constant * np.vdot(yarr, xarr)
    assert almost_equal(weighting.inner(x, y), true_inner)


def test_const_norm(exponent):
    rn = odl.CudaRn(5)
    xarr, x = _vectors(rn)

    constant = 1.5
    weighting = CudaFnConstWeighting(constant, exponent=exponent)

    factor = 1 if exponent == float('inf') else constant ** (1 / exponent)
    true_norm = factor * np.linalg.norm(xarr, ord=exponent)

    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            weighting.norm(x)
    else:
        assert almost_equal(weighting.norm(x), true_norm)

    # Same with free function
    pnorm = odl.cu_weighted_norm(constant, exponent=exponent)
    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            pnorm(x)
    else:
        assert almost_equal(pnorm(x), true_norm)


def test_const_dist(exponent):
    rn = odl.CudaRn(5)
    xarr, yarr, x, y = _vectors(rn, n=2)

    constant = 1.5
    weighting = CudaFnConstWeighting(constant, exponent=exponent)

    factor = 1 if exponent == float('inf') else constant ** (1 / exponent)
    true_dist = factor * np.linalg.norm(xarr - yarr, ord=exponent)

    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            weighting.dist(x, y)
    else:
        assert almost_equal(weighting.dist(x, y), true_dist)

    # Same with free function
    pdist = odl.cu_weighted_dist(constant, exponent=exponent)
    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            pdist(x, y)
    else:
        assert almost_equal(pdist(x, y), true_dist)


def test_vector_init():
    rn = odl.CudaRn(5)
    weight_vec = _pos_vector(rn)

    CudaFnVectorWeighting(weight_vec)
    CudaFnVectorWeighting(rn.element(weight_vec))


def test_vector_isvalid():
    rn = odl.CudaRn(5)
    weight = _pos_vector(rn)

    weighting = CudaFnVectorWeighting(weight)

    assert weighting.vector_is_valid()

    # Invalid
    weight[0] = 0

    weighting = CudaFnVectorWeighting(weight)

    assert not weighting.vector_is_valid()


def test_vector_equals():
    rn = odl.CudaRn(5)
    weight = _pos_vector(rn)

    weighting = CudaFnVectorWeighting(weight)
    weighting2 = CudaFnVectorWeighting(weight)

    assert weighting == weighting2


def test_vector_inner():
    rn = odl.CudaRn(5)
    xarr, yarr, x, y = _vectors(rn, 2)

    weight = _pos_vector(odl.CudaRn(5))

    weighting = CudaFnVectorWeighting(weight)

    true_inner = np.vdot(yarr, xarr * weight.asarray())

    assert almost_equal(weighting.inner(x, y), true_inner)

    # Same with free function
    inner_vec = odl.cu_weighted_inner(weight)

    assert almost_equal(inner_vec(x, y), true_inner)

    # Exponent != 2 -> no inner product, should raise
    with pytest.raises(NotImplementedError):
        CudaFnVectorWeighting(weight, exponent=1.0).inner(x, y)


def test_vector_norm(exponent):
    rn = odl.CudaRn(5)
    xarr, x = _vectors(rn)

    weight = _pos_vector(odl.CudaRn(5))

    weighting = CudaFnVectorWeighting(weight, exponent=exponent)

    if exponent in (1.0, float('inf')):
        true_norm = np.linalg.norm(weight.asarray() * xarr, ord=exponent)
    else:
        true_norm = np.linalg.norm(weight.asarray() ** (1 / exponent) * xarr,
                                   ord=exponent)

    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            weighting.norm(x)
    else:
        assert almost_equal(weighting.norm(x), true_norm)

    # Same with free function
    pnorm = odl.cu_weighted_norm(weight, exponent=exponent)

    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            pnorm(x)
    else:
        assert almost_equal(pnorm(x), true_norm)


def test_vector_dist(exponent):
    rn = odl.CudaRn(5)
    xarr, yarr, x, y = _vectors(rn, n=2)

    weight = _pos_vector(odl.CudaRn(5))

    weighting = CudaFnVectorWeighting(weight, exponent=exponent)

    if exponent in (1.0, float('inf')):
        true_dist = np.linalg.norm(weight.asarray() * (xarr - yarr),
                                   ord=exponent)
    else:
        true_dist = np.linalg.norm(
            weight.asarray() ** (1 / exponent) * (xarr - yarr), ord=exponent)

    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            weighting.dist(x, y)
    else:
        assert almost_equal(weighting.dist(x, y), true_dist)

    # Same with free function
    pdist = odl.cu_weighted_dist(weight, exponent=exponent)

    if exponent == float('inf'):
        # Not yet implemented, should raise
        with pytest.raises(NotImplementedError):
            pdist(x, y)
    else:
        assert almost_equal(pdist(x, y), true_dist)


def test_custom_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    def inner(x, y):
        return np.vdot(y, x)

    w = CudaFnCustomInnerProduct(inner)
    w_same = CudaFnCustomInnerProduct(inner)
    w_other = CudaFnCustomInnerProduct(np.dot)
    w_d = CudaFnCustomInnerProduct(inner, dist_using_inner=False)

    assert w == w
    assert w == w_same
    assert w != w_other
    assert w != w_d

    true_inner = inner(xarr, yarr)
    assert almost_equal(w.inner(x, y), true_inner)

    true_norm = np.linalg.norm(xarr)
    assert almost_equal(w.norm(x), true_norm)

    true_dist = np.linalg.norm(xarr - yarr)
    assert almost_equal(w.dist(x, y), true_dist)
    assert almost_equal(w.dist(x, x), 0)
    assert almost_equal(w_d.dist(x, y), true_dist)
    assert almost_equal(w_d.dist(x, x), 0)

    with pytest.raises(TypeError):
        CudaFnCustomInnerProduct(1)


def test_custom_norm(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    norm = np.linalg.norm

    def other_norm(x):
        return np.linalg.norm(x, ord=1)

    w = CudaFnCustomNorm(norm)
    w_same = CudaFnCustomNorm(norm)
    w_other = CudaFnCustomNorm(other_norm)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    true_norm = np.linalg.norm(xarr)
    assert almost_equal(w.norm(x), true_norm)

    true_dist = np.linalg.norm(xarr - yarr)
    assert almost_equal(w.dist(x, y), true_dist)
    assert almost_equal(w.dist(x, x), 0)

    with pytest.raises(TypeError):
        CudaFnCustomNorm(1)


def test_custom_dist(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    def dist(x, y):
        return np.linalg.norm(x - y)

    def other_dist(x, y):
        return np.linalg.norm(x - y, ord=1)

    w = CudaFnCustomDist(dist)
    w_same = CudaFnCustomDist(dist)
    w_other = CudaFnCustomDist(other_dist)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    with pytest.raises(NotImplementedError):
        w.norm(x)

    true_dist = np.linalg.norm(xarr - yarr)
    assert almost_equal(w.dist(x, y), true_dist)
    assert almost_equal(w.dist(x, x), 0)

    with pytest.raises(TypeError):
        CudaFnCustomDist(1)


@pytest.mark.skipif("not odl.CUDA_AVAILABLE")
def _impl_test_ufuncs(fn, name, n_args, n_out):
    # Get the ufunc from numpy as reference
    ufunc = getattr(np, name)

    # Create some data
    data = _vectors(fn, n_args + n_out)
    in_arrays = data[:n_args]
    out_arrays = data[n_args:n_args + n_out]
    data_vector = data[n_args + n_out]
    in_vectors = data[1 + n_args + n_out:2 * n_args + n_out]
    out_vectors = data[2 * n_args + n_out:]

    # Out of place:
    np_result = ufunc(*in_arrays)
    vec_fun = getattr(data_vector.ufunc, name)
    odl_result = vec_fun(*in_vectors)
    assert all_almost_equal(np_result, odl_result)

    # Test type of output
    if n_out == 1:
        assert isinstance(odl_result, fn.element_type)
    elif n_out > 1:
        for i in range(n_out):
            assert isinstance(odl_result[i], fn.element_type)

    # In place:
    np_result = ufunc(*(in_arrays + out_arrays))
    vec_fun = getattr(data_vector.ufunc, name)
    odl_result = vec_fun(*(in_vectors + out_vectors))
    assert all_almost_equal(np_result, odl_result)

    # Test inplace actually holds:
    if n_out == 1:
        assert odl_result is out_vectors[0]
    elif n_out > 1:
        for i in range(n_out):
            assert odl_result[i] is out_vectors[i]


def test_ufuncs():
    # Cannot use fixture due to bug in pytest
    for fn in spc_params:
        for name, n_args, n_out, _ in odl.util.ufuncs.UFUNCS:
            if (np.issubsctype(fn.dtype, np.floating) and
                    name in ['bitwise_and',
                             'bitwise_or',
                             'bitwise_xor',
                             'invert',
                             'left_shift',
                             'right_shift']):
                # Skip integer only methods if floating point type
                continue
            yield _impl_test_ufuncs, fn, name, n_args, n_out


def _impl_test_reduction(fn, name):
    ufunc = getattr(np, name)

    # Create some data
    x_arr, x = _vectors(fn, 1)

    assert almost_equal(ufunc(x_arr), getattr(x.ufunc, name)())


def test_reductions():
    for fn in spc_params:
        for name, _ in odl.util.ufuncs.REDUCTIONS:
            yield _impl_test_reduction, fn, name


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
