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

# External module imports
import pytest
import numpy as np

# Internal
import odl
from odl.discr.lp_discr import DiscreteLp
from odl.util.testutils import (almost_equal, all_equal, all_almost_equal,
                                never_skip, skip_if_no_cuda, example_vectors)

# Pytest fixture


# Simply modify exp_params to modify the fixture
exp_params = [2.0, 1.0, float('inf'), 0.5, 1.5]
exp_ids = [' p = {} '.format(p) for p in exp_params]


@pytest.fixture(scope="module", ids=exp_ids, params=exp_params)
def exponent(request):
    return request.param


impl_params = [never_skip('numpy'),
               skip_if_no_cuda('cuda')]
impl_ids = [' impl = {} '.format(p.args[1]) for p in impl_params]


@pytest.fixture(scope="module", ids=impl_ids, params=impl_params)
def impl(request):
    return request.param


def test_init(exponent):
    # Validate that the different init patterns work and do not crash.
    space = odl.FunctionSpace(odl.Interval(0, 1))
    part = odl.uniform_partition_fromintv(space.domain, 10)
    rn = odl.Rn(10, exponent=exponent)
    odl.DiscreteLp(space, part, rn, exponent=exponent)
    odl.DiscreteLp(space, part, rn, exponent=exponent, interp='linear')

    # Normal discretization of unit interval with complex
    complex_space = odl.FunctionSpace(odl.Interval(0, 1),
                                      field=odl.ComplexNumbers())
    cn = odl.Cn(10, exponent=exponent)
    odl.DiscreteLp(complex_space, part, cn, exponent=exponent)

    space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    part = odl.uniform_partition_fromintv(space.domain, (10, 10))
    rn = odl.Rn(100, exponent=exponent)
    odl.DiscreteLp(space, part, rn, exponent=exponent,
                   interp=['nearest', 'linear'])

    # Real space should not work with complex
    with pytest.raises(ValueError):
        odl.DiscreteLp(space, part, cn)

    # Complex space should not work with reals
    with pytest.raises(ValueError):
        odl.DiscreteLp(complex_space, part, rn)

    # Wrong size of underlying space
    rn_wrong_size = odl.Rn(20)
    with pytest.raises(ValueError):
        odl.DiscreteLp(space, part, rn_wrong_size)


@skip_if_no_cuda
def test_init_cuda(exponent):
    # Normal discretization of unit interval
    space = odl.FunctionSpace(odl.Interval(0, 1), out_dtype='float32')
    part = odl.uniform_partition_fromintv(space.domain, 10)
    rn = odl.CudaRn(10, exponent=exponent)
    odl.DiscreteLp(space, part, rn, exponent=exponent)


def test_factory(exponent):
    discr = odl.uniform_discr(0, 1, 10, impl='numpy', exponent=exponent)

    assert isinstance(discr.dspace, odl.Fn)
    assert discr.is_rn
    assert discr.dspace.exponent == exponent
    assert discr.dtype == odl.Fn.default_dtype(odl.RealNumbers())

    # Complex
    discr = odl.uniform_discr(0, 1, 10, dtype='complex',
                              impl='numpy', exponent=exponent)

    assert isinstance(discr.dspace, odl.Fn)
    assert discr.is_cn
    assert discr.dspace.exponent == exponent
    assert discr.dtype == odl.Fn.default_dtype(odl.ComplexNumbers())


@skip_if_no_cuda
def test_factory_cuda(exponent):
    discr = odl.uniform_discr(0, 1, 10, impl='cuda', exponent=exponent)
    assert isinstance(discr.dspace, odl.CudaFn)
    assert discr.is_rn
    assert discr.dspace.exponent == exponent
    assert discr.dtype == odl.CudaFn.default_dtype(odl.RealNumbers())

    # Cuda currently does not support complex numbers, check error
    with pytest.raises(NotImplementedError):
        odl.uniform_discr(0, 1, 10, impl='cuda', dtype='complex')


def test_factory_dtypes():
    real_float_dtypes = [np.float32, np.float64]
    nonfloat_dtypes = [np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64]
    complex_float_dtypes = [np.complex64, np.complex128]

    for dtype in real_float_dtypes:
        discr = odl.uniform_discr(0, 1, 10, impl='numpy', dtype=dtype)
        assert isinstance(discr.dspace, odl.Fn)
        assert discr.is_rn

    for dtype in nonfloat_dtypes:
        discr = odl.uniform_discr(0, 1, 10, impl='numpy', dtype=dtype)
        assert isinstance(discr.dspace, odl.Fn)
        assert discr.dspace.element().space.dtype == dtype

    for dtype in complex_float_dtypes:
        discr = odl.uniform_discr(0, 1, 10, impl='numpy', dtype=dtype)
        assert isinstance(discr.dspace, odl.Fn)
        assert discr.is_cn
        assert discr.dspace.element().space.dtype == dtype


@skip_if_no_cuda
def test_factory_dtypes_cuda():
    real_float_dtypes = [np.float32, np.float64]
    nonfloat_dtypes = [np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64]
    complex_float_dtypes = [np.complex64, np.complex128]

    for dtype in real_float_dtypes:
        if dtype not in odl.CUDA_DTYPES:
            with pytest.raises(TypeError):
                odl.uniform_discr(0, 1, 10, impl='cuda', dtype=dtype)
        else:
            discr = odl.uniform_discr(0, 1, 10, impl='cuda', dtype=dtype)
            assert isinstance(discr.dspace, odl.CudaFn)
            assert discr.is_rn
            assert discr.dspace.element().space.dtype == dtype

    for dtype in nonfloat_dtypes:
        if dtype not in odl.CUDA_DTYPES:
            with pytest.raises(TypeError):
                odl.uniform_discr(0, 1, 10, impl='cuda', dtype=dtype)
        else:
            discr = odl.uniform_discr(0, 1, 10, impl='cuda', dtype=dtype)
            assert isinstance(discr.dspace, odl.CudaFn)
            assert not discr.is_rn
            assert discr.dspace.element().space.dtype == dtype

    for dtype in complex_float_dtypes:
        with pytest.raises(NotImplementedError):
            odl.uniform_discr(0, 1, 10, impl='cuda', dtype=dtype)


def test_factory_nd(exponent):
    # 2d
    odl.uniform_discr([0, 0], [1, 1], [5, 5], exponent=exponent)
    odl.uniform_discr([0, 0], [1, 1], [5, 5], exponent=exponent,
                      interp=['linear', 'nearest'])

    # 3d
    odl.uniform_discr([0, 0, 0], [1, 1, 1], [5, 5, 5], exponent=exponent)

    # nd
    odl.uniform_discr([0] * 10, [1] * 10, [5] * 10, exponent=exponent)


def test_element_1d(exponent):
    discr = odl.uniform_discr(0, 1, 3, impl='numpy', exponent=exponent)
    weight = 1.0 if exponent == float('inf') else discr.cell_volume
    dspace = odl.Rn(3, exponent=exponent, weight=weight)
    vec = discr.element()
    assert isinstance(vec, odl.DiscreteLpVector)
    assert vec.ntuple in dspace


def test_element_2d(exponent):
    discr = odl.uniform_discr([0, 0], [1, 1], [3, 3],
                              impl='numpy', exponent=exponent)
    weight = 1.0 if exponent == float('inf') else discr.cell_volume
    dspace = odl.Rn(9, exponent=exponent, weight=weight)
    vec = discr.element()
    assert isinstance(vec, odl.DiscreteLpVector)
    assert vec.ntuple in dspace


def test_element_from_array_1d():
    discr = odl.uniform_discr(0, 1, 3, impl='numpy')
    vec = discr.element([1, 2, 3])

    assert isinstance(vec, odl.DiscreteLpVector)
    assert isinstance(vec.ntuple, odl.FnVector)
    assert all_equal(vec.ntuple, [1, 2, 3])


def test_element_from_array_2d():
    # assert orderings work properly with 2d
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl='numpy', order='C')
    vec = discr.element([[1, 2],
                         [3, 4]])

    assert isinstance(vec, odl.DiscreteLpVector)
    assert isinstance(vec.ntuple, odl.FnVector)

    # Check ordering
    assert all_equal(vec.ntuple, [1, 2, 3, 4])

    # Linear creation works as well
    linear_vec = discr.element([1, 2, 3, 4])
    assert all_equal(vec.ntuple, [1, 2, 3, 4])

    # Fortran order
    discr = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl='numpy', order='F')
    vec = discr.element([[1, 2],
                         [3, 4]])

    # Check ordering
    assert all_equal(vec.ntuple, [1, 3, 2, 4])

    # Linear creation works aswell
    linear_vec = discr.element([1, 2, 3, 4])
    assert all_equal(linear_vec.ntuple, [1, 2, 3, 4])


def test_element_from_array_2d_shape():
    # Verify that the shape is correctly tested for
    discr = odl.uniform_discr([0, 0], [1, 1], [3, 2], impl='numpy', order='C')

    # Correct order
    discr.element([[1, 2],
                   [3, 4],
                   [5, 6]])

    # Wrong order, should throw
    with pytest.raises(ValueError):
        discr.element([[1, 2, 3],
                       [4, 5, 6]])

    # Wrong number of elements, should throw
    with pytest.raises(ValueError):
        discr.element([[1, 2],
                       [3, 4]])


def test_zero():
    discr = odl.uniform_discr(0, 1, 3)
    vec = discr.zero()

    assert isinstance(vec, odl.DiscreteLpVector)
    assert isinstance(vec.ntuple, odl.FnVector)
    assert all_equal(vec, [0, 0, 0])


def _test_unary_operator(discr, function):
    # Verify that the statement y=function(x) gives equivalent results
    # to NumPy
    x_arr, x = example_vectors(discr)

    y_arr = function(x_arr)

    y = function(x)

    assert all_almost_equal([x, y], [x_arr, y_arr])


def _test_binary_operator(discr, function):
    # Verify that the statement z=function(x,y) gives equivalent results
    # to NumPy
    [x_arr, y_arr], [x, y] = example_vectors(discr, 2)

    z_arr = function(x_arr, y_arr)
    z = function(x, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])


def test_operators(impl):
    # Test of all operator overloads against the corresponding NumPy
    # implementation

    discr = odl.uniform_discr(0, 1, 10, impl=impl)

    # Unary operators
    _test_unary_operator(discr, lambda x: +x)
    _test_unary_operator(discr, lambda x: -x)

    # Scalar addition
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def iadd(x):
            x += scalar
        _test_unary_operator(discr, iadd)
        _test_unary_operator(discr, lambda x: x + scalar)

    # Scalar subtraction
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def isub(x):
            x -= scalar
        _test_unary_operator(discr, isub)
        _test_unary_operator(discr, lambda x: x - scalar)

    # Scalar multiplication
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def imul(x):
            x *= scalar
        _test_unary_operator(discr, imul)
        _test_unary_operator(discr, lambda x: x * scalar)

    # Scalar division
    for scalar in [-31.2, -1, 1, 2.13]:
        def idiv(x):
            x /= scalar
        _test_unary_operator(discr, idiv)
        _test_unary_operator(discr, lambda x: x / scalar)

    # Incremental operations
    def iadd(x, y):
        x += y

    def isub(x, y):
        x -= y

    def imul(x, y):
        x *= y

    def idiv(x, y):
        x /= y

    _test_binary_operator(discr, iadd)
    _test_binary_operator(discr, isub)
    _test_binary_operator(discr, imul)
    _test_binary_operator(discr, idiv)

    # Incremental operators with aliased inputs
    def iadd_aliased(x):
        x += x

    def isub_aliased(x):
        x -= x

    def imul_aliased(x):
        x *= x

    def idiv_aliased(x):
        x /= x

    _test_unary_operator(discr, iadd_aliased)
    _test_unary_operator(discr, isub_aliased)
    _test_unary_operator(discr, imul_aliased)
    _test_unary_operator(discr, idiv_aliased)

    # Binary operators
    _test_binary_operator(discr, lambda x, y: x + y)
    _test_binary_operator(discr, lambda x, y: x - y)
    _test_binary_operator(discr, lambda x, y: x * y)
    _test_binary_operator(discr, lambda x, y: x / y)

    # Binary with aliased inputs
    _test_unary_operator(discr, lambda x: x + x)
    _test_unary_operator(discr, lambda x: x - x)
    _test_unary_operator(discr, lambda x: x * x)
    _test_unary_operator(discr, lambda x: x / x)


def test_interp():
    discr = odl.uniform_discr(0, 1, 3, interp='nearest')
    assert isinstance(discr.interpolation, odl.NearestInterpolation)

    discr = odl.uniform_discr(0, 1, 3, interp='linear')
    assert isinstance(discr.interpolation, odl.LinearInterpolation)

    discr = odl.uniform_discr([0, 0], [1, 1], (3, 3),
                              interp=['nearest', 'linear'])
    assert isinstance(discr.interpolation, odl.PerAxisInterpolation)

    with pytest.raises(ValueError):
        # Too many entries in interp
        discr = odl.uniform_discr(0, 1, 3, interp=['nearest', 'linear'])

    with pytest.raises(ValueError):
        # Too few entries in interp
        discr = odl.uniform_discr([0] * 3, [1] * 3, (3,) * 3,
                                  interp=['nearest', 'linear'])


def test_getitem():
    discr = odl.uniform_discr(0, 1, 3)
    vec = discr.element([1, 2, 3])

    assert all_equal(vec, [1, 2, 3])


def test_getslice():
    discr = odl.uniform_discr(0, 1, 3)
    vec = discr.element([1, 2, 3])

    assert isinstance(vec[:], odl.FnVector)
    assert all_equal(vec[:], [1, 2, 3])

    discr = odl.uniform_discr(0, 1, 3, dtype='complex')
    vec = discr.element([1 + 2j, 2 - 2j, 3])

    assert isinstance(vec[:], odl.FnVector)
    assert all_equal(vec[:], [1 + 2j, 2 - 2j, 3])


def test_setitem():
    discr = odl.uniform_discr(0, 1, 3)
    vec = discr.element([1, 2, 3])
    vec[0] = 4
    vec[1] = 5
    vec[2] = 6

    assert all_equal(vec, [4, 5, 6])


def test_setitem_nd():

    # 1D
    discr = odl.uniform_discr(0, 1, 3)
    vec = discr.element([1, 2, 3])

    vec[:] = [4, 5, 6]
    assert all_equal(vec, [4, 5, 6])

    vec[:] = np.array([3, 2, 1])
    assert all_equal(vec, [3, 2, 1])

    vec[:] = 0
    assert all_equal(vec, [0, 0, 0])

    vec[:] = [1]
    assert all_equal(vec, [1, 1, 1])

    with pytest.raises(ValueError):
        vec[:] = [0, 0]  # bad shape

    with pytest.raises(ValueError):
        vec[:] = [0, 0, 1, 2]  # bad shape

    # 2D
    discr = odl.uniform_discr([0, 0], [1, 1], [3, 2])

    vec = discr.element([[1, 2],
                         [3, 4],
                         [5, 6]])

    vec[:] = [[-1, -2],
              [-3, -4],
              [-5, -6]]
    assert all_equal(vec, [-1, -2, -3, -4, -5, -6])

    arr = np.arange(6, 12).reshape([3, 2])
    vec[:] = arr
    assert all_equal(vec, np.arange(6, 12))

    vec[:] = 0
    assert all_equal(vec, [0] * 6)

    vec[:] = [1]
    assert all_equal(vec, [1] * 6)

    with pytest.raises(ValueError):
        vec[:] = [0, 0]  # bad shape

    with pytest.raises(ValueError):
        vec[:] = [0, 0, 0]  # bad shape

    with pytest.raises(ValueError):
        vec[:] = np.arange(6)[:, np.newaxis]  # bad shape (6, 1)

    with pytest.raises(ValueError):
        arr = np.arange(6, 12).reshape([3, 2])
        vec[:] = arr.T  # bad shape (2, 3)

    # nD
    shape = (3,) * 3 + (4,) * 3
    discr = odl.uniform_discr([0] * 6, [1] * 6, shape)
    size = np.prod(shape)
    vec = discr.element(np.zeros(shape))

    arr = np.arange(size).reshape(shape)

    vec[:] = arr
    assert all_equal(vec, np.arange(size))

    vec[:] = 0
    assert all_equal(vec, np.zeros(size))

    vec[:] = [1]
    assert all_equal(vec, np.ones(size))

    with pytest.raises(ValueError):
        # Reversed shape -> bad
        vec[:] = np.arange(size).reshape((4,) * 3 + (3,) * 3)


def test_setslice():
    discr = odl.uniform_discr(0, 1, 3)
    vec = discr.element([1, 2, 3])

    vec[:] = [4, 5, 6]
    assert all_equal(vec, [4, 5, 6])


def test_asarray_2d():
    discr_F = odl.uniform_discr([0, 0], [1, 1], [2, 2], order='F')
    vec_F = discr_F.element([[1, 2],
                             [3, 4]])

    # Verify that returned array equals input data
    assert all_equal(vec_F.asarray(), [[1, 2],
                                       [3, 4]])
    # Check order of out array
    assert vec_F.asarray().flags['F_CONTIGUOUS']

    # test out parameter
    out_F = np.asfortranarray(np.empty([2, 2]))
    result_F = vec_F.asarray(out=out_F)
    assert result_F is out_F
    assert all_equal(out_F, [[1, 2],
                             [3, 4]])

    # Try discontinuous
    out_F_wrong = np.asfortranarray(np.empty([2, 2]))[::2, :]
    with pytest.raises(ValueError):
        result_F = vec_F.asarray(out=out_F_wrong)

    # Try wrong shape
    out_F_wrong = np.asfortranarray(np.empty([2, 3]))
    with pytest.raises(ValueError):
        result_F = vec_F.asarray(out=out_F_wrong)

    # Try wrong order
    out_F_wrong = np.empty([2, 2])
    with pytest.raises(ValueError):
        vec_F.asarray(out=out_F_wrong)

    # Also check with C ordering
    discr_C = odl.uniform_discr([0, 0], [1, 1], (2, 2), order='C')
    vec_C = discr_C.element([[1, 2],
                             [3, 4]])

    # Verify that returned array equals input data
    assert all_equal(vec_C.asarray(), [[1, 2],
                                       [3, 4]])

    # Check order of out array
    assert vec_C.asarray().flags['C_CONTIGUOUS']

    # test out parameter
    out_C = np.empty([2, 2])
    result_C = vec_C.asarray(out=out_C)
    assert result_C is out_C
    assert all_equal(out_C, [[1, 2],
                             [3, 4]])

    # Try discontinuous
    out_C_wrong = np.empty([4, 2])[::2, :]
    with pytest.raises(ValueError):
        result_C = vec_C.asarray(out=out_C_wrong)

    # Try wrong shape
    out_C_wrong = np.empty([2, 3])
    with pytest.raises(ValueError):
        result_C = vec_C.asarray(out=out_C_wrong)

    # Try wrong order
    out_C_wrong = np.asfortranarray(np.empty([2, 2]))
    with pytest.raises(ValueError):
        vec_C.asarray(out=out_C_wrong)


def test_transpose():
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], order='F')
    x = discr.element([[1, 2], [3, 4]])
    y = discr.element([[5, 6], [7, 8]])

    assert isinstance(x.T, odl.Operator)
    assert x.T.is_linear

    assert x.T(y) == x.inner(y)
    assert x.T.T == x
    assert all_equal(x.T.adjoint(1.0), x)


def test_cell_sides():
    # Non-degenerated case, should be same as cell size
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    vec = discr.element()

    assert all_equal(discr.cell_sides, [0.5] * 2)
    assert all_equal(vec.cell_sides, [0.5] * 2)

    # Degenerated case, uses interval size in 1-point dimensions
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 1])
    vec = discr.element()

    assert all_equal(discr.cell_sides, [0.5, 1])
    assert all_equal(vec.cell_sides, [0.5, 1])


def test_cell_volume():
    # Non-degenerated case
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    vec = discr.element()

    assert discr.cell_volume == 0.25
    assert vec.cell_volume == 0.25

    # Degenerated case, uses interval size in 1-point dimensions
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 1])
    vec = discr.element()

    assert discr.cell_volume == 0.5
    assert vec.cell_volume == 0.5


def test_astype():

    rdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float64')
    cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex128')
    rdiscr_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float32')
    cdiscr_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex64')

    # Real
    assert rdiscr.astype('float32') == rdiscr_s
    assert rdiscr.astype('float64') is rdiscr
    assert rdiscr._real_space is rdiscr
    assert rdiscr.astype('complex64') == cdiscr_s
    assert rdiscr.astype('complex128') == cdiscr
    assert rdiscr._complex_space == cdiscr

    # Complex
    assert cdiscr.astype('complex64') == cdiscr_s
    assert cdiscr.astype('complex128') is cdiscr
    assert cdiscr._complex_space is cdiscr
    assert cdiscr.astype('float32') == rdiscr_s
    assert cdiscr.astype('float64') == rdiscr
    assert cdiscr._real_space == rdiscr


def test_ufunc(impl, ufunc):
    space = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl)
    name, n_args, n_out, _ = ufunc
    if (np.issubsctype(space.dtype, np.floating) and
            name in ['bitwise_and',
                     'bitwise_or',
                     'bitwise_xor',
                     'invert',
                     'left_shift',
                     'right_shift']):
        # Skip integer only methods if floating point type
        return

    # Get the ufunc from numpy as reference
    ufunc = getattr(np, name)

    # Create some data
    arrays, vectors = example_vectors(space, n_args + n_out)
    in_arrays = arrays[:n_args]
    out_arrays = arrays[n_args:]
    data_vector = vectors[0]
    in_vectors = vectors[1:n_args]
    out_vectors = vectors[n_args:]

    # Verify type
    assert isinstance(data_vector.ufunc,
                      odl.util.ufuncs.DiscreteLpUFuncs)

    # Out of place:
    np_result = ufunc(*in_arrays)
    vec_fun = getattr(data_vector.ufunc, name)
    odl_result = vec_fun(*in_vectors)
    assert all_almost_equal(np_result, odl_result)

    # Test type of output
    if n_out == 1:
        assert isinstance(odl_result, space.element_type)
    elif n_out > 1:
        for i in range(n_out):
            assert isinstance(odl_result[i], space.element_type)

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

    # Test out of place with np data
    np_result = ufunc(*in_arrays)
    vec_fun = getattr(data_vector.ufunc, name)
    odl_result = vec_fun(*in_arrays[1:])
    assert all_almost_equal(np_result, odl_result)

    # Test type of output
    if n_out == 1:
        assert isinstance(odl_result, space.element_type)
    elif n_out > 1:
        for i in range(n_out):
            assert isinstance(odl_result[i], space.element_type)


def test_real_imag():

    # Get real and imag
    cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=complex)
    rdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=float)

    x = cdiscr.element([[1 - 1j, 2 - 2j], [3 - 3j, 4 - 4j]])
    assert x.real in rdiscr
    assert all_equal(x.real, [1, 2, 3, 4])
    assert x.imag in rdiscr
    assert all_equal(x.imag, [-1, -2, -3, -4])

    # Set with different data types and shapes
    newreal = rdiscr.element([[2, 3], [4, 5]])
    x.real = newreal
    assert all_equal(x.real, [2, 3, 4, 5])
    newreal = [[3, 4], [5, 6]]
    x.real = newreal
    assert all_equal(x.real, [3, 4, 5, 6])
    newreal = [4, 5, 6, 7]
    x.real = newreal
    assert all_equal(x.real, [4, 5, 6, 7])
    newreal = 0
    x.real = newreal
    assert all_equal(x.real, [0, 0, 0, 0])

    newimag = rdiscr.element([-2, -3, -4, -5])
    x.imag = newimag
    assert all_equal(x.imag, [-2, -3, -4, -5])
    newimag = [[-3, -4], [-5, -6]]
    x.imag = newimag
    assert all_equal(x.imag, [-3, -4, -5, -6])
    newimag = [-4, -5, -6, -7]
    x.imag = newimag
    assert all_equal(x.imag, [-4, -5, -6, -7])
    newimag = -1
    x.imag = newimag
    assert all_equal(x.imag, [-1, -1, -1, -1])

    # 'F' ordering
    cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=complex,
                               order='F')

    x = cdiscr.element()
    newreal = [[3, 4], [5, 6]]
    x.real = newreal
    assert all_equal(x.real, [3, 5, 4, 6])  # flattened in 'F' order
    newreal = [4, 5, 6, 7]
    x.real = newreal
    assert all_equal(x.real, [4, 5, 6, 7])


def test_reduction(impl, reduction):
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl)

    name, _ = reduction

    ufunc = getattr(np, name)

    # Create some data
    x_arr, x = example_vectors(space, 1)
    assert almost_equal(ufunc(x_arr), getattr(x.ufunc, name)())


def test_norm_interval(exponent):
    # Test the function f(x) = x^2 on the interval (0, 1). Its
    # L^p-norm is (1 + 2*p)^(-1/p) for finite p and 1 for p=inf
    p = exponent
    fspace = odl.FunctionSpace(odl.Interval(0, 1))
    lpdiscr = odl.uniform_discr_fromspace(fspace, 10, exponent=p)

    testfunc = fspace.element(lambda x: x ** 2)
    discr_testfunc = lpdiscr.element(testfunc)

    if p == float('inf'):
        assert discr_testfunc.norm() <= 1  # Max at boundary not hit
    else:
        true_norm = (1 + 2 * p) ** (-1 / p)
        assert almost_equal(discr_testfunc.norm(), true_norm, places=2)


def test_norm_rectangle(exponent):
    # Test the function f(x) = x_0^2 * x_1^3 on (0, 1) x (-1, 1). Its
    # L^p-norm is ((1 + 2*p) * (1 + 3 * p) / 2)^(-1/p) for finite p
    # and 1 for p=inf
    p = exponent
    fspace = odl.FunctionSpace(odl.Rectangle([0, -1], [1, 1]))
    lpdiscr = odl.uniform_discr_fromspace(fspace, (20, 30), exponent=p)

    testfunc = fspace.element(lambda x: x[0] ** 2 * x[1] ** 3)
    discr_testfunc = lpdiscr.element(testfunc)

    if p == float('inf'):
        assert discr_testfunc.norm() <= 1  # Max at boundary not hit
    else:
        true_norm = ((1 + 2 * p) * (1 + 3 * p) / 2) ** (-1 / p)
        assert almost_equal(discr_testfunc.norm(), true_norm, places=2)


def test_norm_rectangle_boundary(impl, exponent):
    # Check the constant function 1 in different situations regarding the
    # placement of the outermost grid points.

    if exponent == float('inf'):
        pytest.xfail('inf-norm not implemented in CUDA')

    rect = odl.Rectangle([-1, -2], [1, 2])

    # Standard case
    discr = odl.uniform_discr_fromspace(odl.FunctionSpace(rect), (4, 8),
                                        impl=impl, exponent=exponent)
    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))

    # Nodes on the boundary (everywhere)
    discr = odl.uniform_discr_fromspace(
        odl.FunctionSpace(rect), (4, 8), exponent=exponent,
        impl=impl, nodes_on_bdry=True)

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))

    # Nodes on the boundary (selective)
    discr = odl.uniform_discr_fromspace(
        odl.FunctionSpace(rect), (4, 8), exponent=exponent,
        impl=impl, nodes_on_bdry=((False, True), False))

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))

    discr = odl.uniform_discr_fromspace(
        odl.FunctionSpace(rect), (4, 8), exponent=exponent,
        impl=impl, nodes_on_bdry=(False, (True, False)))

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))

    # Completely arbitrary boundary
    grid = odl.RegularGrid([0, 0], [1, 1], (4, 4))
    part = odl.RectPartition(rect, grid)
    weight = 1.0 if exponent == float('inf') else part.cell_volume
    dspace = odl.Rn(part.size, exponent=exponent, weight=weight)
    discr = DiscreteLp(odl.FunctionSpace(rect), part, dspace,
                       impl=impl, exponent=exponent)

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
