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
import numpy as np

import odl

from odl.util.testutils import all_equal, skip_if_no_cuda


# TODO: element from function - waiting for vectorization


# Pytest fixture


# Simply modify exp_params to modify the fixture
exp_params = [2.0, 1.0, float('inf'), 0.5, 1.5]
exp_ids = [' p = {} '.format(p) for p in exp_params]
exp_fixture = pytest.fixture(scope="module", ids=exp_ids, params=exp_params)


@exp_fixture
def exponent(request):
    return request.param


def test_init(exponent):
    # Validate that the different init patterns work and do not crash.
    space = odl.FunctionSpace(odl.Interval(0, 1))
    grid = odl.uniform_sampling(space.domain, 10)
    rn = odl.Rn(10, exponent=exponent)
    odl.DiscreteLp(space, grid, rn, exponent=exponent)

    # Normal discretization of unit interval with complex
    complex_space = odl.FunctionSpace(odl.Interval(0, 1),
                                      field=odl.ComplexNumbers())
    cn = odl.Cn(10, exponent=exponent)
    odl.DiscreteLp(complex_space, grid, cn, exponent=exponent)

    # Real space should not work with complex
    with pytest.raises(ValueError):
        odl.DiscreteLp(space, grid, cn)

    # Complex space should not work with reals
    with pytest.raises(ValueError):
        odl.DiscreteLp(complex_space, grid, rn)

    # Wrong size of underlying space
    rn_wrong_size = odl.Rn(20)
    with pytest.raises(ValueError):
        odl.DiscreteLp(space, grid, rn_wrong_size)


@skip_if_no_cuda
def test_init_cuda(exponent):
    # Normal discretization of unit interval
    space = odl.FunctionSpace(odl.Interval(0, 1))
    grid = odl.uniform_sampling(space.domain, 10)
    rn = odl.CudaRn(10, exponent=exponent)
    odl.DiscreteLp(space, grid, rn, exponent=exponent)


def test_factory(exponent):
    space = odl.FunctionSpace(odl.Interval(0, 1))
    discr = odl.uniform_discr(space, 10, impl='numpy', exponent=exponent)

    assert isinstance(discr.dspace, odl.Rn)
    assert discr.dspace.exponent == exponent

    # Complex
    space = odl.FunctionSpace(odl.Interval(0, 1), field=odl.ComplexNumbers())
    discr = odl.uniform_discr(space, 10, impl='numpy', exponent=exponent)

    assert isinstance(discr.dspace, odl.Cn)
    assert discr.dspace.exponent == exponent


@skip_if_no_cuda
def test_factory_cuda(exponent):
    space = odl.FunctionSpace(odl.Interval(0, 1))
    discr = odl.uniform_discr(space, 10, impl='cuda', exponent=exponent)
    assert isinstance(discr.dspace, odl.CudaRn)
    assert discr.dspace.exponent == exponent

    # Cuda currently does not support complex numbers, check error
    space = odl.FunctionSpace(odl.Interval(0, 1), field=odl.ComplexNumbers())
    with pytest.raises(NotImplementedError):
        odl.uniform_discr(space, 10, impl='cuda')


def test_factory_dtypes():
    real_float_dtypes = [np.float32, np.float64]
    nonfloat_dtypes = [np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64]
    complex_float_dtypes = [np.complex64, np.complex128]

    # Real
    space = odl.FunctionSpace(odl.Interval(0, 1), field=odl.RealNumbers())
    invalid_dtypes = complex_float_dtypes

    for dtype in real_float_dtypes:
        discr = odl.uniform_discr(space, 10, impl='numpy', dtype=dtype)
        assert isinstance(discr.dspace, odl.Rn)
        assert discr.dspace.element().space.dtype == dtype

    for dtype in nonfloat_dtypes:
        discr = odl.uniform_discr(space, 10, impl='numpy', dtype=dtype)
        assert isinstance(discr.dspace, odl.Fn)
        assert discr.dspace.element().space.dtype == dtype

    for dtype in invalid_dtypes:
        with pytest.raises(TypeError):
            odl.uniform_discr(space, 10, impl='numpy', dtype=dtype)

    # Complex
    space = odl.FunctionSpace(odl.Interval(0, 1), field=odl.ComplexNumbers())
    invalid_dtypes = real_float_dtypes + nonfloat_dtypes

    for dtype in complex_float_dtypes:
        discr = odl.uniform_discr(space, 10, impl='numpy', dtype=dtype)
        assert isinstance(discr.dspace, odl.Cn)
        assert discr.dspace.element().space.dtype == dtype

    for dtype in invalid_dtypes:
        with pytest.raises(TypeError):
            odl.uniform_discr(space, 10, impl='numpy', dtype=dtype)


@skip_if_no_cuda
def test_factory_dtypes_cuda():
    real_float_dtypes = [np.float32, np.float64]
    nonfloat_dtypes = [np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64]
    complex_float_dtypes = [np.complex64, np.complex128]

    # Real
    space = odl.FunctionSpace(odl.Interval(0, 1), odl.RealNumbers())
    invalid_dtypes = complex_float_dtypes

    for dtype in real_float_dtypes:
        if dtype not in odl.CUDA_DTYPES:
            with pytest.raises(TypeError):
                odl.uniform_discr(space, 10, impl='cuda', dtype=dtype)
        else:
            discr = odl.uniform_discr(space, 10, impl='cuda', dtype=dtype)
            assert isinstance(discr.dspace, odl.CudaRn)
            assert discr.dspace.element().space.dtype == dtype

    for dtype in nonfloat_dtypes:
        if dtype not in odl.CUDA_DTYPES:
            with pytest.raises(TypeError):
                odl.uniform_discr(space, 10, impl='cuda', dtype=dtype)
        else:
            discr = odl.uniform_discr(space, 10, impl='cuda', dtype=dtype)
            assert isinstance(discr.dspace, odl.CudaFn)
            assert discr.dspace.element().space.dtype == dtype

    for dtype in invalid_dtypes:
        with pytest.raises(TypeError):
            odl.uniform_discr(space, 10, impl='cuda', dtype=dtype)

    # Complex (not implemented)
    space = odl.FunctionSpace(odl.Interval(0, 1), odl.ComplexNumbers())
    invalid_dtypes = real_float_dtypes + nonfloat_dtypes

    for dtype in complex_float_dtypes:
        with pytest.raises(NotImplementedError):
            odl.uniform_discr(space, 10, impl='cuda', dtype=dtype)

    for dtype in invalid_dtypes:
        with pytest.raises(TypeError):
            odl.uniform_discr(space, 10, impl='cuda', dtype=dtype)


def test_factory_nd(exponent):
    # 2d
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    odl.uniform_discr(square_space, (5, 5), exponent=exponent)

    # 3d
    cube_space = odl.FunctionSpace(odl.Cuboid([0, 0, 0], [1, 1, 1]))
    odl.uniform_discr(cube_space, (5, 5, 5), exponent=exponent)

    # nd
    cube10_space = odl.FunctionSpace(odl.IntervalProd([0] * 10, [1] * 10))
    odl.uniform_discr(cube10_space, (5,) * 10, exponent=exponent)


def test_element_1d(exponent):
    space = odl.FunctionSpace(odl.Interval(0, 1))
    discr = odl.uniform_discr(space, 3, impl='numpy', exponent=exponent)
    dspace = odl.Rn(3, exponent=exponent, weight=discr.cell_volume)
    vec = discr.element()
    assert isinstance(vec, odl.DiscreteLpVector)
    assert vec.ntuple in dspace


def test_element_2d(exponent):
    space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(space, (3, 3), impl='numpy', exponent=exponent)
    dspace = odl.Rn(9, exponent=exponent, weight=discr.cell_volume)
    vec = discr.element()
    assert isinstance(vec, odl.DiscreteLpVector)
    assert vec.ntuple in dspace


def test_element_from_array_1d():
    space = odl.FunctionSpace(odl.Interval(0, 1))
    discr = odl.uniform_discr(space, 3, impl='numpy')
    vec = discr.element([1, 2, 3])

    assert isinstance(vec, odl.DiscreteLpVector)
    assert isinstance(vec.ntuple, odl.RnVector)
    assert all_equal(vec.ntuple, [1, 2, 3])


def test_element_from_array_2d():
    # assert orderings work properly with 2d
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(square_space, (2, 2), impl='numpy', order='C')
    vec = discr.element([[1, 2],
                         [3, 4]])

    assert isinstance(vec, odl.DiscreteLpVector)
    assert isinstance(vec.ntuple, odl.RnVector)

    # Check ordering
    assert all_equal(vec.ntuple, [1, 2, 3, 4])

    # Linear creation works as well
    linear_vec = discr.element([1, 2, 3, 4])
    assert all_equal(vec.ntuple, [1, 2, 3, 4])

    # Fortran order
    discr = odl.uniform_discr(square_space, (2, 2), impl='numpy', order='F')
    vec = discr.element([[1, 2],
                         [3, 4]])

    # Check ordering
    assert all_equal(vec.ntuple, [1, 3, 2, 4])

    # Linear creation works aswell
    linear_vec = discr.element([1, 2, 3, 4])
    assert all_equal(linear_vec.ntuple, [1, 2, 3, 4])


def test_element_from_array_2d_shape():
    # Verify that the shape is correctly tested for
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(square_space, (3, 2), impl='numpy', order='C')

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
    discr = odl.uniform_discr(odl.FunctionSpace(odl.Interval(0, 1)), 3)
    vec = discr.zero()

    assert isinstance(vec, odl.DiscreteLpVector)
    assert isinstance(vec.ntuple, odl.RnVector)
    assert all_equal(vec, [0, 0, 0])


def test_getitem():
    discr = odl.uniform_discr(odl.FunctionSpace(odl.Interval(0, 1)), 3)
    vec = discr.element([1, 2, 3])

    assert all_equal(vec, [1, 2, 3])


def test_getslice():
    discr = odl.uniform_discr(odl.FunctionSpace(odl.Interval(0, 1)), 3)
    vec = discr.element([1, 2, 3])

    assert isinstance(vec[:], odl.RnVector)
    assert all_equal(vec[:], [1, 2, 3])

    discr = odl.uniform_discr(
        odl.FunctionSpace(odl.Interval(0, 1), field=odl.ComplexNumbers()),
        3)
    vec = discr.element([1 + 2j, 2 - 2j, 3])

    assert isinstance(vec[:], odl.CnVector)
    assert all_equal(vec[:], [1 + 2j, 2 - 2j, 3])


def test_setitem():
    discr = odl.uniform_discr(odl.FunctionSpace(odl.Interval(0, 1)), 3)
    vec = discr.element([1, 2, 3])
    vec[0] = 4
    vec[1] = 5
    vec[2] = 6

    assert all_equal(vec, [4, 5, 6])


def test_setitem_nd():

    # 1D
    discr = odl.uniform_discr(odl.FunctionSpace(odl.Interval(0, 1)), 3)
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
    discr = odl.uniform_discr(
        odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1])), [3, 2])

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
    cube10_space = odl.FunctionSpace(odl.IntervalProd([0] * 6, [1] * 6))
    shape = (3,) * 3 + (4,) * 3
    discr = odl.uniform_discr(cube10_space, shape)
    ntotal = np.prod(shape)
    vec = discr.element(np.zeros(shape))

    arr = np.arange(ntotal).reshape(shape)

    vec[:] = arr
    assert all_equal(vec, np.arange(ntotal))

    vec[:] = 0
    assert all_equal(vec, np.zeros(ntotal))

    vec[:] = [1]
    assert all_equal(vec, np.ones(ntotal))

    with pytest.raises(ValueError):
        # Reversed shape -> bad
        vec[:] = np.arange(ntotal).reshape((4,) * 3 + (3,) * 3)


def test_setslice():
    discr = odl.uniform_discr(odl.FunctionSpace(odl.Interval(0, 1)), 3)
    vec = discr.element([1, 2, 3])

    vec[:] = [4, 5, 6]
    assert all_equal(vec, [4, 5, 6])


def test_asarray_2d():
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr_F = odl.uniform_discr(square_space, (2, 2), order='F')
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
    discr_C = odl.uniform_discr(square_space, (2, 2), order='C')
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
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(square_space, (2, 2), order='F')
    x = discr.element([[1, 2], [3, 4]])
    y = discr.element([[5, 6], [7, 8]])

    assert isinstance(x.T, odl.Operator)
    assert x.T.is_linear

    assert x.T(y) == x.inner(y)
    assert x.T.T == x
    assert all_equal(x.T.adjoint(1.0), x)


def test_cell_size():
    # Non-degenerated case, should be same as grid stride
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(square_space, (2, 2), order='F')

    assert all_equal(discr.cell_size, [0.5] * 2)

    # Degenerated case, uses interval size in 1-point dimensions
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(square_space, (2, 1), order='F')

    assert all_equal(discr.cell_size, [0.5, 1])


def test_cell_volume():
    # Non-degenerated case
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(square_space, (2, 2), order='F')

    assert discr.cell_volume == 0.25

    # Degenerated case, uses interval size in 1-point dimensions
    square_space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    discr = odl.uniform_discr(square_space, (2, 1), order='F')

    assert discr.cell_volume == 0.5


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
