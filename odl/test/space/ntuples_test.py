# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import numpy as np
import pytest
import operator
import scipy

import odl
from odl import NumpyFn, NumpyFnVector
from odl.operator.operator import Operator
from odl.set.space import LinearSpaceTypeError
from odl.space.npy_ntuples import (
    NumpyFnConstWeighting, NumpyFnArrayWeighting, NumpyFnMatrixWeighting,
    NumpyFnNoWeighting, NumpyFnCustomInner, NumpyFnCustomNorm,
    NumpyFnCustomDist,
    npy_weighted_inner, npy_weighted_norm, npy_weighted_dist)
from odl.util.testutils import (almost_equal, all_almost_equal, all_equal,
                                noise_array, noise_element,
                                noise_elements, simple_fixture)
from odl.util.ufuncs import UFUNCS, REDUCTIONS

from sys import version_info
PYTHON2 = version_info < (3, 0)


# --- Helper functions --- #


def _pos_array(fn):
    """Create an array with positive real entries as weight in `fn`."""
    return np.abs(noise_array(fn)) + 0.1


def _dense_matrix(fn):
    """Create a dense positive definite Hermitian matrix for `fn`."""

    if np.issubdtype(fn.dtype, np.floating):
        mat = np.random.rand(fn.size, fn.size).astype(fn.dtype, copy=False)
    elif np.issubdtype(fn.dtype, np.integer):
        mat = np.random.randint(0, 10, (fn.size, fn.size)).astype(fn.dtype,
                                                                  copy=False)
    elif np.issubdtype(fn.dtype, np.complexfloating):
        mat = (np.random.rand(fn.size, fn.size) +
               1j * np.random.rand(fn.size, fn.size)).astype(fn.dtype,
                                                             copy=False)

    # Make symmetric and positive definite
    return mat + mat.conj().T + fn.size * np.eye(fn.size, dtype=fn.dtype)


def _sparse_matrix(fn):
    """Create a sparse positive definite Hermitian matrix for `fn`."""
    return scipy.sparse.coo_matrix(_dense_matrix(fn))


# --- pytest fixtures --- #


fn = simple_fixture('fn', [odl.rn(10, np.float64), odl.rn(10, np.float32),
                           odl.cn(10, np.complex128), odl.cn(10, np.complex64),
                           odl.rn(100)])
exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])


# ---- Tests of Fn, Rn and Cn ---- #


def test_init():
    # Fn
    NumpyFn(3, int)
    NumpyFn(3, float)
    NumpyFn(3, complex)
    NumpyFn(3, 'S1')

    with pytest.raises(ValueError):
        NumpyFn(3, np.void)

    # Rn
    odl.rn(3, float)

    # Rn only works on reals
    with pytest.raises(ValueError):
        odl.rn(3, complex)
    with pytest.raises(ValueError):
        odl.rn(3, 'S1')
    with pytest.raises(ValueError):
        odl.rn(3, int)

    # Cn
    odl.cn(3, complex)

    # Cn only works on reals
    with pytest.raises(ValueError):
        odl.cn(3, float)
    with pytest.raises(ValueError):
        odl.cn(3, 'S1')

    # Backported int from future fails (not recognized by numpy.dtype())
    # (Python 2 only)
    from builtins import int as future_int
    import sys
    if sys.version_info.major != 3:
        with pytest.raises(TypeError):
            NumpyFn(3, future_int)

    # Init with weights or custom space functions
    const = 1.5
    weight_arr = _pos_array(odl.rn(3, float))
    weight_mat = _dense_matrix(odl.rn(3, float))

    odl.rn(3, weighting=const)
    odl.rn(3, weighting=weight_arr)
    odl.rn(3, weighting=weight_mat)

    # Different exponents
    exponents = [0.5, 1.0, 2.0, 5.0, float('inf')]
    for exponent in exponents:
        odl.cn(3, exponent=exponent)


def test_init_weighting(exponent):
    const = 1.5
    weight_arr = _pos_array(odl.rn(3, float))
    weight_mat = _dense_matrix(odl.rn(3, float))

    spaces = [NumpyFn(3, complex, exponent=exponent, weighting=const),
              NumpyFn(3, complex, exponent=exponent, weighting=weight_arr),
              NumpyFn(3, complex, exponent=exponent, weighting=weight_mat)]
    weightings = [NumpyFnConstWeighting(const, exponent=exponent),
                  NumpyFnArrayWeighting(weight_arr, exponent=exponent),
                  NumpyFnMatrixWeighting(weight_mat, exponent=exponent)]

    for spc, weight in zip(spaces, weightings):
        assert spc.weighting == weight


def test_equals_space(exponent):
    x1 = NumpyFn(3, exponent=exponent)
    x2 = NumpyFn(3, exponent=exponent)
    y = NumpyFn(4, exponent=exponent)

    assert x1 is x1
    assert x1 is not x2
    assert x1 is not y
    assert x1 == x1
    assert x1 == x2
    assert x1 != y
    assert hash(x1) == hash(x2)
    assert hash(x1) != hash(y)


def test_equals_vec(exponent):
    r3 = NumpyFn(3, exponent=exponent)
    r4 = NumpyFn(4, exponent=exponent)
    x1 = r3.element([1, 2, 3])
    x2 = r3.element([1, 2, 3])
    y = r3.element([2, 2, 3])
    z = r4.element([1, 2, 3, 4])

    assert x1 is x1
    assert x1 is not x2
    assert x1 is not y
    assert x1 == x1
    assert x1 == x2
    assert x1 != y
    assert x1 != z


def test_astype():
    rn = odl.rn(3, weighting=1.5)
    cn = odl.cn(3, weighting=1.5)
    rn_s = odl.rn(3, weighting=1.5, dtype='float32')
    cn_s = odl.cn(3, weighting=1.5, dtype='complex64')

    # Real
    assert rn.astype('float32') == rn_s
    assert rn.astype('float64') is rn
    assert rn.real_space is rn
    assert rn.astype('complex64') == cn_s
    assert rn.astype('complex128') == cn
    assert rn.complex_space == cn

    # Complex
    assert cn.astype('complex64') == cn_s
    assert cn.astype('complex128') is cn
    assert cn.real_space == rn
    assert cn.astype('float32') == rn_s
    assert cn.astype('float64') == rn
    assert cn.complex_space is cn


def test_vector_class_init(fn):
    # Test that code runs
    arr = noise_array(fn)

    NumpyFnVector(fn, arr)


def _test_lincomb(fn, a, b):
    # Validate lincomb against the result on host with randomized
    # data and given a,b, contiguous and non-contiguous

    # Unaliased arguments
    [xarr, yarr, zarr], [x, y, z] = noise_elements(fn, 3)
    zarr[:] = a * xarr + b * yarr
    fn.lincomb(a, x, b, y, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # First argument aliased with output
    [xarr, yarr, zarr], [x, y, z] = noise_elements(fn, 3)
    zarr[:] = a * zarr + b * yarr
    fn.lincomb(a, z, b, y, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # Second argument aliased with output
    [xarr, yarr, zarr], [x, y, z] = noise_elements(fn, 3)
    zarr[:] = a * xarr + b * zarr
    fn.lincomb(a, x, b, z, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # Both arguments aliased with each other
    [xarr, yarr, zarr], [x, y, z] = noise_elements(fn, 3)
    zarr[:] = a * xarr + b * xarr
    fn.lincomb(a, x, b, x, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # All aliased
    [xarr, yarr, zarr], [x, y, z] = noise_elements(fn, 3)
    zarr[:] = a * zarr + b * zarr
    fn.lincomb(a, z, b, z, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])


def test_lincomb(fn):
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(fn, a, b)


def test_lincomb_exceptions(fn):
    # Hack to make sure otherfn is different
    otherfn = odl.rn(1) if fn.size != 1 else odl.rn(2)

    otherx = otherfn.zero()
    x, y, z = fn.zero(), fn.zero(), fn.zero()

    with pytest.raises(LinearSpaceTypeError):
        fn.lincomb(1, otherx, 1, y, z)

    with pytest.raises(LinearSpaceTypeError):
        fn.lincomb(1, y, 1, otherx, z)

    with pytest.raises(LinearSpaceTypeError):
        fn.lincomb(1, y, 1, z, otherx)

    with pytest.raises(LinearSpaceTypeError):
        fn.lincomb([], x, 1, y, z)

    with pytest.raises(LinearSpaceTypeError):
        fn.lincomb(1, x, [], y, z)


def test_multiply(fn):
    # space method
    [x_arr, y_arr, out_arr], [x, y, out] = noise_elements(fn, 3)
    out_arr = x_arr * y_arr

    fn.multiply(x, y, out)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])

    # member method
    [x_arr, y_arr, out_arr], [x, y, out] = noise_elements(fn, 3)
    out_arr = x_arr * y_arr

    x.multiply(y, out=out)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])


def test_multiply_exceptions(fn):
    # Hack to make sure otherfn is different
    otherfn = odl.rn(1) if fn.size != 1 else odl.rn(2)

    otherx = otherfn.zero()
    x, y = fn.zero(), fn.zero()

    with pytest.raises(LinearSpaceTypeError):
        fn.multiply(otherx, x, y)

    with pytest.raises(LinearSpaceTypeError):
        fn.multiply(x, otherx, y)

    with pytest.raises(LinearSpaceTypeError):
        fn.multiply(x, y, otherx)


def test_power(fn):

    [x_arr, y_arr], [x, y] = noise_elements(fn, n=2)
    y_pos = fn.element(np.abs(y) + 0.1)
    y_pos_arr = np.abs(y_arr) + 0.1

    # Testing standard positive integer power out-of-place and in-place
    assert all_almost_equal(x ** 2, x_arr ** 2)
    y **= 2
    y_arr **= 2
    assert all_almost_equal(y, y_arr)

    # Real number and negative integer power
    assert all_almost_equal(y_pos ** 1.3, y_pos_arr ** 1.3)
    assert all_almost_equal(y_pos ** (-3), y_pos_arr ** (-3))
    y_pos **= 2.5
    y_pos_arr **= 2.5
    assert all_almost_equal(y_pos, y_pos_arr)

    # Array raised to the power of another array, entry-wise
    assert all_almost_equal(y_pos ** x, y_pos_arr ** x_arr)
    y_pos **= x.real
    y_pos_arr **= x_arr.real
    assert all_almost_equal(y_pos, y_pos_arr)


def test_unary_ops(fn):
    """Verify that the unary operators (`+x` and `-x`) work as expected."""
    for op in [operator.pos, operator.neg]:
        x_arr, x = noise_elements(fn)

        y_arr = op(x_arr)
        y = op(x)

        assert all_almost_equal([x, y], [x_arr, y_arr])


def test_binary_operator_scalar(fn, arithmetic_op):
    """Verify binary operations with scalars.

    Verifies that the statement y=op(x, scalar) gives equivalent results
    to NumPy.
    """

    for scalar in [-31.2, -1, 0, 1, 2.13]:
        x_arr, x = noise_elements(fn)

        # Left op
        if scalar == 0 and arithmetic_op in [operator.truediv,
                                             operator.itruediv]:
            # Check for correct zero division behaviour
            with pytest.raises(ZeroDivisionError):
                y = arithmetic_op(x, scalar)
        else:
            y_arr = arithmetic_op(x_arr, scalar)
            y = arithmetic_op(x, scalar)

            assert all_almost_equal([x, y], [x_arr, y_arr])

        # right op
        x_arr, x = noise_elements(fn)

        y_arr = arithmetic_op(scalar, x_arr)
        y = arithmetic_op(scalar, x)

        assert all_almost_equal([x, y], [x_arr, y_arr])


def test_binary_operator_vector(fn, arithmetic_op):
    """Verify binary operations with vectors.

    Verifies that the statement z=op(x, y) gives equivalent results to NumPy.
    """
    [x_arr, y_arr], [x, y] = noise_elements(fn, 2)

    # non-aliased left
    z_arr = arithmetic_op(x_arr, y_arr)
    z = arithmetic_op(x, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    # non-aliased right
    z_arr = arithmetic_op(y_arr, x_arr)
    z = arithmetic_op(y, x)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    # aliased operation
    z_arr = arithmetic_op(x_arr, x_arr)
    z = arithmetic_op(x, x)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])


def test_binary_operator_array(fn, arithmetic_op):
    """Verify binary operations with vectors and arrays.

    Verifies that the statement z=op(x, y) gives equivalent results to NumPy.
    """
    [x_arr, y_arr], [x, y] = noise_elements(fn, 2)

    # non-aliased left
    y_arr_cpy = y_arr.copy()
    z_arr = arithmetic_op(x_arr, y_arr_cpy)
    z = arithmetic_op(x, y_arr)

    assert isinstance(z, type(x))
    assert z.space == x.space
    assert all_almost_equal([x, y_arr, z], [x_arr, y_arr_cpy, z_arr])

    # non-aliased right
    y_arr_cpy = y_arr.copy()
    z_arr = arithmetic_op(y_arr_cpy, x_arr)
    z = arithmetic_op(y_arr, x)

    if arithmetic_op in [operator.iadd, operator.isub, operator.imul,
                         operator.itruediv]:
        # In place should still be numpy array
        assert isinstance(z, np.ndarray)
        assert all_almost_equal([x, y_arr, z],
                                [x_arr, y_arr_cpy, z_arr])
    else:
        assert isinstance(z, type(x))
        assert z.space == x.space
        assert all_almost_equal([x, y_arr, z],
                                [x_arr, y_arr_cpy, z_arr])


def test_assign(fn):
    x = noise_element(fn)
    y = noise_element(fn)

    y.assign(x)

    assert y == x
    assert y is not x

    # test alignment
    x *= 2
    assert y != x


def test_inner(fn):
    xd = noise_element(fn)
    yd = noise_element(fn)

    correct_inner = np.vdot(yd, xd)
    assert almost_equal(fn.inner(xd, yd), correct_inner)
    assert almost_equal(xd.inner(yd), correct_inner)


def test_inner_exceptions(fn):
    # Hack to make sure otherfn is different
    otherfn = odl.rn(1) if fn.size != 1 else odl.rn(2)

    otherx = otherfn.zero()
    x = fn.zero()

    with pytest.raises(LinearSpaceTypeError):
        fn.inner(otherx, x)

    with pytest.raises(LinearSpaceTypeError):
        fn.inner(x, otherx)


def test_norm(fn):
    xarr, x = noise_elements(fn)

    correct_norm = np.linalg.norm(xarr)

    assert almost_equal(fn.norm(x), correct_norm)
    assert almost_equal(x.norm(), correct_norm)


def test_norm_exceptions(fn):
    # Hack to make sure otherfn is different
    otherfn = odl.rn(1) if fn.size != 1 else odl.rn(2)

    otherx = otherfn.zero()

    with pytest.raises(LinearSpaceTypeError):
        fn.norm(otherx)


def test_pnorm(exponent):
    for fn in (odl.rn(3, exponent=exponent), odl.cn(3, exponent=exponent)):
        xarr, x = noise_elements(fn)
        correct_norm = np.linalg.norm(xarr, ord=exponent)

        assert almost_equal(fn.norm(x), correct_norm)
        assert almost_equal(x.norm(), correct_norm)


def test_dist(fn):
    [xarr, yarr], [x, y] = noise_elements(fn, n=2)

    correct_dist = np.linalg.norm(xarr - yarr)

    assert almost_equal(fn.dist(x, y), correct_dist)
    assert almost_equal(x.dist(y), correct_dist)


def test_dist_exceptions(fn):
    # Hack to make sure otherfn is different
    otherfn = odl.rn(1) if fn.size != 1 else odl.rn(2)

    otherx = otherfn.zero()
    x = fn.zero()

    with pytest.raises(LinearSpaceTypeError):
        fn.dist(otherx, x)

    with pytest.raises(LinearSpaceTypeError):
        fn.dist(x, otherx)


def test_pdist(exponent):
    for fn in (odl.rn(3, exponent=exponent), odl.cn(3, exponent=exponent)):
        [xarr, yarr], [x, y] = noise_elements(fn, n=2)

        correct_dist = np.linalg.norm(xarr - yarr, ord=exponent)

        assert almost_equal(fn.dist(x, y), correct_dist)
        assert almost_equal(x.dist(y), correct_dist)


def test_setitem(fn):
    x = noise_element(fn)

    for index in [0, 1, 2, -1, -2, -3]:
        x[index] = index
        assert almost_equal(x[index], index)


def test_setitem_index_error(fn):
    x = noise_element(fn)

    with pytest.raises(IndexError):
        x[-fn.size - 1] = 0

    with pytest.raises(IndexError):
        x[fn.size] = 0


def _test_getslice(slice):
    # Validate get against python list behaviour
    r6 = odl.rn(6)
    y = [0, 1, 2, 3, 4, 5]
    x = r6.element(y)

    assert all_equal(x[slice].data, y[slice])


def test_getslice():
    # Tests getting all combinations of slices
    steps = [None, -2, -1, 1, 2]
    starts = [None, -1, -3, 0, 2, 5]
    ends = [None, -1, -3, 0, 2, 5]

    for start in starts:
        for end in ends:
            for step in steps:
                _test_getslice(slice(start, end, step))


def _test_setslice(slice):
    # Validate set against python list behaviour
    r6 = odl.rn(6)
    z = [7, 8, 9, 10, 11, 10]
    y = [0, 1, 2, 3, 4, 5]
    x = r6.element(y)

    x[slice] = z[slice]
    y[slice] = z[slice]
    assert all_equal(x, y)


def test_setslice():
    # Tests a range of combination of slices
    steps = [None, -2, -1, 1, 2]
    starts = [None, -1, -3, 0, 2, 5, 10]
    ends = [None, -1, -3, 0, 2, 5, 10]

    for start in starts:
        for end in ends:
            for step in steps:
                _test_setslice(slice(start, end, step))


def test_transpose(fn):
    x = noise_element(fn)
    y = noise_element(fn)

    # Assert linear operator
    assert isinstance(x.T, Operator)
    assert x.T.is_linear

    # Check result
    assert almost_equal(x.T(y), y.inner(x))
    assert all_equal(x.T.adjoint(1.0), x)

    # x.T.T returns self
    assert x.T.T == x


def test_setslice_index_error(fn):
    xd = fn.element()
    n = fn.size

    # Bad slice
    with pytest.raises(ValueError):
        xd[n:n + 3] = [1, 2, 3]

    # Bad size of rhs
    with pytest.raises(ValueError):
        xd[:] = []

    with pytest.raises(ValueError):
        xd[:] = np.zeros(n - 1)

    with pytest.raises(ValueError):
        xd[:] = np.zeros(n + 1)


def test_multiply_by_scalar(fn):
    # Verify that multiplying with numpy scalars does not change the type
    # of the array
    x = fn.zero()
    assert x * 1.0 in fn
    assert x * np.float32(1.0) in fn
    assert 1.0 * x in fn
    assert np.float32(1.0) * x in fn


def test_member_copy(fn):
    x = noise_element(fn)

    y = x.copy()

    assert x == y
    assert y is not x

    # test not aliased
    x *= 2
    assert x != y


def test_python_copy(fn):
    import copy

    x = noise_element(fn)

    y = copy.copy(x)

    assert x == y
    assert y is not x

    # test not aliased
    x *= 2
    assert x != y

    z = copy.deepcopy(x)

    assert x == z
    assert z is not x

    # test not aliased
    x *= 2
    assert x != z


# Vector property tests

def test_space(fn):
    x = fn.element()
    assert x.space is fn


def test_ndim(fn):
    x = fn.element()
    assert x.ndim == 1


def test_dtype(fn):
    x = fn.element()
    assert x.dtype == fn.dtype


def test_size(fn):
    x = fn.element()
    assert x.size == fn.size


def test_shape(fn):
    x = fn.element()
    assert x.shape == (x.size,)


def test_itemsize(fn):
    x = fn.element()
    assert x.itemsize == fn.dtype.itemsize


def test_nbytes(fn):
    x = fn.element()
    assert x.nbytes == x.itemsize * x.size


# Type conversion tests


def test_scalar_method():
    # Too large space
    space = odl.rn(2)
    element = space.one()

    with pytest.raises(TypeError):
        int(element)
    with pytest.raises(TypeError):
        float(element)
    with pytest.raises(TypeError):
        complex(element)
    if PYTHON2:
        with pytest.raises(TypeError):
            long(element)

    # Size 1 real space
    space = odl.rn(1)
    value = 1.5
    element = space.element(value)

    assert int(element) == int(value)
    assert float(element) == float(value)
    assert complex(element) == complex(value)
    if PYTHON2:
        assert long(element) == long(value)

    # Size 1 complex space
    space = odl.cn(1)
    value = 1.5 + 0.5j
    element = space.element(value)

    assert complex(element) == complex(value)


def test_array_method(fn):
    # Verify that the __array__ method works
    x = fn.zero()

    arr = x.__array__()

    assert isinstance(arr, np.ndarray)
    assert all_equal(arr, np.zeros(x.size))


def test_array_wrap_method(fn):
    # Verify that the __array_wrap__ method works. This enables numpy ufuncs
    # on vectors
    x_h, x = noise_elements(fn)
    y_h = np.sin(x_h)
    y = np.sin(x)

    assert all_equal(y, y_h)
    assert y in fn


def test_conj(fn):
    xarr, x = noise_elements(fn)
    xconj = x.conj()
    assert all_equal(xconj, xarr.conj())
    y = x.copy()
    xconj = x.conj(out=y)
    assert xconj is y
    assert all_equal(y, xarr.conj())


# --- Weighting tests --- #


def test_matrix_init(fn, exponent):
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)

    # Just test if the code runs and if repr returns something
    weighting_de = NumpyFnMatrixWeighting(dense_mat, exponent=exponent)
    assert repr(weighting_de) != ''
    if exponent in (1.0, 2.0, float('inf')):
        weighting_sp = NumpyFnMatrixWeighting(sparse_mat, exponent=exponent)
        assert repr(weighting_sp) != ''
    else:
        with pytest.raises(NotImplementedError):
            NumpyFnMatrixWeighting(sparse_mat, exponent=exponent)

    nonsquare_mat = np.eye(10, 5)
    with pytest.raises(ValueError):
        NumpyFnMatrixWeighting(nonsquare_mat)


def test_matrix_matrix():
    fn = odl.rn(5)
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)

    w_sparse = NumpyFnMatrixWeighting(sparse_mat)
    w_dense = NumpyFnMatrixWeighting(dense_mat)

    assert isinstance(w_sparse.matrix, scipy.sparse.spmatrix)
    assert isinstance(w_dense.matrix, np.ndarray)


def test_matrix_is_valid():
    fn = odl.rn(5)
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)
    bad_mat = np.eye(5)
    bad_mat[0, 0] = 0

    w_sparse = NumpyFnMatrixWeighting(sparse_mat)
    w_dense = NumpyFnMatrixWeighting(dense_mat)
    w_bad = NumpyFnMatrixWeighting(bad_mat)

    with pytest.raises(NotImplementedError):
        w_sparse.is_valid()

    assert w_dense.is_valid()
    assert not w_bad.is_valid()


def test_matrix_equals(fn, exponent):
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(fn)
    different_dense_mat = dense_mat.copy()
    different_dense_mat[0, 0] -= 1

    if exponent in (1.0, 2.0, float('inf')):
        w_sparse = NumpyFnMatrixWeighting(sparse_mat, exponent=exponent)
        w_sparse2 = NumpyFnMatrixWeighting(sparse_mat, exponent=exponent)
    w_sparse_as_dense = NumpyFnMatrixWeighting(sparse_mat_as_dense,
                                               exponent=exponent)
    w_dense = NumpyFnMatrixWeighting(dense_mat, exponent=exponent)
    w_dense2 = NumpyFnMatrixWeighting(dense_mat, exponent=exponent)
    w_different_mat = NumpyFnMatrixWeighting(different_dense_mat,
                                             exponent=exponent)
    diff_exp = exponent + 1 if np.isfinite(exponent) else 1
    w_different_exp = NumpyFnMatrixWeighting(dense_mat, exponent=diff_exp)

    # Identical objects -> True
    assert w_dense == w_dense
    # Identical matrices -> True
    if exponent in (1.0, 2.0, float('inf')):
        assert w_sparse == w_sparse2
    assert w_dense == w_dense2
    # Equivalent but not identical matrices -> False
    if exponent in (1.0, 2.0, float('inf')):
        assert w_sparse != w_sparse_as_dense
        assert w_sparse_as_dense != w_sparse
    # Not equivalent -> False
    assert w_dense != w_different_mat
    # Different exponents -> False
    assert w_dense != w_different_exp


def test_matrix_equiv():
    fn = odl.rn(5)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(fn)
    different_dense_mat = dense_mat.copy()
    different_dense_mat[0, 0] = -10

    w_sparse = NumpyFnMatrixWeighting(sparse_mat)
    w_sparse2 = NumpyFnMatrixWeighting(sparse_mat)
    w_sparse_as_dense = NumpyFnMatrixWeighting(sparse_mat_as_dense)
    w_dense = NumpyFnMatrixWeighting(dense_mat)
    w_dense_copy = NumpyFnMatrixWeighting(dense_mat.copy())
    w_different_dense = NumpyFnMatrixWeighting(different_dense_mat)

    # Equal -> True
    assert w_sparse.equiv(w_sparse)
    assert w_sparse.equiv(w_sparse2)
    # Equivalent matrices -> True
    assert w_sparse.equiv(w_sparse_as_dense)
    assert w_dense.equiv(w_dense_copy)
    # Different matrices -> False
    assert not w_dense.equiv(w_different_dense)

    # Test shortcuts
    sparse_eye = scipy.sparse.eye(5)
    w_eye = NumpyFnMatrixWeighting(sparse_eye)
    w_dense_eye = NumpyFnMatrixWeighting(sparse_eye.todense())
    w_eye_vec = NumpyFnArrayWeighting(np.ones(5))

    w_eye_wrong_exp = NumpyFnMatrixWeighting(sparse_eye, exponent=1)

    sparse_smaller_eye = scipy.sparse.eye(4)
    w_smaller_eye = NumpyFnMatrixWeighting(sparse_smaller_eye)

    sparse_shifted_eye = scipy.sparse.eye(5, k=1)
    w_shifted_eye = NumpyFnMatrixWeighting(sparse_shifted_eye)

    sparse_almost_eye = scipy.sparse.dia_matrix((np.ones(4), [0]), (5, 5))
    w_almost_eye = NumpyFnMatrixWeighting(sparse_almost_eye)

    assert w_eye.equiv(w_dense_eye)
    assert w_dense_eye.equiv(w_eye)
    assert w_eye.equiv(w_eye_vec)
    assert not w_eye.equiv(w_eye_wrong_exp)
    assert not w_eye.equiv(w_smaller_eye)
    assert not w_eye.equiv(w_shifted_eye)
    assert not w_smaller_eye.equiv(w_shifted_eye)
    assert not w_eye.equiv(w_almost_eye)

    # Bogus input
    assert not w_eye.equiv(True)
    assert not w_eye.equiv(object)
    assert not w_eye.equiv(None)


def test_matrix_inner(fn):
    [xarr, yarr], [x, y] = noise_elements(fn, 2)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = np.asarray(sparse_mat.todense())
    dense_mat = _dense_matrix(fn)

    true_inner_sparse = np.vdot(yarr, np.dot(sparse_mat_as_dense, xarr))
    true_inner_dense = np.vdot(yarr, np.dot(dense_mat, xarr))

    w_sparse = NumpyFnMatrixWeighting(sparse_mat)
    w_dense = NumpyFnMatrixWeighting(dense_mat)
    assert almost_equal(w_sparse.inner(x, y), true_inner_sparse)
    assert almost_equal(w_dense.inner(x, y), true_inner_dense)

    # With free functions
    w_sparse_inner = npy_weighted_inner(sparse_mat)
    w_dense_inner = npy_weighted_inner(dense_mat)
    assert almost_equal(w_sparse_inner(x, y), true_inner_sparse)
    assert almost_equal(w_dense_inner(x, y), true_inner_dense)

    # Exponent != 2 -> no inner
    w_dense = NumpyFnMatrixWeighting(dense_mat, exponent=1)
    with pytest.raises(NotImplementedError):
        w_dense.inner(x, y)


def test_matrix_norm(fn, exponent):
    xarr, x = noise_elements(fn)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = np.asarray(sparse_mat.todense())
    dense_mat = _dense_matrix(fn)

    # Compute true matrix-weighted norm
    if exponent == 1.0:  # ||x||_{A,1} = ||Ax||_1
        true_norm_sparse = np.linalg.norm(np.dot(sparse_mat_as_dense, xarr),
                                          ord=exponent)
        true_norm_dense = np.linalg.norm(np.dot(dense_mat, xarr),
                                         ord=exponent)
    elif exponent == 2.0:  # ||x||_{A,2} = sqrt(<x, Ax>)
        true_norm_sparse = np.sqrt(
            np.vdot(xarr, np.dot(sparse_mat_as_dense, xarr)))
        true_norm_dense = np.sqrt(np.vdot(xarr, np.dot(dense_mat, xarr)))
    elif exponent == float('inf'):  # ||x||_{A,inf} = ||Ax||_inf
        true_norm_sparse = np.linalg.norm(sparse_mat_as_dense.dot(xarr),
                                          ord=exponent)
        true_norm_dense = np.linalg.norm(dense_mat.dot(xarr), ord=exponent)
    else:  # ||x||_{A,p} = ||A^{1/p} x||_p
        # Calculate matrix power
        eigval, eigvec = scipy.linalg.eigh(dense_mat)
        eigval **= 1.0 / exponent
        mat_pow = (eigval * eigvec).dot(eigvec.conj().T)
        true_norm_dense = np.linalg.norm(np.dot(mat_pow, xarr), ord=exponent)

    # Test weighting
    if exponent in (1.0, 2.0, float('inf')):
        w_sparse = NumpyFnMatrixWeighting(sparse_mat, exponent=exponent)
        assert almost_equal(w_sparse.norm(x), true_norm_sparse)

    w_dense = NumpyFnMatrixWeighting(dense_mat, exponent=exponent)
    assert almost_equal(w_dense.norm(x), true_norm_dense)

    # With free functions
    if exponent not in (1.0, 2.0, float('inf')):
        with pytest.raises(NotImplementedError):
            npy_weighted_norm(sparse_mat, exponent=exponent)
    else:
        w_sparse_norm = npy_weighted_norm(sparse_mat, exponent=exponent)
        assert almost_equal(w_sparse_norm(x), true_norm_sparse)

    w_dense_norm = npy_weighted_norm(dense_mat, exponent=exponent)
    assert almost_equal(w_dense_norm(x), true_norm_dense)


def test_matrix_dist(fn, exponent):
    [xarr, yarr], [x, y] = noise_elements(fn, n=2)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = np.asarray(sparse_mat.todense())
    dense_mat = _dense_matrix(fn)

    if exponent == 1.0:  # d(x, y)_{A,1} = ||A(x-y)||_1
        true_dist_sparse = np.linalg.norm(
            np.dot(sparse_mat_as_dense, xarr - yarr), ord=exponent)
        true_dist_dense = np.linalg.norm(
            np.dot(dense_mat, xarr - yarr), ord=exponent)
    elif exponent == 2.0:  # d(x, y)_{A,2} = sqrt(<x-y, A(x-y)>)
        true_dist_sparse = np.sqrt(
            np.vdot(xarr - yarr, np.dot(sparse_mat_as_dense, xarr - yarr)))
        true_dist_dense = np.sqrt(
            np.vdot(xarr - yarr, np.dot(dense_mat, xarr - yarr)))
    elif exponent == float('inf'):  # d(x, y)_{A,inf} = ||A(x-y)||_inf
        true_dist_sparse = np.linalg.norm(sparse_mat_as_dense.dot(xarr - yarr),
                                          ord=exponent)
        true_dist_dense = np.linalg.norm(dense_mat.dot(xarr - yarr),
                                         ord=exponent)
    else:  # d(x, y)_{A,p} = ||A^{1/p} (x-y)||_p
        # Calculate matrix power
        eigval, eigvec = scipy.linalg.eigh(dense_mat)
        eigval **= 1.0 / exponent
        mat_pow = (eigval * eigvec).dot(eigvec.conj().T)
        true_dist_dense = np.linalg.norm(np.dot(mat_pow, xarr - yarr),
                                         ord=exponent)

    if exponent in (1.0, 2.0, float('inf')):
        w_sparse = NumpyFnMatrixWeighting(sparse_mat, exponent=exponent)
        assert almost_equal(w_sparse.dist(x, y), true_dist_sparse)

    w_dense = NumpyFnMatrixWeighting(dense_mat, exponent=exponent)
    assert almost_equal(w_dense.dist(x, y), true_dist_dense)

    # With free functions
    if exponent in (1.0, 2.0, float('inf')):
        w_sparse_dist = npy_weighted_dist(sparse_mat, exponent=exponent)
        assert almost_equal(w_sparse_dist(x, y), true_dist_sparse)

    w_dense_dist = npy_weighted_dist(dense_mat, exponent=exponent)
    assert almost_equal(w_dense_dist(x, y), true_dist_dense)


def test_array_init(exponent):
    rn = odl.rn(5)
    weight_arr = _pos_array(rn)

    NumpyFnArrayWeighting(weight_arr, exponent=exponent)
    NumpyFnArrayWeighting(rn.element(weight_arr), exponent=exponent)


def test_array_array():
    rn = odl.rn(5)
    weight_arr = _pos_array(rn)
    weight_elem = rn.element(weight_arr)

    weighting_arr = NumpyFnArrayWeighting(weight_arr)
    weighting_elem = NumpyFnArrayWeighting(weight_elem)

    assert isinstance(weighting_arr.array, np.ndarray)
    assert isinstance(weighting_elem.array, NumpyFnVector)


def test_array_is_valid():
    rn = odl.rn(5)
    weight_arr = _pos_array(rn)
    weighting_arr = NumpyFnArrayWeighting(weight_arr)

    assert weighting_arr.is_valid()

    # Invalid
    weight_arr[0] = 0
    weighting_arr = NumpyFnArrayWeighting(weight_arr)
    assert not weighting_arr.is_valid()


def test_array_equals():
    rn = odl.rn(5)
    weight_arr = _pos_array(rn)
    weight_elem = rn.element(weight_arr)

    weighting_arr = NumpyFnArrayWeighting(weight_arr)
    weighting_arr2 = NumpyFnArrayWeighting(weight_arr)
    weighting_elem = NumpyFnArrayWeighting(weight_elem)
    weighting_elem2 = NumpyFnArrayWeighting(weight_elem)
    weighting_other_vec = NumpyFnArrayWeighting(weight_arr - 1)
    weighting_other_exp = NumpyFnArrayWeighting(weight_arr - 1, exponent=1)

    assert weighting_arr == weighting_arr2
    assert weighting_arr != weighting_elem
    assert weighting_elem == weighting_elem2
    assert weighting_arr != weighting_other_vec
    assert weighting_arr != weighting_other_exp


def test_array_equiv():
    rn = odl.rn(5)
    weight_arr = _pos_array(rn)
    weight_elem = rn.element(weight_arr)
    diag_mat = weight_arr * np.eye(5)
    different_vec = weight_arr - 1

    w_vec = NumpyFnArrayWeighting(weight_arr)
    w_elem = NumpyFnArrayWeighting(weight_elem)
    w_diag_mat = NumpyFnMatrixWeighting(diag_mat)
    w_different_vec = NumpyFnArrayWeighting(different_vec)

    # Equal -> True
    assert w_vec.equiv(w_vec)
    assert w_vec.equiv(w_elem)
    # Equivalent matrix -> True
    assert w_vec.equiv(w_diag_mat)
    # Different vector -> False
    assert not w_vec.equiv(w_different_vec)

    # Test shortcuts
    const_vec = np.ones(5) * 1.5

    w_vec = NumpyFnArrayWeighting(const_vec)
    w_const = NumpyFnConstWeighting(1.5)
    w_wrong_const = NumpyFnConstWeighting(1)
    w_wrong_exp = NumpyFnConstWeighting(1.5, exponent=1)

    assert w_vec.equiv(w_const)
    assert not w_vec.equiv(w_wrong_const)
    assert not w_vec.equiv(w_wrong_exp)

    # Bogus input
    assert not w_vec.equiv(True)
    assert not w_vec.equiv(object)
    assert not w_vec.equiv(None)


def test_array_inner(fn):
    [xarr, yarr], [x, y] = noise_elements(fn, 2)

    weight_arr = _pos_array(fn)
    weighting_arr = NumpyFnArrayWeighting(weight_arr)

    true_inner = np.vdot(yarr, xarr * weight_arr)

    assert almost_equal(weighting_arr.inner(x, y), true_inner)

    # With free function
    inner_vec = npy_weighted_inner(weight_arr)

    assert almost_equal(inner_vec(x, y), true_inner)

    # Exponent != 2 -> no inner product, should raise
    with pytest.raises(NotImplementedError):
        NumpyFnArrayWeighting(weight_arr, exponent=1.0).inner(x, y)


def test_array_norm(fn, exponent):
    xarr, x = noise_elements(fn)

    weight_arr = _pos_array(fn)
    weighting_arr = NumpyFnArrayWeighting(weight_arr, exponent=exponent)

    if exponent == float('inf'):
        true_norm = np.linalg.norm(weight_arr * xarr, ord=float('inf'))
    else:
        true_norm = np.linalg.norm(weight_arr ** (1 / exponent) * xarr,
                                   ord=exponent)

    assert almost_equal(weighting_arr.norm(x), true_norm)

    # With free function
    pnorm_vec = npy_weighted_norm(weight_arr, exponent=exponent)
    assert almost_equal(pnorm_vec(x), true_norm)


def test_array_dist(fn, exponent):
    [xarr, yarr], [x, y] = noise_elements(fn, n=2)

    weight_arr = _pos_array(fn)
    weighting_arr = NumpyFnArrayWeighting(weight_arr, exponent=exponent)

    if exponent == float('inf'):
        true_dist = np.linalg.norm(
            weight_arr * (xarr - yarr), ord=float('inf'))
    else:
        true_dist = np.linalg.norm(
            weight_arr ** (1 / exponent) * (xarr - yarr), ord=exponent)

    assert almost_equal(weighting_arr.dist(x, y), true_dist)

    # With free function
    pdist_vec = npy_weighted_dist(weight_arr, exponent=exponent)
    assert almost_equal(pdist_vec(x, y), true_dist)


def test_constant_init(exponent):
    constant = 1.5

    # Just test if the code runs
    NumpyFnConstWeighting(constant, exponent=exponent)

    with pytest.raises(ValueError):
        NumpyFnConstWeighting(0)
    with pytest.raises(ValueError):
        NumpyFnConstWeighting(-1)
    with pytest.raises(ValueError):
        NumpyFnConstWeighting(float('inf'))


def test_constant_equals():
    n = 10
    constant = 1.5

    w_const = NumpyFnConstWeighting(constant)
    w_const2 = NumpyFnConstWeighting(constant)
    w_other_const = NumpyFnConstWeighting(constant + 1)
    w_other_exp = NumpyFnConstWeighting(constant, exponent=1)

    const_sparse_mat = scipy.sparse.dia_matrix(([constant] * n, [0]),
                                               shape=(n, n))
    const_dense_mat = constant * np.eye(n)
    w_matrix_sp = NumpyFnMatrixWeighting(const_sparse_mat)
    w_matrix_de = NumpyFnMatrixWeighting(const_dense_mat)

    assert w_const == w_const
    assert w_const == w_const2
    assert w_const2 == w_const
    # Equivalent but not equal -> False
    assert w_const != w_matrix_sp
    assert w_const != w_matrix_de

    # Different
    assert w_const != w_other_const
    assert w_const != w_other_exp


def test_constant_equiv():
    n = 10
    constant = 1.5

    w_const = NumpyFnConstWeighting(constant)
    w_const2 = NumpyFnConstWeighting(constant)

    const_sparse_mat = scipy.sparse.dia_matrix(([constant] * n, [0]),
                                               shape=(n, n))
    const_dense_mat = constant * np.eye(n)
    w_matrix_sp = NumpyFnMatrixWeighting(const_sparse_mat)
    w_matrix_de = NumpyFnMatrixWeighting(const_dense_mat)

    # Equal -> True
    assert w_const.equiv(w_const)
    assert w_const.equiv(w_const2)
    # Equivalent matrix representation -> True
    assert w_const.equiv(w_matrix_sp)
    assert w_const.equiv(w_matrix_de)

    w_different_const = NumpyFnConstWeighting(2.5)
    assert not w_const.equiv(w_different_const)

    # Bogus input
    assert not w_const.equiv(True)
    assert not w_const.equiv(object)
    assert not w_const.equiv(None)


def test_constant_inner(fn):
    [xarr, yarr], [x, y] = noise_elements(fn, 2)

    constant = 1.5
    true_result_const = constant * np.vdot(yarr, xarr)

    w_const = NumpyFnConstWeighting(constant)
    assert almost_equal(w_const.inner(x, y), true_result_const)

    # Exponent != 2 -> no inner
    w_const = NumpyFnConstWeighting(constant, exponent=1)
    with pytest.raises(NotImplementedError):
        w_const.inner(x, y)


def test_constant_norm(fn, exponent):
    xarr, x = noise_elements(fn)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_norm = factor * np.linalg.norm(xarr, ord=exponent)

    w_const = NumpyFnConstWeighting(constant, exponent=exponent)
    assert almost_equal(w_const.norm(x), true_norm)

    # With free function
    w_const_norm = npy_weighted_norm(constant, exponent=exponent)
    assert almost_equal(w_const_norm(x), true_norm)


def test_constant_dist(fn, exponent):
    [xarr, yarr], [x, y] = noise_elements(fn, 2)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_dist = factor * np.linalg.norm(xarr - yarr, ord=exponent)

    w_const = NumpyFnConstWeighting(constant, exponent=exponent)
    assert almost_equal(w_const.dist(x, y), true_dist)

    # With free function
    w_const_dist = npy_weighted_dist(constant, exponent=exponent)
    assert almost_equal(w_const_dist(x, y), true_dist)


def test_noweight():
    w = NumpyFnNoWeighting()
    w_same1 = NumpyFnNoWeighting()
    w_same2 = NumpyFnNoWeighting(2)
    w_other_exp = NumpyFnNoWeighting(exponent=1)

    # Singleton pattern
    for same in (w_same1, w_same2):
        assert w is same

    # Proper creation
    assert w is not w_other_exp
    assert w != w_other_exp


def test_custom_inner(fn):
    [xarr, yarr], [x, y] = noise_elements(fn, 2)

    def inner(x, y):
        return np.vdot(y, x)

    w = NumpyFnCustomInner(inner)
    w_same = NumpyFnCustomInner(inner)
    w_other = NumpyFnCustomInner(np.dot)

    assert w == w
    assert w == w_same
    assert w != w_other

    true_inner = inner(xarr, yarr)
    assert almost_equal(w.inner(x, y), true_inner)

    true_norm = np.linalg.norm(xarr)
    assert almost_equal(w.norm(x), true_norm)

    true_dist = np.linalg.norm(xarr - yarr)
    # Using 3 places (single precision default) since the result is always
    # double even if the underlying computation was only single precision
    assert almost_equal(w.dist(x, y), true_dist, places=3)

    with pytest.raises(TypeError):
        NumpyFnCustomInner(1)  # wrong type


def test_custom_norm(fn):
    [xarr, yarr], [x, y] = noise_elements(fn, 2)

    norm = np.linalg.norm

    def other_norm(x):
        return np.linalg.norm(x, ord=1)

    w = NumpyFnCustomNorm(norm)
    w_same = NumpyFnCustomNorm(norm)
    w_other = NumpyFnCustomNorm(other_norm)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    true_norm = np.linalg.norm(xarr)
    assert almost_equal(w.norm(x), true_norm)

    true_dist = np.linalg.norm(xarr - yarr)
    assert almost_equal(w.dist(x, y), true_dist)

    with pytest.raises(TypeError):
        NumpyFnCustomNorm(1)


def test_custom_dist(fn):
    [xarr, yarr], [x, y] = noise_elements(fn, 2)

    def dist(x, y):
        return np.linalg.norm(x - y)

    def other_dist(x, y):
        return np.linalg.norm(x - y, ord=1)

    w = NumpyFnCustomDist(dist)
    w_same = NumpyFnCustomDist(dist)
    w_other = NumpyFnCustomDist(other_dist)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    with pytest.raises(NotImplementedError):
        w.norm(x)

    true_dist = np.linalg.norm(xarr - yarr)
    assert almost_equal(w.dist(x, y), true_dist)

    with pytest.raises(TypeError):
        NumpyFnCustomDist(1)


def test_ufuncs(fn, ufunc):
    name, n_args, n_out, _ = ufunc
    if (np.issubsctype(fn.dtype, np.floating) or
            np.issubsctype(fn.dtype, np.complexfloating)and
            name in ['bitwise_and',
                     'bitwise_or',
                     'bitwise_xor',
                     'invert',
                     'left_shift',
                     'right_shift']):
        # Skip integer only methods if floating point type
        return

    if (np.issubsctype(fn.dtype, np.complexfloating) and
            name in ['remainder',
                     'trunc',
                     'signbit',
                     'invert',
                     'left_shift',
                     'right_shift',
                     'rad2deg',
                     'deg2rad',
                     'copysign',
                     'mod',
                     'modf',
                     'fmod',
                     'logaddexp2',
                     'logaddexp',
                     'hypot',
                     'arctan2',
                     'floor',
                     'ceil']):
        # Skip real only methods for complex
        return

    # Get the ufunc from numpy as reference
    npufunc = getattr(np, name)

    # Create some data
    arrays, vectors = noise_elements(fn, n_args + n_out)
    in_arrays = arrays[:n_args]
    out_arrays = arrays[n_args:]
    data_vector = vectors[0]
    in_vectors = vectors[1:n_args]
    out_vectors = vectors[n_args:]

    # Out-of-place:
    np_result = npufunc(*in_arrays)
    vec_fun = getattr(data_vector.ufuncs, name)
    odl_result = vec_fun(*in_vectors)
    assert all_almost_equal(np_result, odl_result)

    # Test type of output
    if n_out == 1:
        assert isinstance(odl_result, fn.element_type)
    elif n_out > 1:
        for i in range(n_out):
            assert isinstance(odl_result[i], fn.element_type)

    # In-place:
    np_result = npufunc(*(in_arrays + out_arrays))
    vec_fun = getattr(data_vector.ufuncs, name)
    odl_result = vec_fun(*(in_vectors + out_vectors))
    assert all_almost_equal(np_result, odl_result)

    # Test in-place actually holds:
    if n_out == 1:
        assert odl_result is out_vectors[0]
    elif n_out > 1:
        for i in range(n_out):
            assert odl_result[i] is out_vectors[i]


def test_reduction(fn, reduction):
    name, _ = reduction

    ufunc = getattr(np, name)

    # Create some data
    x_arr, x = noise_elements(fn, 1)

    assert ufunc(x_arr) == getattr(x.ufuncs, name)()


def test_ufunc_reduction_docs_notempty():
    for _, __, ___, doc in UFUNCS:
        assert doc.splitlines()[0] != ''

    for _, doc in REDUCTIONS:
        assert doc.splitlines()[0] != ''


if __name__ == '__main__':
    odl.util.test_file(__file__)
