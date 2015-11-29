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
from math import ceil
import numpy as np
import pytest
import scipy as sp

# ODL imports
from odl import Ntuples, Fn, FnVector, Rn, Cn
from odl.operator.operator import Operator
from odl.space.ntuples import (
    FnConstWeighting, FnVectorWeighting, FnMatrixWeighting, FnNoWeighting,
    FnCustomInnerProduct, FnCustomNorm, FnCustomDist,
    weighted_inner, weighted_norm, weighted_dist,
    MatVecOperator)
from odl.util.testutils import almost_equal, all_almost_equal, all_equal

# TODO: add tests for:
# * inner, norm, dist as free functions
# * Vector weighting
# * Custom inner/norm/dist


# Helpers to generate data

def _array(fn):
    # Generate numpy vectors, real or complex or int
    if np.issubdtype(fn.dtype, np.floating):
        return np.random.rand(fn.size).astype(fn.dtype)
    elif np.issubdtype(fn.dtype, np.integer):
        return np.random.randint(0, 10, fn.size).astype(fn.dtype)
    elif np.issubdtype(fn.dtype, np.complexfloating):
        return (np.random.rand(fn.size) +
                1j * np.random.rand(fn.size)).astype(fn.dtype)
    else:
        raise TypeError('unable to handle data type {!r}'.format(fn.dtype))


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


def _dense_matrix(fn):
    """Create a dense positive definite Hermitian matrix for `fn`."""

    if np.issubdtype(fn.dtype, np.floating):
        mat = np.random.rand(fn.size, fn.size).astype(fn.dtype)
    elif np.issubdtype(fn.dtype, np.integer):
        mat = np.random.randint(0, 10, (fn.size, fn.size)).astype(fn.dtype)
    elif np.issubdtype(fn.dtype, np.complexfloating):
        mat = (np.random.rand(fn.size, fn.size) +
               1j * np.random.rand(fn.size, fn.size)).astype(fn.dtype)

    # Make symmetric and positive definite
    return mat + mat.conj().T + fn.size * np.eye(fn.size, dtype=fn.dtype)


def _sparse_matrix(fn):
    """Create a sparse positive definite Hermitian matrix for `fn`."""
    return sp.sparse.coo_matrix(_dense_matrix(fn))


# Pytest fixtures

# Simply modify spc_params to modify the fixture
spc_params = [Rn(10, np.float64), Rn(10, np.float32),
              Cn(10, np.complex128), Cn(10, np.complex64),
              Rn(100)]
spc_ids = [' {!r} '.format(spc) for spc in spc_params]
spc_fixture = pytest.fixture(scope="module", ids=spc_ids, params=spc_params)


@spc_fixture
def fn(request):
    return request.param


# Simply modify exp_params to modify the fixture
exp_params = [2.0, 1.0, float('inf'), 0.5, 1.5]
exp_ids = [' p = {} '.format(p) for p in exp_params]
exp_fixture = pytest.fixture(scope="module", ids=exp_ids, params=exp_params)


@exp_fixture
def exponent(request):
    return request.param


# ---- Ntuples, Rn and Cn ---- #


def test_init():
    # Test run
    Ntuples(3, int)
    Ntuples(3, float)
    Ntuples(3, complex)
    Ntuples(3, 'S1')

    # Fn
    Fn(3, int)
    Fn(3, float)
    Fn(3, complex)

    # Fn only works on scalars
    with pytest.raises(TypeError):
        Fn(3, 'S1')

    # Rn
    Rn(3, float)

    # Rn only works on reals
    with pytest.raises(TypeError):
        Rn(3, complex)
    with pytest.raises(TypeError):
        Rn(3, 'S1')
    with pytest.raises(TypeError):
        Rn(3, int)

    # Cn
    Cn(3, complex)

    # Cn only works on reals
    with pytest.raises(TypeError):
        Cn(3, float)
    with pytest.raises(TypeError):
        Cn(3, 'S1')

    # Backported int from future fails (not recognized by numpy.dtype())
    # (Python 2 only)
    from builtins import int as future_int
    import sys
    if sys.version_info.major != 3:
        with pytest.raises(TypeError):
            Fn(3, future_int)

    # Init with weights or custom space functions
    const = 1.5
    weight_vec = _pos_array(Rn(3, float))
    weight_mat = _dense_matrix(Rn(3, float))

    Rn(3, weight=const)
    Rn(3, weight=weight_vec)
    Rn(3, weight=weight_mat)

    # Different exponents
    exponents = [0.5, 1.0, 2.0, 5.0, float('inf')]
    for exponent in exponents:
        Cn(3, exponent=exponent)


def test_init_space_funcs(exponent):
    const = 1.5
    weight_vec = _pos_array(Rn(3, float))
    weight_mat = _dense_matrix(Rn(3, float))

    spaces = [Fn(3, complex, exponent=exponent, weight=const),
              Fn(3, complex, exponent=exponent, weight=weight_vec),
              Fn(3, complex, exponent=exponent, weight=weight_mat)]
    weightings = [FnConstWeighting(const, exponent=exponent),
                  FnVectorWeighting(weight_vec, exponent=exponent),
                  FnMatrixWeighting(weight_mat, exponent=exponent)]

    for spc, weight in zip(spaces, weightings):
        assert spc._space_funcs == weight


def test_vector_class_init(fn):
    # Test that code runs
    arr = _array(fn)

    FnVector(fn, arr)

    # Space has to be an actual space
    for non_space in [1, complex, np.array([1, 2])]:
        with pytest.raises(TypeError):
            FnVector(non_space, arr)

    # Data has to be a numpy array
    with pytest.raises(TypeError):
        FnVector(fn, list(arr))

    # Data has to be a numpy array or correct dtype
    with pytest.raises(TypeError):
        FnVector(fn, arr.astype(int))


def _test_lincomb(fn, a, b):
    # Validate lincomb against the result on host with randomized
    # data and given a,b, contiguous and non-contiguous

    # Unaliased arguments
    xarr, yarr, zarr, x, y, z = _vectors(fn, 3)
    zarr[:] = a * xarr + b * yarr
    fn.lincomb(a, x, b, y, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # First argument aliased with output
    xarr, yarr, zarr, x, y, z = _vectors(fn, 3)
    zarr[:] = a * zarr + b * yarr
    fn.lincomb(a, z, b, y, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # Second argument aliased with output
    xarr, yarr, zarr, x, y, z = _vectors(fn, 3)
    zarr[:] = a * xarr + b * zarr
    fn.lincomb(a, x, b, z, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # Both arguments aliased with each other
    xarr, yarr, zarr, x, y, z = _vectors(fn, 3)
    zarr[:] = a * xarr + b * xarr
    fn.lincomb(a, x, b, x, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # All aliased
    xarr, yarr, zarr, x, y, z = _vectors(fn, 3)
    zarr[:] = a * zarr + b * zarr
    fn.lincomb(a, z, b, z, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])


def test_lincomb(fn):
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(fn, a, b)


def test_multiply(fn):
    # space method
    x_arr, y_arr, out_arr, x, y, out = _vectors(fn, 3)
    out_arr = x_arr * y_arr

    fn.multiply(x, y, out)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])

    # member method
    x_arr, y_arr, out_arr, x, y, out = _vectors(fn, 3)
    out_arr = x_arr * y_arr

    out.multiply(x, y)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])


def _test_unary_operator(fn, function):
    # Verify that the statement y=function(x) gives equivalent results
    # to NumPy
    x_arr, x = _vectors(fn)

    y_arr = function(x_arr)
    y = function(x)

    assert all_almost_equal([x, y], [x_arr, y_arr])


def _test_binary_operator(fn, function):
    # Verify that the statement z=function(x,y) gives equivalent results
    # to NumPy
    x_arr, y_arr, x, y = _vectors(fn, 2)

    z_arr = function(x_arr, y_arr)
    z = function(x, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])


def test_operators(fn):
    # Test of all operator overloads against the corresponding NumPy
    # implementation

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


def test_inner(fn):
    xd = _element(fn)
    yd = _element(fn)

    correct_inner = np.vdot(yd, xd)
    assert almost_equal(fn.inner(xd, yd), correct_inner)
    assert almost_equal(xd.inner(yd), correct_inner)


def test_norm(fn):
    xarr, x = _vectors(fn)

    correct_norm = np.linalg.norm(xarr)

    assert almost_equal(fn.norm(x), correct_norm)
    assert almost_equal(x.norm(), correct_norm)


def test_pnorm(exponent):
    for fn in (Rn(3, exponent=exponent), Cn(3, exponent=exponent)):
        xarr, x = _vectors(fn)
        correct_norm = np.linalg.norm(xarr, ord=exponent)

        assert almost_equal(fn.norm(x), correct_norm)
        assert almost_equal(x.norm(), correct_norm)


def test_dist(fn):
    xarr, yarr, x, y = _vectors(fn, n=2)

    correct_dist = np.linalg.norm(xarr - yarr)

    assert almost_equal(fn.dist(x, y), correct_dist)
    assert almost_equal(x.dist(y), correct_dist)


def test_pdist(exponent):
    for fn in (Rn(3, exponent=exponent), Cn(3, exponent=exponent)):
        xarr, yarr, x, y = _vectors(fn, n=2)

        correct_dist = np.linalg.norm(xarr - yarr, ord=exponent)

        assert almost_equal(fn.dist(x, y), correct_dist)
        assert almost_equal(x.dist(y), correct_dist)


def test_setitem(fn):
    x = _element(fn)

    for index in [0, 1, 2, -1, -2, -3]:
        x[index] = index
        assert almost_equal(x[index], index)


def test_setitem_index_error(fn):
    x = _element(fn)

    with pytest.raises(IndexError):
        x[-fn.size - 1] = 0

    with pytest.raises(IndexError):
        x[fn.size] = 0


def _test_getslice(slice):
    # Validate get against python list behaviour
    r6 = Rn(6)
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
    r6 = Rn(6)
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
    x = _element(fn)
    y = _element(fn)

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


def test_copy(fn):
    import copy

    x = _element(fn)

    y = copy.copy(x)

    assert x == y
    assert y is not x

    z = copy.deepcopy(x)

    assert x == z
    assert z is not x


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


# Numpy Array tests

def test_array_method(fn):
    # Verify that the __array__ method works
    x = fn.zero()

    arr = x.__array__()

    assert isinstance(arr, np.ndarray)
    assert all_equal(arr, np.zeros(x.size))


def test_array_wrap_method(fn):
    # Verify that the __array_wrap__ method works. This enables numpy ufuncs
    # on vectors
    x_h, x = _vectors(fn)
    y_h = np.sin(x_h)
    y = np.sin(x)

    assert all_equal(y, y_h)
    assert y in fn


def test_conj(fn):
    xarr, x = _vectors(fn)
    xconj = x.conj()
    assert all_equal(xconj, xarr.conj())
    y = x.copy()
    xconj = x.conj(out=y)
    assert xconj is y
    assert all_equal(y, xarr.conj())


# ---- MatVecOperator ---- #


def test_matvec_init(fn):
    # Square matrices, sparse and dense
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)

    MatVecOperator(sparse_mat, fn, fn)
    MatVecOperator(dense_mat, fn, fn)

    # Test defaults
    op_float = MatVecOperator([[1.0, 2],
                               [-1, 0.5]])

    assert isinstance(op_float.domain, Rn)
    assert isinstance(op_float.range, Rn)

    op_complex = MatVecOperator([[1.0, 2 + 1j],
                                 [-1 - 1j, 0.5]])

    assert isinstance(op_complex.domain, Cn)
    assert isinstance(op_complex.range, Cn)

    op_int = MatVecOperator([[1, 2],
                             [-1, 0]])

    assert isinstance(op_int.domain, Fn)
    assert isinstance(op_int.range, Fn)

    # Rectangular
    rect_mat = 2 * np.eye(2, 3)
    r2 = Rn(2)
    r3 = Rn(3)

    MatVecOperator(rect_mat, r3, r2)

    with pytest.raises(ValueError):
        MatVecOperator(rect_mat, r2, r2)

    with pytest.raises(ValueError):
        MatVecOperator(rect_mat, r3, r3)

    with pytest.raises(ValueError):
        MatVecOperator(rect_mat, r2, r3)

    # Rn to Cn okay
    MatVecOperator(rect_mat, r3, Cn(2))

    # Cn to Rn not okay (no safe cast)
    with pytest.raises(TypeError):
        MatVecOperator(rect_mat, Cn(3), r2)

    # Complex matrix between real spaces not okay
    rect_complex_mat = rect_mat + 1j
    with pytest.raises(TypeError):
        MatVecOperator(rect_complex_mat, r3, r2)

    # Init with array-like structure (including numpy.matrix)
    MatVecOperator(rect_mat.tolist(), r3, r2)
    MatVecOperator(np.asmatrix(rect_mat), r3, r2)


def test_matvec_simple_properties():
    # Matrix - always ndarray in for dense input, scipy.sparse.spmatrix else
    rect_mat = 2 * np.eye(2, 3)
    r2 = Rn(2)
    r3 = Rn(3)

    op = MatVecOperator(rect_mat, r3, r2)
    assert isinstance(op.matrix, np.ndarray)

    op = MatVecOperator(np.asmatrix(rect_mat), r3, r2)
    assert isinstance(op.matrix, np.ndarray)

    op = MatVecOperator(rect_mat.tolist(), r3, r2)
    assert isinstance(op.matrix, np.ndarray)
    assert not op.matrix_issparse

    sparse_mat = _sparse_matrix(Rn(5))
    op = MatVecOperator(sparse_mat, Rn(5), Rn(5))
    assert isinstance(op.matrix, sp.sparse.spmatrix)
    assert op.matrix_issparse


def test_matvec_adjoint(fn):
    # Square cases
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)

    op_sparse = MatVecOperator(sparse_mat, fn, fn)
    op_dense = MatVecOperator(dense_mat, fn, fn)

    # Just test if it runs, nothing interesting to test here
    op_sparse.adjoint
    op_dense.adjoint

    # Rectangular case
    rect_mat = 2 * np.eye(2, 3)
    r2, r3 = Rn(2), Rn(3)
    c2 = Cn(2)

    op = MatVecOperator(rect_mat, r3, r2)
    op_adj = op.adjoint
    assert op_adj.domain == op.range
    assert op_adj.range == op.domain
    assert np.array_equal(op_adj.matrix, op.matrix.conj().T)
    assert np.array_equal(op_adj.adjoint.matrix, op.matrix)

    # The operator Rn -> Cn has no adjoint
    op_noadj = MatVecOperator(rect_mat, r3, c2)
    with pytest.raises(NotImplementedError):
        op_noadj.adjoint


def test_matvec_call(fn):
    # Square cases
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)
    xarr, x = _vectors(fn)

    op_sparse = MatVecOperator(sparse_mat, fn, fn)
    op_dense = MatVecOperator(dense_mat, fn, fn)

    yarr_sparse = sparse_mat.dot(xarr)
    yarr_dense = dense_mat.dot(xarr)

    # Out-of-place
    y = op_sparse(x)
    assert all_almost_equal(y, yarr_sparse)

    y = op_dense(x)
    assert all_almost_equal(y, yarr_dense)

    # In-place
    y = fn.element()
    op_sparse(x, out=y)
    assert all_almost_equal(y, yarr_sparse)

    y = fn.element()
    op_dense(x, out=y)
    assert all_almost_equal(y, yarr_dense)

    # Rectangular case
    rect_mat = 2 * np.eye(2, 3)
    r2, r3 = Rn(2), Rn(3)

    op = MatVecOperator(rect_mat, r3, r2)
    xarr = np.arange(3, dtype=float)
    x = r3.element(xarr)

    yarr = rect_mat.dot(xarr)

    # Out-of-place
    y = op(x)
    assert all_almost_equal(y, yarr)

    # In-place
    y = r2.element()
    op(x, out=y)
    assert all_almost_equal(y, yarr)


# --- Weighting tests --- #


def test_matrix_init(fn, exponent):
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)

    # Just test if the code runs
    FnMatrixWeighting(dense_mat, exponent=exponent)
    if exponent in (1.0, 2.0, float('inf')):
        FnMatrixWeighting(sparse_mat, exponent=exponent)
    else:
        with pytest.raises(NotImplementedError):
            FnMatrixWeighting(sparse_mat, exponent=exponent)

    nonsquare_mat = np.eye(10, 5)
    with pytest.raises(ValueError):
        FnMatrixWeighting(nonsquare_mat)


def test_matrix_matrix():
    fn = Rn(5)
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)

    w_sparse = FnMatrixWeighting(sparse_mat)
    w_dense = FnMatrixWeighting(dense_mat)

    assert isinstance(w_sparse.matrix, sp.sparse.spmatrix)
    assert isinstance(w_dense.matrix, np.ndarray)


def test_matrix_isvalid():
    fn = Rn(5)
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)
    bad_mat = np.eye(5)
    bad_mat[0, 0] = 0

    w_sparse = FnMatrixWeighting(sparse_mat)
    w_dense = FnMatrixWeighting(dense_mat)
    w_bad = FnMatrixWeighting(bad_mat)

    with pytest.raises(NotImplementedError):
        w_sparse.matrix_isvalid()

    assert w_dense.matrix_isvalid()
    assert not w_bad.matrix_isvalid()


def test_matrix_equals(fn, exponent):
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(fn)
    different_dense_mat = dense_mat.copy()
    different_dense_mat[0, 0] -= 1

    if exponent in (1.0, 2.0, float('inf')):
        w_sparse = FnMatrixWeighting(sparse_mat, exponent=exponent)
        w_sparse2 = FnMatrixWeighting(sparse_mat, exponent=exponent)
    w_sparse_as_dense = FnMatrixWeighting(sparse_mat_as_dense,
                                          exponent=exponent)
    w_dense = FnMatrixWeighting(dense_mat, exponent=exponent)
    w_dense2 = FnMatrixWeighting(dense_mat, exponent=exponent)
    w_different_mat = FnMatrixWeighting(different_dense_mat,
                                        exponent=exponent)
    diff_exp = exponent + 1 if np.isfinite(exponent) else 1
    w_different_exp = FnMatrixWeighting(dense_mat, exponent=diff_exp)

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
    fn = Rn(5)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(fn)
    different_dense_mat = dense_mat.copy()
    different_dense_mat[0, 0] = -10

    w_sparse = FnMatrixWeighting(sparse_mat)
    w_sparse2 = FnMatrixWeighting(sparse_mat)
    w_sparse_as_dense = FnMatrixWeighting(sparse_mat_as_dense)
    w_dense = FnMatrixWeighting(dense_mat)
    w_dense_copy = FnMatrixWeighting(dense_mat.copy())
    w_different_dense = FnMatrixWeighting(different_dense_mat)

    # Equal -> True
    assert w_sparse.equiv(w_sparse)
    assert w_sparse.equiv(w_sparse2)
    # Equivalent matrices -> True
    assert w_sparse.equiv(w_sparse_as_dense)
    assert w_dense.equiv(w_dense_copy)
    # Different matrices -> False
    assert not w_dense.equiv(w_different_dense)

    # Test shortcuts
    sparse_eye = sp.sparse.eye(5)
    w_eye = FnMatrixWeighting(sparse_eye)
    w_dense_eye = FnMatrixWeighting(sparse_eye.todense())
    w_eye_vec = FnVectorWeighting(np.ones(5))

    w_eye_wrong_exp = FnMatrixWeighting(sparse_eye, exponent=1)

    sparse_smaller_eye = sp.sparse.eye(4)
    w_smaller_eye = FnMatrixWeighting(sparse_smaller_eye)

    sparse_shifted_eye = sp.sparse.eye(5, k=1)
    w_shifted_eye = FnMatrixWeighting(sparse_shifted_eye)

    sparse_almost_eye = sp.sparse.dia_matrix((np.ones(4), [0]), (5, 5))
    w_almost_eye = FnMatrixWeighting(sparse_almost_eye)

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
    xarr, yarr, x, y = _vectors(fn, 2)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = np.asarray(sparse_mat.todense())
    dense_mat = _dense_matrix(fn)

    true_inner_sparse = np.vdot(yarr, np.dot(sparse_mat_as_dense, xarr))
    true_inner_dense = np.vdot(yarr, np.dot(dense_mat, xarr))

    w_sparse = FnMatrixWeighting(sparse_mat)
    w_dense = FnMatrixWeighting(dense_mat)
    assert almost_equal(w_sparse.inner(x, y), true_inner_sparse)
    assert almost_equal(w_dense.inner(x, y), true_inner_dense)

    # With free functions
    w_sparse_inner = weighted_inner(sparse_mat)
    w_dense_inner = weighted_inner(dense_mat)
    assert almost_equal(w_sparse_inner(x, y), true_inner_sparse)
    assert almost_equal(w_dense_inner(x, y), true_inner_dense)

    # Exponent != 2 -> no inner
    w_dense = FnMatrixWeighting(dense_mat, exponent=1)
    with pytest.raises(NotImplementedError):
        w_dense.inner(x, y)


def test_matrix_norm(fn, exponent):
    xarr, x = _vectors(fn)
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
        eigval, eigvec = sp.linalg.eigh(dense_mat)
        eigval **= 1.0 / exponent
        mat_pow = (eigval * eigvec).dot(eigvec.conj().T)
        true_norm_dense = np.linalg.norm(np.dot(mat_pow, xarr), ord=exponent)

    # Test weighting
    if exponent in (1.0, 2.0, float('inf')):
        w_sparse = FnMatrixWeighting(sparse_mat, exponent=exponent)
        assert almost_equal(w_sparse.norm(x), true_norm_sparse)

    w_dense = FnMatrixWeighting(dense_mat, exponent=exponent)
    assert almost_equal(w_dense.norm(x), true_norm_dense)

    # With free functions
    if exponent not in (1.0, 2.0, float('inf')):
        with pytest.raises(NotImplementedError):
            weighted_norm(sparse_mat, exponent=exponent)
    else:
        w_sparse_norm = weighted_norm(sparse_mat, exponent=exponent)
        assert almost_equal(w_sparse_norm(x), true_norm_sparse)

    w_dense_norm = weighted_norm(dense_mat, exponent=exponent)
    assert almost_equal(w_dense_norm(x), true_norm_dense)


def test_matrix_dist(fn, exponent):
    xarr, yarr, x, y = _vectors(fn, n=2)
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
        eigval, eigvec = sp.linalg.eigh(dense_mat)
        eigval **= 1.0 / exponent
        mat_pow = (eigval * eigvec).dot(eigvec.conj().T)
        true_dist_dense = np.linalg.norm(np.dot(mat_pow, xarr - yarr),
                                         ord=exponent)

    if exponent in (1.0, 2.0, float('inf')):
        w_sparse = FnMatrixWeighting(sparse_mat, exponent=exponent)
        assert almost_equal(w_sparse.dist(x, y), true_dist_sparse)

    w_dense = FnMatrixWeighting(dense_mat, exponent=exponent)
    assert almost_equal(w_dense.dist(x, y), true_dist_dense)

    # With free functions
    if exponent in (1.0, 2.0, float('inf')):
        w_sparse_dist = weighted_dist(sparse_mat, exponent=exponent)
        assert almost_equal(w_sparse_dist(x, y), true_dist_sparse)

    w_dense_dist = weighted_dist(dense_mat, exponent=exponent)
    assert almost_equal(w_dense_dist(x, y), true_dist_dense)


def test_matrix_dist_using_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)
    mat = _dense_matrix(fn)

    w = FnMatrixWeighting(mat, dist_using_inner=True)

    true_dist = np.sqrt(np.vdot(xarr - yarr, np.dot(mat, xarr - yarr)))
    assert almost_equal(w.dist(x, y), true_dist)

    # Only possible for exponent=2
    with pytest.raises(ValueError):
        FnMatrixWeighting(mat, exponent=1, dist_using_inner=True)

    # With free function
    w_dist = weighted_dist(mat, use_inner=True)
    assert almost_equal(w_dist(x, y), true_dist)
    assert almost_equal(w.dist(x, x), 0)


def test_vector_init(exponent):
    rn = Rn(5)
    weight_vec = _pos_array(rn)

    FnVectorWeighting(weight_vec, exponent=exponent)
    FnVectorWeighting(rn.element(weight_vec), exponent=exponent)


def test_vector_vector():
    rn = Rn(5)
    weight_vec = _pos_array(rn)
    weight_elem = rn.element(weight_vec)

    weighting_vec = FnVectorWeighting(weight_vec)
    weighting_elem = FnVectorWeighting(weight_elem)

    assert isinstance(weighting_vec.vector, np.ndarray)
    assert isinstance(weighting_elem.vector, FnVector)


def test_vector_isvalid():
    rn = Rn(5)
    weight_vec = _pos_array(rn)
    weighting_vec = FnVectorWeighting(weight_vec)

    assert weighting_vec.vector_is_valid()

    # Invalid
    weight_vec[0] = 0
    weighting_vec = FnVectorWeighting(weight_vec)
    assert not weighting_vec.vector_is_valid()


def test_vector_equals():
    rn = Rn(5)
    weight_vec = _pos_array(rn)
    weight_elem = rn.element(weight_vec)

    weighting_vec = FnVectorWeighting(weight_vec)
    weighting_vec2 = FnVectorWeighting(weight_vec)
    weighting_elem = FnVectorWeighting(weight_elem)
    weighting_elem2 = FnVectorWeighting(weight_elem)
    weighting_other_vec = FnVectorWeighting(weight_vec - 1)
    weighting_other_exp = FnVectorWeighting(weight_vec - 1, exponent=1)

    assert weighting_vec == weighting_vec2
    assert weighting_vec != weighting_elem
    assert weighting_elem == weighting_elem2
    assert weighting_vec != weighting_other_vec
    assert weighting_vec != weighting_other_exp


def test_vector_equiv():
    rn = Rn(5)
    weight_vec = _pos_array(rn)
    weight_elem = rn.element(weight_vec)
    diag_mat = weight_vec * np.eye(5)
    different_vec = weight_vec - 1

    w_vec = FnVectorWeighting(weight_vec)
    w_elem = FnVectorWeighting(weight_elem)
    w_diag_mat = FnMatrixWeighting(diag_mat)
    w_different_vec = FnVectorWeighting(different_vec)

    # Equal -> True
    assert w_vec.equiv(w_vec)
    assert w_vec.equiv(w_elem)
    # Equivalent matrix -> True
    assert w_vec.equiv(w_diag_mat)
    # Different vector -> False
    assert not w_vec.equiv(w_different_vec)

    # Test shortcuts
    const_vec = np.ones(5) * 1.5

    w_vec = FnVectorWeighting(const_vec)
    w_const = FnConstWeighting(1.5)
    w_wrong_const = FnConstWeighting(1)
    w_wrong_exp = FnConstWeighting(1.5, exponent=1)

    assert w_vec.equiv(w_const)
    assert not w_vec.equiv(w_wrong_const)
    assert not w_vec.equiv(w_wrong_exp)

    # Bogus input
    assert not w_vec.equiv(True)
    assert not w_vec.equiv(object)
    assert not w_vec.equiv(None)


def test_vector_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    weight_vec = _pos_array(fn)
    weighting_vec = FnVectorWeighting(weight_vec)

    true_inner = np.vdot(yarr, xarr * weight_vec)

    assert almost_equal(weighting_vec.inner(x, y), true_inner)

    # With free function
    inner_vec = weighted_inner(weight_vec)

    assert almost_equal(inner_vec(x, y), true_inner)

    # Exponent != 2 -> no inner product, should raise
    with pytest.raises(NotImplementedError):
        FnVectorWeighting(weight_vec, exponent=1.0).inner(x, y)


def test_vector_norm(fn, exponent):
    xarr, x = _vectors(fn)

    weight_vec = _pos_array(fn)
    weighting_vec = FnVectorWeighting(weight_vec, exponent=exponent)

    if exponent == float('inf'):
        true_norm = np.linalg.norm(weight_vec * xarr, ord=float('inf'))
    else:
        true_norm = np.linalg.norm(weight_vec ** (1 / exponent) * xarr,
                                   ord=exponent)

    assert almost_equal(weighting_vec.norm(x), true_norm)

    # With free function
    pnorm_vec = weighted_norm(weight_vec, exponent=exponent)
    assert almost_equal(pnorm_vec(x), true_norm)


def test_vector_dist(fn, exponent):
    xarr, yarr, x, y = _vectors(fn, n=2)

    weight_vec = _pos_array(fn)
    weighting_vec = FnVectorWeighting(weight_vec, exponent=exponent)

    if exponent == float('inf'):
        true_dist = np.linalg.norm(
            weight_vec * (xarr - yarr), ord=float('inf'))
    else:
        true_dist = np.linalg.norm(
            weight_vec ** (1 / exponent) * (xarr - yarr), ord=exponent)

    assert almost_equal(weighting_vec.dist(x, y), true_dist)

    # With free function
    pdist_vec = weighted_dist(weight_vec, exponent=exponent)
    assert almost_equal(pdist_vec(x, y), true_dist)


def test_vector_dist_using_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    weight_vec = _pos_array(fn)
    w = FnVectorWeighting(weight_vec)

    true_dist = np.linalg.norm(np.sqrt(weight_vec) * (xarr - yarr))
    assert almost_equal(w.dist(x, y), true_dist)
    assert almost_equal(w.dist(x, x), 0)

    # Only possible for exponent=2
    with pytest.raises(ValueError):
        FnVectorWeighting(weight_vec, exponent=1, dist_using_inner=True)

    # With free function
    w_dist = weighted_dist(weight_vec, use_inner=True)
    assert almost_equal(w_dist(x, y), true_dist)


def test_constant_init(exponent):
    constant = 1.5

    # Just test if the code runs
    FnConstWeighting(constant, exponent=exponent)

    with pytest.raises(ValueError):
        FnConstWeighting(0)
    with pytest.raises(ValueError):
        FnConstWeighting(-1)
    with pytest.raises(ValueError):
        FnConstWeighting(float('inf'))


def test_constant_equals():
    n = 10
    constant = 1.5

    w_const = FnConstWeighting(constant)
    w_const2 = FnConstWeighting(constant)
    w_other_const = FnConstWeighting(constant + 1)
    w_other_exp = FnConstWeighting(constant, exponent=1)

    const_sparse_mat = sp.sparse.dia_matrix(([constant] * n, [0]),
                                            shape=(n, n))
    const_dense_mat = constant * np.eye(n)
    w_matrix_sp = FnMatrixWeighting(const_sparse_mat)
    w_matrix_de = FnMatrixWeighting(const_dense_mat)

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

    w_const = FnConstWeighting(constant)
    w_const2 = FnConstWeighting(constant)

    const_sparse_mat = sp.sparse.dia_matrix(([constant] * n, [0]),
                                            shape=(n, n))
    const_dense_mat = constant * np.eye(n)
    w_matrix_sp = FnMatrixWeighting(const_sparse_mat)
    w_matrix_de = FnMatrixWeighting(const_dense_mat)

    # Equal -> True
    assert w_const.equiv(w_const)
    assert w_const.equiv(w_const2)
    # Equivalent matrix representation -> True
    assert w_const.equiv(w_matrix_sp)
    assert w_const.equiv(w_matrix_de)

    w_different_const = FnConstWeighting(2.5)
    assert not w_const.equiv(w_different_const)

    # Bogus input
    assert not w_const.equiv(True)
    assert not w_const.equiv(object)
    assert not w_const.equiv(None)


def test_constant_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    constant = 1.5
    true_result_const = constant * np.vdot(yarr, xarr)

    w_const = FnConstWeighting(constant)
    assert almost_equal(w_const.inner(x, y), true_result_const)

    # Exponent != 2 -> no inner
    w_const = FnConstWeighting(constant, exponent=1)
    with pytest.raises(NotImplementedError):
        w_const.inner(x, y)


def test_constant_norm(fn, exponent):
    xarr, x = _vectors(fn)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_norm = factor * np.linalg.norm(xarr, ord=exponent)

    w_const = FnConstWeighting(constant, exponent=exponent)
    assert almost_equal(w_const.norm(x), true_norm)

    # With free function
    w_const_norm = weighted_norm(constant, exponent=exponent)
    assert almost_equal(w_const_norm(x), true_norm)


def test_constant_dist(fn, exponent):
    xarr, yarr, x, y = _vectors(fn, 2)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_dist = factor * np.linalg.norm(xarr - yarr, ord=exponent)

    w_const = FnConstWeighting(constant, exponent=exponent)
    assert almost_equal(w_const.dist(x, y), true_dist)

    # With free function
    w_const_dist = weighted_dist(constant, exponent=exponent)
    assert almost_equal(w_const_dist(x, y), true_dist)


def test_const_dist_using_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    constant = 1.5
    w = FnConstWeighting(constant)

    true_dist = np.sqrt(constant) * np.linalg.norm(xarr - yarr)
    assert almost_equal(w.dist(x, y), true_dist)
    assert almost_equal(w.dist(x, x), 0)

    # Only possible for exponent=2
    with pytest.raises(ValueError):
        FnConstWeighting(constant, exponent=1, dist_using_inner=True)

    # With free function
    w_dist = weighted_dist(constant, use_inner=True)
    assert almost_equal(w_dist(x, y), true_dist)


def test_noweight():
    w = FnNoWeighting()
    w_same1 = FnNoWeighting()
    w_same2 = FnNoWeighting(2)
    w_same3 = FnNoWeighting(2, False)
    w_same4 = FnNoWeighting(2, dist_using_inner=False)
    w_same5 = FnNoWeighting(exponent=2, dist_using_inner=False)
    w_other_exp = FnNoWeighting(exponent=1)
    w_dist_inner = FnNoWeighting(dist_using_inner=True)

    # Singleton pattern
    for same in (w_same1, w_same2, w_same3, w_same4, w_same5):
        assert w is same

    # Proper creation
    assert w is not w_other_exp
    assert w is not w_dist_inner
    assert w != w_other_exp
    assert w != w_dist_inner


def test_custom_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    def inner(x, y):
        return np.vdot(y, x)

    w = FnCustomInnerProduct(inner)
    w_same = FnCustomInnerProduct(inner)
    w_other = FnCustomInnerProduct(np.dot)
    w_d = FnCustomInnerProduct(inner, dist_using_inner=True)

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
        FnCustomInnerProduct(1)


def test_custom_norm(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    norm = np.linalg.norm

    def other_norm(x):
        return np.linalg.norm(x, ord=1)

    w = FnCustomNorm(norm)
    w_same = FnCustomNorm(norm)
    w_other = FnCustomNorm(other_norm)

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
        FnCustomNorm(1)


def test_custom_dist(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    def dist(x, y):
        return np.linalg.norm(x - y)

    def other_dist(x, y):
        return np.linalg.norm(x - y, ord=1)

    w = FnCustomDist(dist)
    w_same = FnCustomDist(dist)
    w_other = FnCustomDist(other_dist)

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
        FnCustomDist(1)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
