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
import scipy as sp
from math import ceil

# ODL imports
from odl import Ntuples, Fn, Rn, Cn
from odl.operator.operator import Operator
from odl.space.ntuples import FnConstWeighting, FnMatrixWeighting
from odl.util.testutils import almost_equal, all_almost_equal, all_equal

# TODO: add tests for:
# * Ntuples (different data types)
# * metric, normed, Hilbert space variants
# * MatVecOperator
# * inner, norm, dist as free functions
# * Vector weighting
# * Custom inner/norm/dist


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


def _sparse_matrix(fn):
    nnz = np.random.randint(0, int(ceil((fn.size ** 2) / 2)))
    coo_r = np.random.randint(0, fn.size, size=nnz)
    coo_c = np.random.randint(0, fn.size, size=nnz)
    values = np.random.rand(nnz)
    mat = sp.sparse.coo_matrix((values, (coo_r, coo_c)),
                               shape=(fn.size, fn.size))
    # Make symmetric and positive definite
    return mat + mat.T + fn.size * sp.sparse.eye(fn.size)


def _dense_matrix(fn):
    mat = np.asarray(np.random.rand(fn.size, fn.size), dtype=float)
    # Make symmetric and positive definite
    return mat + mat.T + fn.size * np.eye(fn.size)


@pytest.fixture(scope="module",
                ids=['R10 float64', 'R10 float32',
                     'C10 complex128', 'C10 complex64',
                     'R100'],
                params=[Rn(10, np.float64), Rn(10, np.float32),
                        Cn(10, np.complex128), Cn(10, np.complex64),
                        Rn(100)])
def fn(request):
    return request.param


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
    Rn(3, int)

    # Rn only works on reals
    with pytest.raises(TypeError):
        Rn(3, complex)
    with pytest.raises(TypeError):
        Rn(3, 'S1')


def test_vector_init(fn):
    # Test that code runs
    arr = _array(fn)

    fn.Vector(fn, arr)

    # Space has to be an actual space
    for non_space in [1, complex, np.array([1, 2])]:
        with pytest.raises(TypeError):
            fn.Vector(non_space, arr)

    # Data has to be a numpy array
    with pytest.raises(TypeError):
        fn.Vector(fn, list(arr))

    # Data has to be a numpy array or correct dtype
    with pytest.raises(TypeError):
        fn.Vector(fn, arr.astype(int))


def _test_lincomb(fn, a, b):
    # Validates lincomb against the result on host with randomized
    # data and given a,b

    # Unaliased arguments
    x, y, z, xVec, yVec, zVec = _vectors(fn, 3)

    z[:] = a * x + b * y
    fn.lincomb(a, xVec, b, yVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

    # First argument aliased with output
    x, y, z, xVec, yVec, zVec = _vectors(fn, 3)

    z[:] = a * z + b * y
    fn.lincomb(a, zVec, b, yVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

    # Second argument aliased with output
    x, y, z, xVec, yVec, zVec = _vectors(fn, 3)

    z[:] = a * x + b * z
    fn.lincomb(a, xVec, b, zVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

    # Both arguments aliased with each other
    x, y, z, xVec, yVec, zVec = _vectors(fn, 3)

    z[:] = a * x + b * x
    fn.lincomb(a, xVec, b, xVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

    # All aliased
    x, y, z, xVec, yVec, zVec = _vectors(fn, 3)
    z[:] = a * z + b * z
    fn.lincomb(a, zVec, b, zVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])


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
    """ Verifies that the statement y=function(x) gives equivalent
    results to Numpy.
    """

    x_arr, x = _vectors(fn)

    y_arr = function(x_arr)
    y = function(x)

    assert all_almost_equal([x, y], [x_arr, y_arr])


def _test_binary_operator(fn, function):
    """ Verifies that the statement z=function(x,y) gives equivalent
    results to Numpy.
    """

    x_arr, y_arr, x, y = _vectors(fn, 2)

    z_arr = function(x_arr, y_arr)
    z = function(x, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])


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


def test_norm(fn):
    xd = _element(fn)

    correct_norm = np.linalg.norm(xd.asarray())

    assert almost_equal(fn.norm(xd), correct_norm)
    assert almost_equal(xd.norm(), correct_norm)


def test_inner(fn):
    xd = _element(fn)
    yd = _element(fn)

    correct_inner = np.vdot(yd, xd)
    assert almost_equal(fn.inner(xd, yd), correct_inner)
    assert almost_equal(xd.inner(yd), correct_inner)


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
    """Verifies that multiplying with numpy scalars
    does not change the type of the array
    """

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
    """ Verifies that the __array__ method works
    """
    x = fn.zero()

    arr = x.__array__()

    assert isinstance(arr, np.ndarray)
    assert all_equal(arr, np.zeros(x.size))


def test_array_wrap_method(fn):
    """ Verifies that the __array_wrap__ method works
    This enables us to use numpy ufuncs on vectors
    """
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


def test_matrix_init(fn):
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)

    # Just test if the code runs
    FnMatrixWeighting(sparse_mat)
    FnMatrixWeighting(dense_mat)

    nonsquare_mat = np.eye(10, 5)
    with pytest.raises(ValueError):
        FnMatrixWeighting(nonsquare_mat)


def test_matrix_equals(fn):
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(fn)
    different_dense_mat = dense_mat.copy()
    different_dense_mat[0, 0] = -10

    w_sparse = FnMatrixWeighting(sparse_mat)
    w_sparse2 = FnMatrixWeighting(sparse_mat)
    w_sparse_as_dense = FnMatrixWeighting(sparse_mat_as_dense)
    w_dense = FnMatrixWeighting(dense_mat)
    w_dense2 = FnMatrixWeighting(dense_mat)
    w_different_dense = FnMatrixWeighting(different_dense_mat)

    # Identical objects -> True
    assert w_sparse == w_sparse
    # Identical matrices -> True
    assert w_sparse == w_sparse2
    assert w_dense == w_dense2
    # Equivalent but not identical matrices -> False
    assert w_sparse != w_sparse_as_dense
    assert w_sparse_as_dense != w_sparse
    # Not equivalent -> False
    assert w_dense != w_different_dense


def test_matrix_equiv(fn):
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


def test_matrix_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(fn)

    w_sparse = FnMatrixWeighting(sparse_mat)
    w_dense = FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.inner(x, y)
    result_dense = w_dense.inner(x, y)

    true_result_sparse = np.vdot(
        yarr, np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
    true_result_dense = np.vdot(
        yarr, np.asarray(np.dot(dense_mat, xarr)).squeeze())

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)


def test_matrix_norm(fn):
    xarr, yarr, x, y = _vectors(fn, 2)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(fn)

    w_sparse = FnMatrixWeighting(sparse_mat)
    w_dense = FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.norm(x)
    result_dense = w_dense.norm(x)

    true_result_sparse = np.sqrt(np.vdot(
        xarr, np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze()))
    true_result_dense = np.sqrt(np.vdot(
        xarr, np.asarray(np.dot(dense_mat, xarr)).squeeze()))

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)


def test_matrix_dist(fn):
    xarr, yarr, x, y = _vectors(fn, 2)
    sparse_mat = _sparse_matrix(fn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(fn)

    w_sparse = FnMatrixWeighting(sparse_mat)
    w_dense = FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.dist(x, y)
    result_dense = w_dense.dist(x, y)

    true_result_sparse = np.sqrt(np.vdot(
        xarr - yarr,
        np.asarray(np.dot(sparse_mat_as_dense, xarr - yarr)).squeeze()))
    true_result_dense = np.sqrt(np.vdot(
        xarr - yarr, np.asarray(np.dot(dense_mat, xarr - yarr)).squeeze()))

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)


def test_matrix_dist_squared(fn):
    xarr, yarr, x, y = _vectors(fn, 2)
    mat = _dense_matrix(fn)

    w = FnMatrixWeighting(mat, dist_using_inner=True)

    result = w.dist(x, y)

    true_result = np.sqrt(np.vdot(
        xarr - yarr, np.asarray(np.dot(mat, xarr - yarr)).squeeze()))

    assert almost_equal(result, true_result)


def test_constant_init():
    constant = 1.5

    # Just test if the code runs
    FnConstWeighting(constant)


def test_constant_equals():
    n = 10
    constant = 1.5

    w_const = FnConstWeighting(constant)
    w_const2 = FnConstWeighting(constant)

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

    w_different_const = FnConstWeighting(2.5)
    assert w_const != w_different_const


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


def test_constant_inner(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    constant = 1.5
    w_const = FnConstWeighting(constant)

    result_const = w_const.inner(x, y)
    true_result_const = constant * np.vdot(yarr, xarr)

    assert almost_equal(result_const, true_result_const)


def test_constant_norm(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    constant = 1.5
    w_const = FnConstWeighting(constant)

    result_const = w_const.norm(x)
    true_result_const = np.sqrt(constant * np.vdot(xarr, xarr))

    assert almost_equal(result_const, true_result_const)


def test_constant_dist(fn):
    xarr, yarr, x, y = _vectors(fn, 2)

    constant = 1.5
    w_const = FnConstWeighting(constant)

    result_const = w_const.dist(x, y)
    true_result_const = np.sqrt(constant * np.vdot(xarr - yarr, xarr - yarr))

    assert almost_equal(result_const, true_result_const)


def test_constant_repr():
    constant = 1.5
    w_const = FnConstWeighting(constant)

    repr_str = 'FnConstWeighting(1.5)'
    assert repr(w_const) == repr_str


def test_constant_str():
    constant = 1.5
    w_const = FnConstWeighting(constant)

    print_str = 'Weighting: const = 1.5'
    assert str(w_const) == print_str

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
