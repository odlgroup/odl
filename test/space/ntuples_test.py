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
from math import sqrt, ceil
from textwrap import dedent

# ODL imports
from odl import Rn, Cn
from odl.space.ntuples import _FnConstWeighting, _FnMatrixWeighting
from odl.util.testutils import almost_equal, all_almost_equal

# TODO: add tests for:
# * Ntuples (different data types)
# * metric, normed, Hilbert space variants
# * Cn
# * Rn, Cn with non-standard data types
# * vector multiplication
# * MatVecOperator
# * Custom inner/norm/dist

def _vectors(fn, n=1):
    # Generate numpy vectors, real or complex
    if isinstance(fn, Rn):
        arrs = [np.random.rand(fn.size) for _ in range(n)]
    else:
        arrs = [np.random.rand(fn.size) + 1j * np.random.rand(fn.size) for _ in range(n)]

    # Make Fn vectors
    vecs = [fn.element(arr) for arr in arrs]
    return arrs + vecs

def _sparse_matrix(fn):
    nnz = np.random.randint(0, int(ceil(fn.size**2/2)))
    coo_r = np.random.randint(0, fn.size, size=nnz)
    coo_c = np.random.randint(0, fn.size, size=nnz)
    values = np.random.rand(nnz)
    mat = sp.sparse.coo_matrix((values, (coo_r, coo_c)),
                               shape=(fn.size, fn.size))
    # Make symmetric and positive definite
    return mat + mat.T + fn.size * sp.sparse.eye(fn.size)

def _dense_matrix(fn):
    mat = np.asmatrix(np.random.rand(fn.size, fn.size), dtype=float)
    # Make symmetric and positive definite
    return mat + mat.T + fn.size * np.eye(fn.size)

def _test_lincomb(a, b, n=10):
    # Validates lincomb against the result on host with randomized
    # data and given a,b
    rn = Rn(n)

    # Unaliased arguments
    x, y, z, xVec, yVec, zVec = _vectors(rn, 3)

    z[:] = a*x + b*y
    rn.lincomb(a, xVec, b, yVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

    # First argument aliased with output
    x, y, z, xVec, yVec, zVec = _vectors(rn, 3)

    z[:] = a*z + b*y
    rn.lincomb(a, zVec, b, yVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

    # Second argument aliased with output
    x, y, z, xVec, yVec, zVec = _vectors(rn, 3)

    z[:] = a*x + b*z
    rn.lincomb(a, xVec, b, zVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

    # Both arguments aliased with each other
    x, y, z, xVec, yVec, zVec = _vectors(rn, 3)

    z[:] = a*x + b*x
    rn.lincomb(a, xVec, b, xVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

    # All aliased
    x, y, z, xVec, yVec, zVec = _vectors(rn, 3)
    z[:] = a*z + b*z
    rn.lincomb(a, zVec, b, zVec, out=zVec)
    assert all_almost_equal([xVec, yVec, zVec], [x, y, z])

def test_lincomb():
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(a, b)


def _test_unary_operator(function, n=10):
    """ Verifies that the statement y=function(x) gives equivalent
    results to Numpy.
    """
    rn = Rn(n)

    x_arr = np.random.rand(n)
    y_arr = function(x_arr)

    x = rn.element(x_arr)
    y = function(x)

    assert all_almost_equal([x, y], [x_arr, y_arr])

def _test_binary_operator(function, n=10):
    """ Verifies that the statement z=function(x,y) gives equivalent
    results to Numpy.
    """
    rn = Rn(n)

    x_arr = np.random.rand(n)
    y_arr = np.random.rand(n)
    z_arr = function(x_arr, y_arr)

    x = rn.element(x_arr)
    y = rn.element(y_arr)
    z = function(x, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

def test_operators():
    """ Test of all operator overloads against the corresponding
    Numpy implementation
    """
    # Unary operators
    _test_unary_operator(lambda x: +x)
    _test_unary_operator(lambda x: -x)

    # Scalar multiplication
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def imul(x):
            x *= scalar
        _test_unary_operator(imul)
        _test_unary_operator(lambda x: x*scalar)

    # Scalar division
    for scalar in [-31.2, -1, 1, 2.13]:
        def idiv(x):
            x /= scalar
        _test_unary_operator(idiv)
        _test_unary_operator(lambda x: x/scalar)

    # Incremental operations
    def iadd(x, y):
        x += y

    def isub(x, y):
        x -= y

    _test_binary_operator(iadd)
    _test_binary_operator(isub)

    # Incremental operators with aliased inputs
    def iadd_aliased(x):
        x += x

    def isub_aliased(x):
        x -= x
    _test_unary_operator(iadd_aliased)
    _test_unary_operator(isub_aliased)

    # Binary operators
    _test_binary_operator(lambda x, y: x + y)
    _test_binary_operator(lambda x, y: x - y)

    # Binary with aliased inputs
    _test_unary_operator(lambda x: x + x)
    _test_unary_operator(lambda x: x - x)


def test_norm():
    r3 = Rn(3)
    xd = r3.element([1, 2, 3])

    correct_norm = sqrt(1**2 + 2**2 + 3**2)
    assert almost_equal(r3.norm(xd), correct_norm)


def test_inner():
    r3 = Rn(3)
    xd = r3.element([1, 2, 3])
    yd = r3.element([5, -3, 9])

    correct_inner = 1*5 + 2*(-3) + 3*9
    assert almost_equal(r3.inner(xd, yd), correct_inner)


def test_setitem():
    r3 = Rn(3)
    x = r3.element([42, 42, 42])

    for index in [0, 1, 2, -1, -2, -3]:
        x[index] = index
        assert almost_equal(x[index], index)

def test_setitem_index_error():
    r3 = Rn(3)
    x = r3.element([1, 2, 3])

    with pytest.raises(IndexError):
        x[-4] = 0

    with pytest.raises(IndexError):
        x[3] = 0

def _test_getslice(slice):
    # Validate get against python list behaviour
    r6 = Rn(6)
    y = [0, 1, 2, 3, 4, 5]
    x = r6.element(y)

    assert all_almost_equal(x[slice].data, y[slice])

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
    assert all_almost_equal(x, y)

def test_setslice():
    # Tests a range of combination of slices
    steps = [None, -2, -1, 1, 2]
    starts = [None, -1, -3, 0, 2, 5, 10]
    ends = [None, -1, -3, 0, 2, 5, 10]

    for start in starts:
        for end in ends:
            for step in steps:
                _test_setslice(slice(start, end, step))

def test_setslice_index_error():
    r3 = Rn(3)
    xd = r3.element([1, 2, 3])

    # Bad slice
    with pytest.raises(ValueError):
        xd[10:13] = [1, 2, 3]

    # Bad size of rhs
    with pytest.raises(ValueError):
        xd[:] = []

    with pytest.raises(ValueError):
        xd[:] = [1, 2]

    with pytest.raises(ValueError):
        xd[:] = [1, 2, 3, 4]

def test_multiply_by_scalar():
    """Verifies that multiplying with numpy scalars
    does not change the type of the array
    """

    r3 = Rn(3)
    x = r3.zero()
    assert x * 1.0 in r3
    assert x * np.float32(1.0) in r3
    assert 1.0 * x in r3
    assert np.float32(1.0) * x in r3

def test_array_method():
    """ Verifies that the __array__ method works
    """
    r3 = Rn(3)
    x = r3.zero()

    arr = x.__array__()

    assert isinstance(arr, np.ndarray)
    assert all_almost_equal(arr, [0.0, 0.0, 0.0])

def test_array_wrap_method():
    """ Verifies that the __array_wrap__ method works
    This enables us to use numpy ufuncs on vectors
    """
    r3 = Rn(3)
    x_h = [0.0, 1.0, 2.0]
    x = r3.element([0.0, 1.0, 2.0])
    y_h = np.sin(x_h)
    y = np.sin(x)

    assert all_almost_equal(y, y_h)
    assert y in r3

    #Test with a non-standard dtype
    r3_f32 = Rn(3, dtype=np.float32)
    x_h = [0.0, 1.0, 2.0]
    x = r3_f32.element(x_h)
    y_h = np.sin(x_h)
    y = np.sin(x)

    assert all_almost_equal(y, y_h)
    assert y in r3_f32

def test_matrix_init():
    rn = Rn(10)
    sparse_mat = _sparse_matrix(rn)
    dense_mat = _dense_matrix(rn)

    # Just test if the code runs
    _FnMatrixWeighting(sparse_mat)
    _FnMatrixWeighting(dense_mat)

    nonsquare_mat = np.eye(10, 5)
    with pytest.raises(ValueError):
        _FnMatrixWeighting(nonsquare_mat)

def _test_matrix_equals(n):
    rn = Rn(n)
    sparse_mat = _sparse_matrix(rn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(rn)
    different_dense_mat = dense_mat.copy()
    different_dense_mat[0, 0] = -10

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_sparse2 = _FnMatrixWeighting(sparse_mat)
    w_sparse_as_dense = _FnMatrixWeighting(sparse_mat_as_dense)
    w_dense = _FnMatrixWeighting(dense_mat)
    w_dense2 = _FnMatrixWeighting(dense_mat)
    w_different_dense = _FnMatrixWeighting(different_dense_mat)

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

def test_matrix_equals():
    for n in range(1, 20):
        _test_matrix_equals(n)

def _test_matrix_equiv(n):
    rn = Rn(n)
    sparse_mat = _sparse_matrix(rn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(rn)
    different_dense_mat = dense_mat.copy()
    different_dense_mat[0, 0] = -10

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_sparse2 = _FnMatrixWeighting(sparse_mat)
    w_sparse_as_dense = _FnMatrixWeighting(
        rn, sparse_mat_as_dense)
    w_dense = _FnMatrixWeighting(dense_mat)
    w_dense_copy = _FnMatrixWeighting(dense_mat.copy())
    w_different_dense = _FnMatrixWeighting(
        rn, different_dense_mat)

    # Equal -> True
    assert w_sparse.equiv(w_sparse)
    assert w_sparse.equiv(w_sparse2)
    # Equivalent matrices -> True
    assert w_sparse.equiv(w_sparse_as_dense)
    assert w_dense.equiv(w_dense_copy)
    # Different matrices -> False
    assert not w_dense.equiv(w_different_dense)

def _test_matrix_inner_real(n):
    rn = Rn(n)
    xarr, yarr, x, y = _vectors(rn, 2)
    sparse_mat = _sparse_matrix(rn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(rn)

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_dense = _FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.inner(x, y)
    result_dense = w_dense.inner(x, y)

    true_result_sparse = np.dot(
        yarr, np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
    true_result_dense = np.dot(
        yarr, np.asarray(np.dot(dense_mat, xarr)).squeeze())

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)

def _test_matrix_norm_real(n):
    rn = Rn(n)
    xarr, yarr, x, y = _vectors(rn, 2)
    sparse_mat = _sparse_matrix(rn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(rn)

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_dense = _FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.norm(x)
    result_dense = w_dense.norm(x)

    true_result_sparse = np.sqrt(np.dot(
        xarr, np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze()))
    true_result_dense = np.sqrt(np.dot(
        xarr, np.asarray(np.dot(dense_mat, xarr)).squeeze()))

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)

def _test_matrix_dist_real(n):
    rn = Rn(n)
    xarr, yarr, x, y = _vectors(rn, 2)
    sparse_mat = _sparse_matrix(rn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(rn)

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_dense = _FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.dist(x, y)
    result_dense = w_dense.dist(x, y)

    true_result_sparse = np.sqrt(np.dot(
        xarr-yarr,
        np.asarray(np.dot(sparse_mat_as_dense, xarr-yarr)).squeeze()))
    true_result_dense = np.sqrt(np.dot(
        xarr-yarr, np.asarray(np.dot(dense_mat, xarr-yarr)).squeeze()))

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)

def _test_matrix_inner_complex(n):
    cn = Cn(n)
    xarr, yarr, x, y = _vectors(cn, 2)
    sparse_mat = _sparse_matrix(cn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(cn)

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_dense = _FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.inner(x, y)
    result_dense = w_dense.inner(x, y)

    true_result_sparse = np.vdot(
        yarr,
        np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
    true_result_dense = np.vdot(
        yarr, np.asarray(np.dot(dense_mat, xarr)).squeeze())

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)

def _test_matrix_norm_complex(n):
    cn = Cn(n)
    xarr, yarr, x, y = _vectors(cn, 2)
    sparse_mat = _sparse_matrix(cn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(cn)

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_dense = _FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.norm(x)
    result_dense = w_dense.norm(x)

    true_result_sparse = np.sqrt(np.vdot(
        xarr,
        np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze()))
    true_result_dense = np.sqrt(np.vdot(
        xarr, np.asarray(np.dot(dense_mat, xarr)).squeeze()))

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)

def _test_matrix_dist_complex(n):
    cn = Cn(n)
    xarr, yarr, x, y = _vectors(cn, 2)
    sparse_mat = _sparse_matrix(cn)
    sparse_mat_as_dense = sparse_mat.todense()
    dense_mat = _dense_matrix(cn)

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_dense = _FnMatrixWeighting(dense_mat)

    result_sparse = w_sparse.dist(x, y)
    result_dense = w_dense.dist(x, y)

    true_result_sparse = np.sqrt(np.vdot(
        xarr-yarr,
        np.asarray(np.dot(sparse_mat_as_dense, xarr-yarr)).squeeze()))
    true_result_dense = np.sqrt(np.vdot(
        xarr-yarr,
        np.asarray(np.dot(dense_mat, xarr-yarr)).squeeze()))

    assert almost_equal(result_sparse, true_result_sparse)
    assert almost_equal(result_dense, true_result_dense)

def test_matrix_methods():
    for n in range(2, 20):
        _test_matrix_inner_real(n)
        _test_matrix_norm_real(n)
        _test_matrix_dist_real(n)
        _test_matrix_inner_complex(n)
        _test_matrix_norm_complex(n)
        _test_matrix_dist_complex(n)

def test_matrix_repr():
    n = 5
    sparse_mat = sp.sparse.dia_matrix((np.arange(n, dtype=float), [0]),
                                      shape=(n, n))
    dense_mat = sparse_mat.todense()

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_dense = _FnMatrixWeighting(dense_mat)

    mat_str_sparse = ("<(5, 5) sparse matrix, format 'dia', "
                      "5 stored entries>")
    mat_str_dense = dedent('''
    matrix([[ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  0.],
            [ 0.,  0.,  2.,  0.,  0.],
            [ 0.,  0.,  0.,  3.,  0.],
            [ 0.,  0.,  0.,  0.,  4.]])
    ''')

    repr_str_sparse = ('_FnMatrixWeighting({})'
                       ''.format(mat_str_sparse))
    repr_str_dense = '_FnMatrixWeighting({})'.format(mat_str_dense)
    assert repr(w_sparse) == repr_str_sparse
    assert repr(w_dense) == repr_str_dense

def test_matrix_str():
    n = 5
    sparse_mat = sp.sparse.dia_matrix((np.arange(n, dtype=float), [0]),
                                      shape=(n, n))
    dense_mat = sparse_mat.todense()

    w_sparse = _FnMatrixWeighting(sparse_mat)
    w_dense = _FnMatrixWeighting(dense_mat)

    mat_str_sparse = '''
  (1, 1)\t1.0
  (2, 2)\t2.0
  (3, 3)\t3.0
  (4, 4)\t4.0'''
    mat_str_dense = dedent('''
    [[ 0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.]
     [ 0.  0.  2.  0.  0.]
     [ 0.  0.  0.  3.  0.]
     [ 0.  0.  0.  0.  4.]]''')

    print_str_sparse = ('Weighting: matrix ={}'
                        ''.format(mat_str_sparse))
    assert str(w_sparse) == print_str_sparse

    print_str_dense = ('Weighting: matrix ={}'
                       ''.format(mat_str_dense))
    assert str(w_dense) == print_str_dense

def test_constant_init():
    constant = 1.5

    # Just test if the code runs
    _FnConstWeighting(constant)

def test_constant_equals():
    n = 10
    constant = 1.5

    w_const = _FnConstWeighting(constant)
    w_const2 = _FnConstWeighting(constant)

    const_sparse_mat = sp.sparse.dia_matrix(([constant]*n, [0]),
                                            shape=(n, n))
    const_dense_mat = constant * np.eye(n)
    w_matrix_sp = _FnMatrixWeighting(const_sparse_mat)
    w_matrix_de = _FnMatrixWeighting(const_dense_mat)

    assert w_const == w_const
    assert w_const == w_const2
    assert w_const2 == w_const
    # Equivalent but not equal -> False
    assert w_const != w_matrix_sp
    assert w_const != w_matrix_de

    w_different_const = _FnConstWeighting(2.5)
    assert w_const != w_different_const

def test_constant_equiv():
    n = 10
    constant = 1.5

    w_const = _FnConstWeighting(constant)
    w_const2 = _FnConstWeighting(constant)

    const_sparse_mat = sp.sparse.dia_matrix(([constant]*n, [0]),
                                            shape=(n, n))
    const_dense_mat = constant * np.eye(n)
    w_matrix_sp = _FnMatrixWeighting(const_sparse_mat)
    w_matrix_de = _FnMatrixWeighting(const_dense_mat)

    # Equal -> True
    assert w_const.equiv(w_const)
    assert w_const.equiv(w_const2)
    # Equivalent matrix representation -> True
    assert w_const.equiv(w_matrix_sp)
    assert w_const.equiv(w_matrix_de)

    w_different_const = _FnConstWeighting(2.5)
    assert not w_const.equiv(w_different_const)

def _test_constant_inner_real( n):
    rn = Rn(n)
    xarr, yarr, x, y = _vectors(rn, 2)

    constant = 1.5
    w_const = _FnConstWeighting(constant)

    result_const = w_const.inner(x, y)
    true_result_const = constant * np.dot(yarr, xarr)

    assert almost_equal(result_const, true_result_const)

def _test_constant_norm_real(n):
    rn = Rn(n)
    xarr, yarr, x, y = _vectors(rn, 2)

    constant = 1.5
    w_const = _FnConstWeighting(constant)

    result_const = w_const.norm(x)
    true_result_const = np.sqrt(constant * np.dot(xarr, xarr))

    assert almost_equal(result_const, true_result_const)

def _test_constant_dist_real(n):
    rn = Rn(n)
    xarr, yarr, x, y = _vectors(rn, 2)

    constant = 1.5
    w_const = _FnConstWeighting(constant)

    result_const = w_const.dist(x, y)
    true_result_const = np.sqrt(constant * np.dot(xarr-yarr, xarr-yarr))

    assert almost_equal(result_const, true_result_const)

def _test_constant_inner_complex(n):
    cn = Cn(n)
    xarr, yarr, x, y = _vectors(cn, 2)

    constant = 1.5
    w_const = _FnConstWeighting(constant)

    result_const = w_const.inner(x, y)
    true_result_const = constant * np.vdot(yarr, xarr)

    assert almost_equal(result_const, true_result_const)

def _test_constant_norm_complex(n):
    cn = Cn(n)
    xarr, yarr, x, y = _vectors(cn, 2)

    constant = 1.5
    w_const = _FnConstWeighting(constant)

    result_const = w_const.norm(x)
    true_result_const = np.sqrt(constant * np.vdot(xarr, xarr))

    assert almost_equal(result_const, true_result_const)

def _test_constant_dist_complex(n):
    cn = Cn(n)
    xarr, yarr, x, y = _vectors(cn, 2)

    constant = 1.5
    w_const = _FnConstWeighting(constant)

    result_const = w_const.dist(x, y)
    true_result_const = np.sqrt(constant * np.vdot(xarr-yarr, xarr-yarr))

    assert almost_equal(result_const, true_result_const)

def test_constant_methods():
    for n in range(2, 20):
        _test_constant_inner_real(n)
        _test_constant_norm_real(n)
        _test_constant_dist_real(n)
        _test_constant_inner_complex(n)
        _test_constant_norm_complex(n)
        _test_constant_dist_complex(n)

def test_constant_repr():
    constant = 1.5
    w_const = _FnConstWeighting(constant)

    repr_str = '_FnConstWeighting(1.5)'
    assert repr(w_const) == repr_str

def test_constant_str():
    constant = 1.5
    w_const = _FnConstWeighting(constant)

    print_str = 'Weighting: const = 1.5'
    assert str(w_const) == print_str

if __name__ == '__main__':
    pytest.main(str(__file__))
