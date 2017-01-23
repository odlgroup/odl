# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `tensor_ops`."""

from __future__ import division
import pytest
import numpy as np
import scipy

import odl
from odl.operator.tensor_ops import (
    PointwiseNorm, PointwiseInner, PointwiseSum, MatrixOperator)
from odl.space.pspace import ProductSpace
from odl.util.testutils import (
    all_almost_equal, all_equal, simple_fixture, noise_element, noise_elements)


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


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 3.5, 1.5])
fn = simple_fixture('fn', [odl.rn(10, np.float64), odl.rn(10, np.float32),
                           odl.cn(10, np.complex128), odl.cn(10, np.complex64),
                           odl.rn(100), odl.uniform_discr(0, 1, 5)])


# ---- PointwiseNorm ----


def test_pointwise_norm_init_properties():
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 1, exponent=1)

    # Make sure the code runs and test the properties
    pwnorm = PointwiseNorm(vfspace)
    assert pwnorm.base_space == fspace
    assert all_equal(pwnorm.weights, [1])
    assert not pwnorm.is_weighted
    assert pwnorm.exponent == 1.0
    repr(pwnorm)

    pwnorm = PointwiseNorm(vfspace, exponent=2)
    assert pwnorm.exponent == 2

    pwnorm = PointwiseNorm(vfspace, weighting=2)
    assert all_equal(pwnorm.weights, [2])
    assert pwnorm.is_weighted

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3, exponent=1)

    # Make sure the code runs and test the properties
    pwnorm = PointwiseNorm(vfspace)
    assert pwnorm.base_space == fspace
    assert all_equal(pwnorm.weights, [1, 1, 1])
    assert not pwnorm.is_weighted
    assert pwnorm.exponent == 1.0
    repr(pwnorm)

    pwnorm = PointwiseNorm(vfspace, exponent=2)
    assert pwnorm.exponent == 2

    pwnorm = PointwiseNorm(vfspace, weighting=[1, 2, 3])
    assert all_equal(pwnorm.weights, [1, 2, 3])
    assert pwnorm.is_weighted

    # Bad input
    with pytest.raises(TypeError):
        PointwiseNorm(odl.rn(3))  # No power space

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, exponent=0.5)  # < 1 not allowed

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, weighting=-1)  # < 0 not allowed

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, weighting=[1, 0, 1])  # 0 invalid


def test_pointwise_norm_real(exponent):
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 1)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = np.array([[[1, 2],
                         [3, 4]]])

    true_norm = np.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm)

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm)

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = np.array([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]])

    true_norm = np.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm)

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm)


def test_pointwise_norm_complex(exponent):
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 3)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = np.array([[[1 + 1j, 2],
                         [3, 4 - 2j]],
                        [[0, -1],
                         [0, 1]],
                        [[1j, 1j],
                         [1j, 1j]]])

    true_norm = np.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm)

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm)


def test_pointwise_norm_weighted(exponent):
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3)
    weight = np.array([1.0, 2.0, 3.0])
    pwnorm = PointwiseNorm(vfspace, exponent, weighting=weight)

    testarr = np.array([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]])

    if exponent in (1.0, float('inf')):
        true_norm = np.linalg.norm(weight[:, None, None] * testarr,
                                   ord=exponent, axis=0)
    else:
        true_norm = np.linalg.norm(
            weight[:, None, None] ** (1 / exponent) * testarr, ord=exponent,
            axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm)

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm)


# ---- PointwiseInner ----


def test_pointwise_inner_init_properties():
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3, exponent=2)

    # Make sure the code runs and test the properties
    pwinner = PointwiseInner(vfspace, vfspace.one())
    assert pwinner.base_space == fspace
    assert all_equal(pwinner.weights, [1, 1, 1])
    assert not pwinner.is_weighted
    repr(pwinner)

    pwinner = PointwiseInner(vfspace, vfspace.one(), weighting=[1, 2, 3])
    assert all_equal(pwinner.weights, [1, 2, 3])
    assert pwinner.is_weighted

    # Bad input
    with pytest.raises(TypeError):
        PointwiseInner(odl.rn(3), odl.rn(3).one())  # No power space

    # TODO: Does not raise currently, although bad_vecfield not in vfspace!
    """
    bad_vecfield = ProductSpace(fspace, 3, exponent=1).one()
    with pytest.raises(TypeError):
        PointwiseInner(vfspace, bad_vecfield)
    """


def test_pointwise_inner_real():
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 1)
    array = np.array([[[-1, -3],
                       [2, 0]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[[1, 2],
                         [3, 4]]])

    true_inner = np.sum(testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner)

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner)

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3)
    array = np.array([[[-1, -3],
                       [2, 0]],
                      [[0, 0],
                       [0, 1]],
                      [[-1, 1],
                       [1, 1]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]])

    true_inner = np.sum(testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner)

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner)


def test_pointwise_inner_complex():
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 3)
    array = np.array([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[[1 + 1j, 2],
                         [3, 4 - 2j]],
                        [[0, -1],
                         [0, 1]],
                        [[1j, 1j],
                         [1j, 1j]]])

    true_inner = np.sum(testarr * array.conj(), axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner)

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner)


def test_pointwise_inner_weighted():
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3)
    array = np.array([[[-1, -3],
                       [2, 0]],
                      [[0, 0],
                       [0, 1]],
                      [[-1, 1],
                       [1, 1]]])

    weight = np.array([1.0, 2.0, 3.0])
    pwinner = PointwiseInner(vfspace, vecfield=array, weighting=weight)

    testarr = np.array([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]])

    true_inner = np.sum(weight[:, None, None] * testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner)

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner)


def test_pointwise_inner_adjoint():
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 1)
    array = np.array([[[-1, -3],
                       [2, 0]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[1 + 1j, 2],
                        [3, 4 - 2j]])

    true_inner_adj = testarr[None, :, :] * array

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj, true_inner_adj)

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj)

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 3)
    array = np.array([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[1 + 1j, 2],
                        [3, 4 - 2j]])

    true_inner_adj = testarr[None, :, :] * array

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj, true_inner_adj)

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj)


def test_pointwise_inner_adjoint_weighted():
    # Weighted product space only
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 3, weighting=[2, 4, 6])
    array = np.array([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[1 + 1j, 2],
                        [3, 4 - 2j]])

    true_inner_adj = testarr[None, :, :] * array  # same as unweighted case

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj, true_inner_adj)

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj)

    # Using different weighting in the inner product
    pwinner = PointwiseInner(vfspace, vecfield=array, weighting=[4, 8, 12])

    testarr = np.array([[1 + 1j, 2],
                        [3, 4 - 2j]])

    true_inner_adj = 2 * testarr[None, :, :] * array  # w / v = (2, 2, 2)

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj, true_inner_adj)

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj)


# ---- PointwiseSum ----


def test_pointwise_sum():
    """PointwiseSum currently depends on PointwiseInner, we verify that."""

    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3, exponent=2)

    # Make sure the code runs and test the properties
    psum = PointwiseSum(vfspace)
    assert isinstance(psum, PointwiseInner)
    assert psum.base_space == fspace
    assert all_equal(psum.weights, [1, 1, 1])
    assert all_equal(psum.vecfield, psum.domain.one())


# ---- MatrixOperator ---- #


def test_mat_op_init_and_basic_properties():
    """Test initialization and basic properties of MatrixOperator."""
    # Test default domain and range
    r2 = odl.rn(2)
    op_real = MatrixOperator([[1.0, 2],
                              [-1, 0.5]])

    assert op_real.domain == r2
    assert op_real.range == r2

    c2 = odl.cn(2)
    op_complex = MatrixOperator([[1.0, 2 + 1j],
                                 [-1 - 1j, 0.5]])

    assert op_complex.domain == c2
    assert op_complex.range == c2

    int2 = odl.fn(2, dtype=int)
    op_int = MatrixOperator([[1, 2],
                             [-1, 0]])

    assert op_int.domain == int2
    assert op_int.range == int2

    # Rectangular
    rect_mat = 2 * np.eye(2, 3)
    r3 = odl.rn(3)

    op = MatrixOperator(rect_mat)
    assert op.domain == r3
    assert op.range == r2

    MatrixOperator(rect_mat, domain=r3, range=r2)

    with pytest.raises(ValueError):
        MatrixOperator(rect_mat, domain=r2, range=r2)

    with pytest.raises(ValueError):
        MatrixOperator(rect_mat, domain=r3, range=r3)

    with pytest.raises(ValueError):
        MatrixOperator(rect_mat, domain=r2, range=r3)

    # Rn to Cn okay
    MatrixOperator(rect_mat, domain=r3, range=odl.cn(2))

    # Cn to Rn not okay (no safe cast)
    with pytest.raises(TypeError):
        MatrixOperator(rect_mat, domain=odl.cn(3), range=r2)

    # Complex matrix between real spaces not okay
    rect_complex_mat = rect_mat + 1j
    with pytest.raises(TypeError):
        MatrixOperator(rect_complex_mat, domain=r3, range=r2)

    # Init with array-like structure (including numpy.matrix)
    op = MatrixOperator(rect_mat, domain=r3, range=r2)
    assert isinstance(op.matrix, np.ndarray)

    op = MatrixOperator(np.asmatrix(rect_mat), domain=r3, range=r2)
    assert isinstance(op.matrix, np.ndarray)

    op = MatrixOperator(rect_mat.tolist(), domain=r3, range=r2)
    assert isinstance(op.matrix, np.ndarray)
    assert not op.matrix_issparse

    sparse_mat = _sparse_matrix(odl.rn(5))
    op = MatrixOperator(sparse_mat, domain=odl.rn(5), range=odl.rn(5))
    assert isinstance(op.matrix, scipy.sparse.spmatrix)
    assert op.matrix_issparse

    # Init with uniform_discr space (subclass of FnBase)
    dom = odl.uniform_discr(0, 1, 3)
    ran = odl.uniform_discr(0, 1, 2)
    MatrixOperator(rect_mat, domain=dom, range=ran)


def test_mat_op_adjoint(fn):
    # Square cases
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)

    op_sparse = MatrixOperator(sparse_mat, fn, fn)
    op_dense = MatrixOperator(dense_mat, fn, fn)

    # Just test if it runs, nothing interesting to test here
    op_sparse.adjoint
    op_dense.adjoint

    # Rectangular case
    rect_mat = 2 * np.eye(2, 3)
    r2, r3 = odl.rn(2), odl.rn(3)
    c2 = odl.cn(2)

    op = MatrixOperator(rect_mat, r3, r2)
    op_adj = op.adjoint
    assert op_adj.domain == op.range
    assert op_adj.range == op.domain
    assert np.array_equal(op_adj.matrix, op.matrix.conj().T)
    assert np.array_equal(op_adj.adjoint.matrix, op.matrix)

    # The operator Rn -> Cn has no adjoint
    op_noadj = MatrixOperator(rect_mat, r3, c2)
    with pytest.raises(NotImplementedError):
        op_noadj.adjoint


def test_mat_op_inverse(fn):
    # Sparse case
    sparse_mat = _sparse_matrix(fn)
    op_sparse = MatrixOperator(sparse_mat, fn, fn)

    op_sparse_inv = op_sparse.inverse
    assert op_sparse_inv.domain == op_sparse.range
    assert op_sparse_inv.range == op_sparse.domain
    assert all_almost_equal(op_sparse_inv.matrix,
                            np.linalg.inv(op_sparse.matrix.todense()))
    assert all_almost_equal(op_sparse_inv.inverse.matrix,
                            op_sparse.matrix.todense())

    # Test application
    x = noise_element(fn)
    assert all_almost_equal(x, op_sparse.inverse(op_sparse(x)))

    # Dense case
    dense_mat = _dense_matrix(fn)
    op_dense = MatrixOperator(dense_mat, fn, fn)
    op_dense_inv = op_dense.inverse
    assert op_dense_inv.domain == op_dense.range
    assert op_dense_inv.range == op_dense.domain
    assert all_almost_equal(op_dense_inv.matrix,
                            np.linalg.inv(op_dense.matrix))
    assert all_almost_equal(op_dense_inv.inverse.matrix,
                            op_dense.matrix)

    # Test application
    x = noise_element(fn)
    assert all_almost_equal(x, op_dense.inverse(op_dense(x)))


def test_mat_op_call(fn):
    # Square cases
    sparse_mat = _sparse_matrix(fn)
    dense_mat = _dense_matrix(fn)
    xarr, x = noise_elements(fn)

    op_sparse = MatrixOperator(sparse_mat, fn, fn)
    op_dense = MatrixOperator(dense_mat, fn, fn)

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
    r2, r3 = odl.rn(2), odl.rn(3)

    op = MatrixOperator(rect_mat, r3, r2)
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


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
