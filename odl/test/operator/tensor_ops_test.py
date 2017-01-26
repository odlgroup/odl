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
from odl.util import moveaxis
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


matrix_dtype = simple_fixture(
    name='matrix_dtype',
    params=['float32', 'complex64', 'float64', 'complex128'])


@pytest.fixture(scope='module')
def matrix(matrix_dtype):
    dtype = np.dtype(matrix_dtype)
    if np.issubsctype(dtype, np.floating):
        return np.ones((3, 4), dtype=dtype)
    elif np.issubsctype(dtype, np.complexfloating):
        return np.ones((3, 4), dtype=dtype) * (1 + 1j)
    else:
        assert 0


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


def test_matrix_op_init(matrix):
    """Test initialization and properties of matrix operators."""
    dense_matrix = matrix
    sparse_matrix = scipy.sparse.coo_matrix(dense_matrix)

    # Just check if the code runs
    MatrixOperator(dense_matrix)
    MatrixOperator(sparse_matrix)

    # Test default domain and range
    mat_op = MatrixOperator(dense_matrix)
    assert mat_op.domain == odl.tensor_space(4, matrix.dtype)
    assert mat_op.range == odl.tensor_space(3, matrix.dtype)
    assert np.all(mat_op.matrix == dense_matrix)
    sparse_matrix = scipy.sparse.coo_matrix(dense_matrix)
    mat_op = MatrixOperator(sparse_matrix)
    assert mat_op.domain == odl.tensor_space(4, matrix.dtype)
    assert mat_op.range == odl.tensor_space(3, matrix.dtype)
    assert (mat_op.matrix != sparse_matrix).getnnz() == 0

    # Explicit domain and range
    dom = odl.tensor_space(4, matrix.dtype)
    ran = odl.tensor_space(3, matrix.dtype)
    mat_op = MatrixOperator(dense_matrix, domain=dom, range=ran)
    assert mat_op.domain == dom
    assert mat_op.range == ran
    mat_op = MatrixOperator(sparse_matrix, domain=dom, range=ran)
    assert mat_op.domain == dom
    assert mat_op.range == ran

    # Bad 1d sizes
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix, domain=odl.cn(4), range=odl.cn(4))
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix, range=odl.cn(4))
    # Invalid range dtype
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix.astype(complex), range=odl.rn(4))

    # Data type promotion
    # real space, complex matrix -> complex space
    dom = odl.rn(4)
    mat_op = MatrixOperator(dense_matrix.astype(complex), domain=dom)
    assert mat_op.domain == dom
    assert mat_op.range == odl.cn(3)

    # complex space, real matrix -> complex space
    dom = odl.cn(4)
    mat_op = MatrixOperator(dense_matrix.real, domain=dom)
    assert mat_op.domain == dom
    assert mat_op.range == odl.cn(3)

    # Multi-dimensional spaces
    dom = odl.tensor_space((6, 5, 4), matrix.dtype)
    ran = odl.tensor_space((6, 5, 3), matrix.dtype)
    mat_op = MatrixOperator(dense_matrix, domain=dom, axis=2)
    assert mat_op.range == ran
    mat_op = MatrixOperator(dense_matrix, domain=dom, range=ran, axis=2)
    assert mat_op.range == ran

    with pytest.raises(ValueError):
        bad_dom = odl.tensor_space((6, 6, 6), matrix.dtype)  # wrong shape
        MatrixOperator(dense_matrix, domain=bad_dom)
    with pytest.raises(ValueError):
        dom = odl.tensor_space((6, 5, 4), matrix.dtype)
        bad_ran = odl.tensor_space((6, 6, 6), matrix.dtype)  # wrong shape
        MatrixOperator(dense_matrix, domain=dom, range=bad_ran)
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix, domain=dom, axis=1)
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix, domain=dom, axis=0)
    with pytest.raises(ValueError):
        bad_ran = odl.tensor_space((6, 3, 4), matrix.dtype)
        MatrixOperator(dense_matrix, domain=dom, range=bad_ran, axis=2)
    with pytest.raises(ValueError):
        bad_dom_for_sparse = odl.rn((6, 5, 4))
        MatrixOperator(sparse_matrix, domain=bad_dom_for_sparse, axis=2)

    # Init with uniform_discr space (subclass of FnBase)
    dom = odl.uniform_discr(0, 1, 4, dtype=dense_matrix.dtype)
    ran = odl.uniform_discr(0, 1, 3, dtype=dense_matrix.dtype)
    MatrixOperator(dense_matrix, domain=dom, range=ran)

    # Make sure this runs at all
    str(mat_op)
    repr(mat_op)


def test_matrix_op_call(matrix):
    """Validate result from calls to matrix operators against Numpy."""
    dense_matrix = matrix
    sparse_matrix = scipy.sparse.coo_matrix(dense_matrix)

    # Default 1d case
    dmat_op = MatrixOperator(dense_matrix)
    smat_op = MatrixOperator(sparse_matrix)
    xarr, x = noise_elements(dmat_op.domain)

    true_result_dense = dense_matrix.dot(xarr)
    true_result_sparse = sparse_matrix.dot(xarr)
    assert all_almost_equal(dmat_op(x), true_result_dense)
    assert all_almost_equal(smat_op(x), true_result_sparse)
    out = dmat_op.range.element()
    dmat_op(x, out=out)
    assert all_almost_equal(out, true_result_dense)
    smat_op(x, out=out)
    assert all_almost_equal(out, true_result_sparse)

    # Multi-dimensional case
    domain = odl.rn((2, 2, 4))
    mat_op = MatrixOperator(dense_matrix, domain, axis=2)
    xarr, x = noise_elements(mat_op.domain)
    true_result = moveaxis(np.tensordot(dense_matrix, xarr, (1, 2)), 0, 2)
    assert all_almost_equal(mat_op(x), true_result)
    out = mat_op.range.element()
    mat_op(x, out=out)
    assert all_almost_equal(out, true_result)


def test_matrix_op_call_explicit():
    """Validate result from call to matrix op against explicit calculation."""
    mat = np.ones((3, 2))
    xarr = np.array([[[0, 1],
                      [2, 3]],
                     [[4, 5],
                      [6, 7]]], dtype=float)

    # Multiplication along `axis` with `mat` is the same as summation
    # along `axis` and stacking 3 times along the same axis
    for axis in range(3):
        mat_op = MatrixOperator(mat, domain=odl.rn(xarr.shape),
                                axis=axis)
        result = mat_op(xarr)
        true_result = np.repeat(np.sum(xarr, axis=axis, keepdims=True),
                                repeats=3, axis=axis)
        assert result.shape == true_result.shape
        assert np.allclose(result, true_result)


def test_matrix_op_adjoint(matrix):
    """Test if the adjoint of matrix operators is correct."""
    dense_matrix = matrix
    sparse_matrix = scipy.sparse.coo_matrix(dense_matrix)

    tol = 2 * matrix.size * np.finfo(matrix.dtype).resolution

    # Default 1d case
    dmat_op = MatrixOperator(dense_matrix)
    smat_op = MatrixOperator(sparse_matrix)
    x = noise_element(dmat_op.domain)
    y = noise_element(dmat_op.range)

    inner_ran = dmat_op(x).inner(y)
    inner_dom = x.inner(dmat_op.adjoint(y))
    assert inner_ran == pytest.approx(inner_dom, rel=tol, abs=tol)
    inner_ran = smat_op(x).inner(y)
    inner_dom = x.inner(smat_op.adjoint(y))
    assert inner_ran == pytest.approx(inner_dom, rel=tol, abs=tol)

    # Multi-dimensional case
    domain = odl.tensor_space((2, 2, 4), matrix.dtype)
    mat_op = MatrixOperator(dense_matrix, domain, axis=2)
    x = noise_element(mat_op.domain)
    y = noise_element(mat_op.range)
    inner_ran = mat_op(x).inner(y)
    inner_dom = x.inner(mat_op.adjoint(y))
    assert inner_ran == pytest.approx(inner_dom, rel=tol, abs=tol)


def test_matrix_op_inverse():
    """Test if the inverse of matrix operators is correct."""
    dense_matrix = np.ones((3, 3)) + 4 * np.eye(3)  # invertible
    sparse_matrix = scipy.sparse.coo_matrix(dense_matrix)

    # Default 1d case
    dmat_op = MatrixOperator(dense_matrix)
    smat_op = MatrixOperator(sparse_matrix)
    x = noise_element(dmat_op.domain)
    md_x = dmat_op(x)
    mdinv_md_x = dmat_op.inverse(md_x)
    assert all_almost_equal(x, mdinv_md_x)
    ms_x = smat_op(x)
    msinv_ms_x = smat_op.inverse(ms_x)
    assert all_almost_equal(x, msinv_ms_x)

    # Multi-dimensional case
    domain = odl.tensor_space((2, 2, 3), dense_matrix.dtype)
    mat_op = MatrixOperator(dense_matrix, domain, axis=2)
    x = noise_element(mat_op.domain)
    m_x = mat_op(x)
    minv_m_x = mat_op.inverse(m_x)
    assert all_almost_equal(x, minv_m_x)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
