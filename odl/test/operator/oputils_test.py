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

import odl
from odl.operator.oputils import matrix_representation, power_method_opnorm
from odl.operator.pspace_ops import ProductSpaceOperator
from odl.util.testutils import all_almost_equal


def test_matrix_representation():
    """Verify that the matrix repr returns the correct matrix"""
    n = 3
    A = np.random.rand(n, n)

    Aop = odl.MatrixOperator(A)
    matrix_repr = matrix_representation(Aop)

    assert all_almost_equal(A, matrix_repr)


def test_matrix_representation_product_to_lin_space():
    """Verify that the matrix repr works for product spaces.

    Here, since the domain shape ``(2, 3)`` and the range has shape ``(1, 3)``,
    the shape of the matrix representation will be ``(2, 3, 1, 3)``.
    """
    n = 3
    A = np.random.rand(n, n)
    Aop = odl.MatrixOperator(A)

    B = np.random.rand(n, n)
    Bop = odl.MatrixOperator(B)

    ABop = ProductSpaceOperator([[Aop, Bop]])
    matrix_repr = matrix_representation(ABop)

    assert matrix_repr.shape == (1, n, 2, n)
    assert np.linalg.norm(A - matrix_repr[0, :, 0, :]) == pytest.approx(0)
    assert np.linalg.norm(B - matrix_repr[0, :, 1, :]) == pytest.approx(0)


def test_matrix_representation_lin_space_to_product():
    """Verify that the matrix repr works for product spaces.

    Here, since the domain shape ``(1, 3)`` and the range has shape ``(2, 3)``,
    the shape of the matrix representation will be ``(2, 3, 1, 3)``.
    """
    n = 3
    A = np.random.rand(n, n)
    Aop = odl.MatrixOperator(A)

    B = np.random.rand(n, n)
    Bop = odl.MatrixOperator(B)

    ABop = ProductSpaceOperator([[Aop],
                                 [Bop]])

    matrix_repr = matrix_representation(ABop)

    assert matrix_repr.shape == (2, n, 1, n)
    assert np.linalg.norm(A - matrix_repr[0, :, 0, :]) == pytest.approx(0)
    assert np.linalg.norm(B - matrix_repr[1, :, 0, :]) == pytest.approx(0)


def test_matrix_representation_product_to_product():
    """Verify that the matrix repr works for product spaces.

    Here, since the domain and range has shape ``(2, 3)``, the shape of the
    matrix representation will be ``(2, 3, 2, 3)``.
    """
    n = 3
    A = np.random.rand(n, n)
    Aop = odl.MatrixOperator(A)

    B = np.random.rand(n, n)
    Bop = odl.MatrixOperator(B)

    ABop = ProductSpaceOperator([[Aop, 0],
                                 [0, Bop]])
    matrix_repr = matrix_representation(ABop)

    assert matrix_repr.shape == (2, n, 2, n)
    assert np.linalg.norm(A - matrix_repr[0, :, 0, :]) == pytest.approx(0)
    assert np.linalg.norm(B - matrix_repr[1, :, 1, :]) == pytest.approx(0)


def test_matrix_representation_not_linear_op():
    """Verify error when operator is non-linear"""
    class MyNonLinOp(odl.Operator):
        """Small nonlinear test operator."""
        def _call(self, x):
            return x ** 2

    nonlin_op = MyNonLinOp(domain=odl.rn(3), range=odl.rn(3), linear=False)
    with pytest.raises(ValueError):
        matrix_representation(nonlin_op)


def test_matrix_representation_wrong_domain():
    """Verify that the matrix representation function gives correct error"""
    class MyOp(odl.Operator):
        """Small test operator."""
        def __init__(self):
            super(MyOp, self).__init__(
                domain=odl.rn(3) * odl.rn(3) ** 2,
                range=odl.rn(4),
                linear=True)

        def _call(self, x, out):
            return odl.rn(np.random.rand(4))

    nonlin_op = MyOp()
    with pytest.raises(TypeError):
        matrix_representation(nonlin_op)


def test_matrix_representation_wrong_range():
    """Verify that the matrix representation function gives correct error"""
    class MyOp(odl.Operator):
        """Small test operator."""
        def __init__(self):
            super(MyOp, self).__init__(
                domain=odl.rn(3),
                range=odl.rn(3) * odl.rn(3) ** 2,
                linear=True)

        def _call(self, x, out):
            return odl.rn(np.random.rand(4))

    nonlin_op = MyOp()
    with pytest.raises(TypeError):
        matrix_representation(nonlin_op)


def test_power_method_opnorm_symm():
    """Test the power method on a symmetrix matrix operator"""
    # Test matrix with eigenvalues 1 and -2
    # Rather nasty case since the eigenvectors are almost parallel
    mat = np.array([[10, -18],
                    [6, -11]], dtype=float)

    op = odl.MatrixOperator(mat)
    true_opnorm = 2
    opnorm_est = power_method_opnorm(op)
    assert opnorm_est == pytest.approx(true_opnorm, rel=1e-2)

    # Start at a different point
    xstart = odl.rn(2).element([0.8, 0.5])
    opnorm_est = power_method_opnorm(op, xstart=xstart)
    assert opnorm_est == pytest.approx(true_opnorm, rel=1e-2)


def test_power_method_opnorm_nonsymm():
    """Test the power method on a nonsymmetrix matrix operator"""
    # Singular values 5.5 and 6
    mat = np.array([[-1.52441557, 5.04276365],
                    [1.90246927, 2.54424763],
                    [5.32935411, 0.04573162]])

    op = odl.MatrixOperator(mat)
    true_opnorm = 6

    # Start vector (1, 1) is close to the wrong eigenvector
    xstart = odl.rn(2).element([1, 1])
    opnorm_est = power_method_opnorm(op, xstart=xstart, maxiter=50)
    assert opnorm_est == pytest.approx(true_opnorm, rel=1e-2)

    # Start close to the correct eigenvector, converges very fast
    xstart = odl.rn(2).element([-0.8, 0.5])
    opnorm_est = power_method_opnorm(op, xstart=xstart, maxiter=6)
    assert opnorm_est == pytest.approx(true_opnorm, rel=1e-2)


def test_power_method_opnorm_exceptions():
    """Test the exceptions"""
    space = odl.rn(2)
    op = odl.IdentityOperator(space)

    with pytest.raises(ValueError):
        # Too small number of iterates
        power_method_opnorm(op, maxiter=0)

    with pytest.raises(ValueError):
        # Negative number of iterates
        power_method_opnorm(op, maxiter=-5)

    with pytest.raises(ValueError):
        # Input vector is zero
        power_method_opnorm(op, maxiter=2, xstart=space.zero())

    with pytest.raises(ValueError):
        # Input vector in the nullspace
        op = odl.MatrixOperator([[0., 1.],
                                 [0., 0.]])

        power_method_opnorm(op, maxiter=2, xstart=op.domain.one())

    with pytest.raises(ValueError):
        # Uneven number of iterates for non square operator
        op = odl.MatrixOperator([[1., 2., 3.],
                                 [4., 5., 6.]])

        power_method_opnorm(op, maxiter=1, xstart=op.domain.one())


if __name__ == '__main__':
    odl.util.test_file(__file__)
