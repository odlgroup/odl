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
from odl.util.testutils import all_almost_equal, noise_elements

from odl.array_API_support.utils import get_array_and_backend

@pytest.fixture(scope="module", ids=['True', 'False'], params=[True, False])
def dom_eq_ran_mat(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    shape = (3,3)
    space = odl.rn(shape, impl=impl, device=device)
    mat, _ = noise_elements(space)
    return mat


def test_matrix_representation(dom_eq_ran_mat):
    """Verify that the matrix repr returns the correct matrix"""
    Aop = odl.MatrixOperator(dom_eq_ran_mat)
    matrix_repr = matrix_representation(Aop)

    assert all_almost_equal(dom_eq_ran_mat, matrix_repr)


def test_matrix_representation_product_to_lin_space(dom_eq_ran_mat):
    """Verify that the matrix repr works for product spaces.

    Here, since the domain shape ``(2, 3)`` and the range has shape ``(1, 3)``,
    the shape of the matrix representation will be ``(2, 3, 1, 3)``.
    """
    A = dom_eq_ran_mat
    Aop = odl.MatrixOperator(A)

    B = dom_eq_ran_mat+0.1
    Bop = odl.MatrixOperator(B)

    ABop = ProductSpaceOperator([[Aop, Bop]])
    matrix_repr = matrix_representation(ABop)

    assert matrix_repr.shape == (1, 3, 2, 3)

    _, backend = get_array_and_backend(A)
    
    assert backend.to_cpu(
        backend.array_namespace.linalg.norm(A - matrix_repr[0, :, 0, :])) == pytest.approx(0)
    assert backend.to_cpu(
        backend.array_namespace.linalg.norm(B - matrix_repr[0, :, 1, :])) == pytest.approx(0)


def test_matrix_representation_lin_space_to_product(dom_eq_ran_mat):
    """Verify that the matrix repr works for product spaces.

    Here, since the domain shape ``(1, 3)`` and the range has shape ``(2, 3)``,
    the shape of the matrix representation will be ``(2, 3, 1, 3)``.
    """
    n=3
    A = dom_eq_ran_mat
    Aop = odl.MatrixOperator(A)

    B = dom_eq_ran_mat+0.1
    Bop = odl.MatrixOperator(B)

    ABop = ProductSpaceOperator([[Aop],
                                 [Bop]])

    matrix_repr = matrix_representation(ABop)

    _, backend = get_array_and_backend(A)
    assert matrix_repr.shape == (2, n, 1, n)
    assert backend.to_cpu(
        backend.array_namespace.linalg.norm(A - matrix_repr[0, :, 0, :])) == pytest.approx(0)
    assert backend.to_cpu(
        backend.array_namespace.linalg.norm(B - matrix_repr[1, :, 0, :])) == pytest.approx(0)


def test_matrix_representation_product_to_product(dom_eq_ran_mat):
    """Verify that the matrix repr works for product spaces.

    Here, since the domain and range has shape ``(2, 3)``, the shape of the
    matrix representation will be ``(2, 3, 2, 3)``.
    """
    n = 3
    A = dom_eq_ran_mat
    Aop = odl.MatrixOperator(A)

    B = dom_eq_ran_mat+0.1
    Bop = odl.MatrixOperator(B)

    ABop = ProductSpaceOperator([[Aop, 0],
                                 [0, Bop]])
    matrix_repr = matrix_representation(ABop)

    assert matrix_repr.shape == (2, n, 2, n)
    _, backend = get_array_and_backend(A)
    assert matrix_repr.shape == (2, n, 2, n)
    assert backend.to_cpu(
        backend.array_namespace.linalg.norm(A - matrix_repr[0, :, 0, :])) == pytest.approx(0)
    assert backend.to_cpu(
        backend.array_namespace.linalg.norm(B - matrix_repr[1, :, 1, :])) == pytest.approx(0)



def test_matrix_representation_not_linear_op(odl_impl_device_pairs):
    """Verify error when operator is non-linear"""
    impl, device = odl_impl_device_pairs
    class MyNonLinOp(odl.Operator):
        """Small nonlinear test operator."""
        def _call(self, x):
            return x ** 2

    nonlin_op = MyNonLinOp(
        domain=odl.rn(3,impl=impl, device=device), 
        range=odl.rn(3,impl=impl, device=device), 
        linear=False)
    with pytest.raises(ValueError):
        matrix_representation(nonlin_op)


def test_matrix_representation_wrong_domain(odl_impl_device_pairs):
    """Verify that the matrix representation function gives correct error"""
    impl, device = odl_impl_device_pairs
    class MyOp(odl.Operator):
        """Small test operator."""
        def __init__(self):
            super(MyOp, self).__init__(
                domain=odl.rn(3,impl=impl, device=device) * odl.rn(3,impl=impl, device=device) ** 2,
                range=odl.rn(4,impl=impl, device=device),
                linear=True)

        def _call(self, x, out):
            return odl.rn([4], impl=impl, device=device)

    nonlin_op = MyOp()
    with pytest.raises(TypeError):
        matrix_representation(nonlin_op)


def test_matrix_representation_wrong_range(odl_impl_device_pairs):
    """Verify that the matrix representation function gives correct error"""
    impl, device = odl_impl_device_pairs
    class MyOp(odl.Operator):
        """Small test operator."""
        def __init__(self):
            super(MyOp, self).__init__(
                domain=odl.rn(3,impl=impl, device=device),
                range=odl.rn(3,impl=impl, device=device) * odl.rn(3,impl=impl, device=device) ** 2,
                linear=True)

        def _call(self, x, out):
             return odl.rn([4], impl=impl, device=device)

    nonlin_op = MyOp()
    with pytest.raises(TypeError):
        matrix_representation(nonlin_op)


def test_power_method_opnorm_symm(odl_impl_device_pairs):
    """Test the power method on a symmetrix matrix operator"""
    impl, device = odl_impl_device_pairs
    # Test matrix with singular values 1.2 and 1.0
    space = odl.rn([2,2], impl=impl, device=device)
    mat = space.element([[0.9509044, -0.64566614],
                    [-0.44583952, -0.95923051]])

    op = odl.MatrixOperator(mat)
    true_opnorm = 1.2

    # Run with a random starting point. This should give the right
    # result most of the time, but can fail if the random point fell
    # very close to the lesser eigenvector. To confirm that the probability
    # is low, we run the method three times (they will have different
    # random starting points). Usually all three should give the right
    # result, but one of them may occasionally fail (and give the other
    # singular value instead). Two wrong ones out of three should
    # be exeedingly rare though, so we treat this as a test failure.
    n_failures = 0
    for _ in range(3):
        opnorm_est = power_method_opnorm(op, maxiter=100)
        if n_failures<1:
            if opnorm_est != pytest.approx(true_opnorm, rel=1e-2):
                n_failures += 1
        else:
            assert opnorm_est == pytest.approx(true_opnorm, rel=1e-2)

    # Start at a deterministic point. This should _always_ succeed.
    xstart = odl.rn(2, impl=impl, device=device).element([0.8, 0.5])
    opnorm_est = power_method_opnorm(op, xstart=xstart, maxiter=100)
    assert opnorm_est == pytest.approx(true_opnorm, rel=1e-2)


def test_power_method_opnorm_nonsymm(odl_impl_device_pairs):
    """Test the power method on a nonsymmetrix matrix operator"""
    impl, device = odl_impl_device_pairs
    # Singular values 5.5 and 6
    space = odl.rn([3,2], impl=impl, device=device)
    mat = space.element([[-1.52441557, 5.04276365],
                    [1.90246927, 2.54424763],
                    [5.32935411, 0.04573162]])

    op = odl.MatrixOperator(mat)
    true_opnorm = 6

    # Start vector (1, 1) is close to the wrong eigenvector
    xstart = odl.rn(2, impl=impl, device=device).element([1, 1])
    opnorm_est = power_method_opnorm(op, xstart=xstart, maxiter=50)
    assert opnorm_est == pytest.approx(true_opnorm, rel=1e-2)

    # Start close to the correct eigenvector, converges very fast
    xstart = odl.rn(2, impl=impl, device=device).element([-0.8, 0.5])
    opnorm_est = power_method_opnorm(op, xstart=xstart, maxiter=6)
    assert opnorm_est == pytest.approx(true_opnorm, rel=1e-2)


def test_power_method_opnorm_exceptions(odl_impl_device_pairs):
    """Test the exceptions"""
    impl, device = odl_impl_device_pairs
    space = odl.rn(2, impl=impl, device=device)
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
        # Input vector in the null space
        op = odl.MatrixOperator([[0., 1.],
                                 [0., 0.]])

        power_method_opnorm(op, maxiter=2, xstart=[1, 0])

    with pytest.raises(ValueError):
        # Uneven number of iterates for non square operator
        op = odl.MatrixOperator([[1., 2., 3.],
                                 [4., 5., 6.]])

        power_method_opnorm(op, maxiter=1, xstart=op.domain.one())


if __name__ == '__main__':
    odl.util.test_file(__file__)
