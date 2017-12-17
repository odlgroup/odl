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
from odl.operator.oputils import (
    matrix_representation, power_method_opnorm, auto_adjoint_weighting)
from odl.operator.pspace_ops import ProductSpaceOperator
from odl.space.pspace import ProductSpace
from odl.util.testutils import (
    almost_equal, all_equal, simple_fixture, noise_element)


optimize_weighting = simple_fixture('optimize', [True, False])
call_variant = simple_fixture('call_variant', ['oop', 'ip', 'dual'])
weighting = simple_fixture('weighting', [1.0, 2.0, [1.0, 2.0]])


def test_matrix_representation():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    A = np.random.rand(n, n)

    Aop = odl.MatrixOperator(A)
    matrix_repr = matrix_representation(Aop)

    assert almost_equal(np.sum(np.abs(A - matrix_repr)), 1e-6)


def test_matrix_representation_product_to_lin_space():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.rn(n)
    A = np.random.rand(n, n)
    Aop = odl.MatrixOperator(A)

    m = 2
    rm = odl.rn(m)
    B = np.random.rand(n, m)
    Bop = odl.MatrixOperator(B)

    dom = ProductSpace(rn, rm)
    ran = ProductSpace(rn, 1)

    AB_matrix = np.hstack([A, B])
    ABop = ProductSpaceOperator([[Aop, Bop]], dom, ran)

    matrix_repr = matrix_representation(ABop)

    assert almost_equal(np.sum(np.abs(AB_matrix - matrix_repr)), 1e-6)


def test_matrix_representation_lin_space_to_product():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.rn(n)
    A = np.random.rand(n, n)
    Aop = odl.MatrixOperator(A)

    m = 2
    rm = odl.rn(m)
    B = np.random.rand(m, n)
    Bop = odl.MatrixOperator(B)

    dom = ProductSpace(rn, 1)
    ran = ProductSpace(rn, rm)

    AB_matrix = np.vstack([A, B])
    ABop = ProductSpaceOperator([[Aop], [Bop]], dom, ran)

    matrix_repr = matrix_representation(ABop)

    assert almost_equal(np.sum(np.abs(AB_matrix - matrix_repr)), 1e-6)


def test_matrix_representation_product_to_product():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.rn(n)
    A = np.random.rand(n, n)
    Aop = odl.MatrixOperator(A)

    m = 2
    rm = odl.rn(m)
    B = np.random.rand(m, m)
    Bop = odl.MatrixOperator(B)

    ran_and_dom = ProductSpace(rn, rm)

    AB_matrix = np.vstack([np.hstack([A, np.zeros((n, m))]),
                           np.hstack([np.zeros((m, n)), B])])
    ABop = ProductSpaceOperator([[Aop, 0],
                                 [0, Bop]],
                                ran_and_dom, ran_and_dom)
    matrix_repr = matrix_representation(ABop)

    assert almost_equal(np.sum(np.abs(AB_matrix - matrix_repr)), 1e-6)


def test_matrix_representation_product_to_product_two():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.rn(n)
    A = np.random.rand(n, n)
    Aop = odl.MatrixOperator(A)

    B = np.random.rand(n, n)
    Bop = odl.MatrixOperator(B)

    ran_and_dom = ProductSpace(rn, 2)

    AB_matrix = np.vstack([np.hstack([A, np.zeros((n, n))]),
                           np.hstack([np.zeros((n, n)), B])])
    ABop = ProductSpaceOperator([[Aop, 0],
                                 [0, Bop]],
                                ran_and_dom, ran_and_dom)
    matrix_repr = matrix_representation(ABop)

    assert almost_equal(np.sum(np.abs(AB_matrix - matrix_repr)), 1e-6)


def test_matrix_representation_not_linear_op():
    # Verify that the matrix representation function gives correct error
    class MyNonLinOp(odl.Operator):
        """Small nonlinear test operator."""
        def __init__(self):
            super(MyNonLinOp, self).__init__(
                domain=odl.rn(3), range=odl.rn(4), linear=False)

        def _call(self, x):
            return x ** 2

    nonlin_op = MyNonLinOp()
    with pytest.raises(ValueError):
        matrix_representation(nonlin_op)


def test_matrix_representation_wrong_domain():
    # Verify that the matrix representation function gives correct error
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
    # Verify that the matrix representation function gives correct error
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
    # Test the power method on a matrix operator

    # Test matrix with eigenvalues 1 and -2
    # Rather nasty case since the eigenvectors are almost parallel
    mat = np.array([[10, -18],
                    [6, -11]], dtype=float)

    op = odl.MatrixOperator(mat)
    true_opnorm = 2
    opnorm_est = power_method_opnorm(op)
    assert almost_equal(opnorm_est, true_opnorm, places=2)

    # Start at a different point
    xstart = odl.rn(2).element([0.8, 0.5])
    opnorm_est = power_method_opnorm(op, xstart=xstart)
    assert almost_equal(opnorm_est, true_opnorm, places=2)


def test_power_method_opnorm_nonsymm():
    # Test the power method on a matrix operator

    # Singular values 5.5 and 6
    mat = np.array([[-1.52441557, 5.04276365],
                    [1.90246927, 2.54424763],
                    [5.32935411, 0.04573162]])

    op = odl.MatrixOperator(mat)
    true_opnorm = 6

    # Start vector (1, 1) is close to the wrong eigenvector
    opnorm_est = power_method_opnorm(op, maxiter=50)
    assert almost_equal(opnorm_est, true_opnorm, places=2)

    # Start close to the correct eigenvector, converges very fast
    xstart = odl.rn(2).element([-0.8, 0.5])
    opnorm_est = power_method_opnorm(op, maxiter=6, xstart=xstart)
    assert almost_equal(opnorm_est, true_opnorm, places=2)


def test_power_method_opnorm_exceptions():
    # Test the exceptions

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


def test_auto_weighting(call_variant, weighting, optimize_weighting):
    """Test the auto_weighting decorator for different adjoint variants."""

    class ScalingOpBase(odl.Operator):

        def __init__(self, dom, ran, c):
            super(ScalingOpBase, self).__init__(dom, ran, linear=True)
            self.c = c

    if call_variant == 'oop':

        class ScalingOp(ScalingOpBase):

            def _call(self, x):
                return self.c * x

            @property
            @auto_adjoint_weighting(optimize=optimize_weighting)
            def adjoint(self):
                return ScalingOp(self.range, self.domain, self.c)

    elif call_variant == 'ip':

        class ScalingOp(ScalingOpBase):

            def _call(self, x, out):
                out[:] = self.c * x
                return out

            @property
            @auto_adjoint_weighting(optimize=optimize_weighting)
            def adjoint(self):
                return ScalingOp(self.range, self.domain, self.c)

    elif call_variant == 'dual':

        class ScalingOp(ScalingOpBase):

            def _call(self, x, out=None):
                if out is None:
                    out = self.c * x
                else:
                    out[:] = self.c * x
                return out

            @property
            @auto_adjoint_weighting(optimize=optimize_weighting)
            def adjoint(self):
                return ScalingOp(self.range, self.domain, self.c)

    else:
        assert False

    # Test Rn space
    rn = odl.rn(2)
    rn_w = odl.rn(2, weighting=weighting)
    op1 = ScalingOp(rn, rn_w, np.random.uniform(-2, 2))
    op2 = ScalingOp(rn_w, rn, np.random.uniform(-2, 2))

    for op in [op1, op2]:
        dom_el = noise_element(op.domain)
        ran_el = noise_element(op.range)
        assert pytest.approx(op(dom_el).inner(ran_el),
                             dom_el.inner(op.adjoint(ran_el)))

    # Test product space
    pspace = odl.ProductSpace(odl.rn(3), 2)
    pspace_w = odl.ProductSpace(odl.rn(3), 2, weighting=weighting)
    op1 = ScalingOp(pspace, pspace_w, np.random.uniform(-2, 2))
    op2 = ScalingOp(pspace_w, pspace, np.random.uniform(-2, 2))

    for op in [op1, op2]:
        dom_el = noise_element(op.domain)
        ran_el = noise_element(op.range)
        assert pytest.approx(op(dom_el).inner(ran_el),
                             dom_el.inner(op.adjoint(ran_el)))

    # Test product space of product space
    ppspace = odl.ProductSpace(odl.ProductSpace(odl.rn(3), 2), 2)
    ppspace_w = odl.ProductSpace(
        odl.ProductSpace(odl.rn(3), 2, weighting=weighting),
        2, weighting=weighting)
    op1 = ScalingOp(ppspace, ppspace_w, np.random.uniform(-2, 2))
    op2 = ScalingOp(ppspace_w, ppspace, np.random.uniform(-2, 2))

    for op in [op1, op2]:
        dom_el = noise_element(op.domain)
        ran_el = noise_element(op.range)
        assert pytest.approx(op(dom_el).inner(ran_el),
                             dom_el.inner(op.adjoint(ran_el)))


def test_auto_weighting_noarg():
    """Test the auto_weighting decorator without the optimize argument."""
    rn = odl.rn(2)
    weighting_const = np.random.uniform(0.5, 2)
    rn_w = odl.rn(2, weighting=weighting_const)

    class ScalingOp(odl.Operator):

        def __init__(self, dom, ran, c):
            super(ScalingOp, self).__init__(dom, ran, linear=True)
            self.c = c

        def _call(self, x):
            return self.c * x

        @property
        @auto_adjoint_weighting
        def adjoint(self):
            return ScalingOp(self.range, self.domain, self.c)

    op1 = ScalingOp(rn, rn, np.random.uniform(-2, 2))
    op2 = ScalingOp(rn_w, rn_w, np.random.uniform(-2, 2))
    op3 = ScalingOp(rn, rn_w, np.random.uniform(-2, 2))
    op4 = ScalingOp(rn_w, rn, np.random.uniform(-2, 2))

    for op in [op1, op2, op3, op4]:
        dom_el = noise_element(op.domain)
        ran_el = noise_element(op.range)
        assert pytest.approx(op(dom_el).inner(ran_el),
                             dom_el.inner(op.adjoint(ran_el)))


def test_auto_weighting_cached_adjoint():
    """Check if auto_weighting plays well with adjoint caching."""
    rn = odl.rn(2)
    weighting_const = np.random.uniform(0.5, 2)
    rn_w = odl.rn(2, weighting=weighting_const)

    class ScalingOp(odl.Operator):

        def __init__(self, dom, ran, c):
            super(ScalingOp, self).__init__(dom, ran, linear=True)
            self.c = c
            self._adjoint = None

        def _call(self, x):
            return self.c * x

        @property
        @auto_adjoint_weighting
        def adjoint(self):
            if self._adjoint is None:
                self._adjoint = ScalingOp(self.range, self.domain, self.c)
            return self._adjoint

    op = ScalingOp(rn, rn_w, np.random.uniform(-2, 2))
    dom_el = noise_element(op.domain)
    op_eval_before = op(dom_el)

    adj = op.adjoint
    adj_again = op.adjoint
    assert adj_again is adj

    # Check that original op is intact
    assert not hasattr(op, '_call_unweighted')  # op shouldn't be mutated
    op_eval_after = op(dom_el)
    assert all_equal(op_eval_before, op_eval_after)

    dom_el = noise_element(op.domain)
    ran_el = noise_element(op.range)
    op(dom_el)
    op.adjoint(ran_el)
    assert pytest.approx(op(dom_el).inner(ran_el),
                         dom_el.inner(op.adjoint(ran_el)))


def test_auto_weighting_raise_on_return_self():
    """Check that auto_weighting raises when adjoint returns self."""
    rn = odl.rn(2)

    class InvalidScalingOp(odl.Operator):

        def __init__(self, dom, ran, c):
            super(InvalidScalingOp, self).__init__(dom, ran, linear=True)
            self.c = c

        def _call(self, x):
            return self.c * x

        @property
        @auto_adjoint_weighting
        def adjoint(self):
            return self

    # This would be a vaild situation for adjoint just returning self
    op = InvalidScalingOp(rn, rn, 1.5)
    with pytest.raises(TypeError):
        op.adjoint


if __name__ == '__main__':
    odl.util.test_file(__file__)
