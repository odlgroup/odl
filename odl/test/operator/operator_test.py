# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import pytest
import numpy as np
import sys

import odl
from odl import (Operator, OperatorSum, OperatorComp,
                 OperatorLeftScalarMult, OperatorRightScalarMult,
                 FunctionalLeftVectorMult, OperatorRightVectorMult,
                 MatrixOperator, OperatorLeftVectorMult,
                 OpTypeError, OpDomainError, OpRangeError)
from odl.operator.operator import _function_signature, _dispatch_call_args
from odl.util.testutils import (
    all_almost_equal, noise_element, noise_elements, simple_fixture)
from odl.util.utility import getargspec


# --- Fixtures --- #


# This fixture is intended to activate testing of operator evaluation
# with aliased input and output, which is only possible if domain == range.
dom_eq_ran = simple_fixture('dom_eq_ran', [True, False])


# --- Auxilliary --- #


class MultiplyAndSquareOp(Operator):
    """Example of a nonlinear operator, x --> (mat*x)**2."""

    def __init__(self, matrix, domain=None, range=None):
        dom = (odl.rn(matrix.shape[1])
               if domain is None else domain)
        ran = (odl.rn(matrix.shape[0])
               if range is None else range)

        super(MultiplyAndSquareOp, self).__init__(dom, ran)
        self.matrix = matrix

    def _call(self, x, out=None):
        if out is None:
            out = self.range.element()
        out[:] = np.dot(self.matrix, x.data)
        out **= 2

    def derivative(self, x):
        return 2 * odl.MatrixOperator(self.matrix)

    def __str__(self):
        return "MaS: " + str(self.matrix) + " ** 2"


def mult_sq_np(mat, x):
    """NumPy reference implementation of MultiplyAndSquareOp."""
    return np.dot(mat, x) ** 2


def check_call(operator, x, expected):
    """Assert that operator(point) == expected."""
    # Out-of-place check
    assert all_almost_equal(operator(x), expected)

    # In-place check, no aliasing
    out = operator.range.element()
    operator(x, out=out)
    assert all_almost_equal(out, expected)

    # In-place check, aliased
    if operator.domain == operator.range:
        y = x.copy()
        operator(y, out=y)
        assert all_almost_equal(y, expected)


# --- Unit tests --- #


def test_operator_call(dom_eq_ran):
    """Check operator evaluation against NumPy reference."""
    if dom_eq_ran:
        mat = np.random.rand(3, 3)
        op = MultiplyAndSquareOp(mat)
        assert op.domain == op.range
    else:
        mat = np.random.rand(4, 3)
        op = MultiplyAndSquareOp(mat)

    xarr, x = noise_elements(op.domain)
    assert all_almost_equal(op(x), mult_sq_np(mat, xarr))


def test_operator_call_in_place_wrong_return():
    """Test that operator with out parameter actually returns out."""
    class BadInplaceOperator(odl.Operator):
        def _call(self, x, out):
            # badly implemented operator
            out = 42
            return out

    space = odl.rn(3)
    op = BadInplaceOperator(space, space)

    with pytest.raises(ValueError):
        op(space.zero())

    with pytest.raises(ValueError):
        out = space.zero()
        op(space.zero(), out=out)


def test_operator_sum(dom_eq_ran):
    """Check operator sum against NumPy reference."""
    if dom_eq_ran:
        mat1 = np.random.rand(3, 3)
        mat2 = np.random.rand(3, 3)
    else:
        mat1 = np.random.rand(4, 3)
        mat2 = np.random.rand(4, 3)

    op1 = MultiplyAndSquareOp(mat1)
    op2 = MultiplyAndSquareOp(mat2)
    xarr, x = noise_elements(op1.domain)

    # Explicit instantiation
    sum_op = OperatorSum(op1, op2)
    assert not sum_op.is_linear
    check_call(sum_op, x, mult_sq_np(mat1, xarr) + mult_sq_np(mat2, xarr))

    # Using operator overloading
    check_call(op1 + op2, x, mult_sq_np(mat1, xarr) + mult_sq_np(mat2, xarr))

    # Verify that unmatched operator domains fail
    op_wrong_dom = MultiplyAndSquareOp(mat1[:, :-1])
    with pytest.raises(OpTypeError):
        OperatorSum(op1, op_wrong_dom)

    # Verify that unmatched operator ranges fail
    op_wrong_ran = MultiplyAndSquareOp(mat1[:-1, :])
    with pytest.raises(OpTypeError):
        OperatorSum(op1, op_wrong_ran)


def test_operator_scaling(dom_eq_ran):
    """Check operator scaling against NumPy reference."""
    if dom_eq_ran:
        mat = np.random.rand(3, 3)
    else:
        mat = np.random.rand(4, 3)

    op = MultiplyAndSquareOp(mat)
    xarr, x = noise_elements(op.domain)

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1)).
    scalars = [-1.432, -1, 0, 1, 3.14]
    for scalar in scalars:
        lscaled = OperatorLeftScalarMult(op, scalar)
        rscaled = OperatorRightScalarMult(op, scalar)

        assert not lscaled.is_linear
        assert not rscaled.is_linear

        # Explicit
        check_call(lscaled, x, scalar * mult_sq_np(mat, xarr))
        check_call(rscaled, x, mult_sq_np(mat, scalar * xarr))

        # Using operator overloading
        check_call(scalar * op, x, scalar * mult_sq_np(mat, xarr))
        check_call(op * scalar, x, mult_sq_np(mat, scalar * xarr))

    # Fail when scaling by wrong scalar type (complex number)
    wrongscalars = [1j, [1, 2], (1, 2)]
    for wrongscalar in wrongscalars:
        with pytest.raises(TypeError):
            OperatorLeftScalarMult(op, wrongscalar)

        with pytest.raises(TypeError):
            OperatorRightScalarMult(op, wrongscalar)

        with pytest.raises(TypeError):
            op * wrongscalar

        with pytest.raises(TypeError):
            wrongscalar * op


def test_operator_vector_mult(dom_eq_ran):
    """Check operator-vector multiplication against NumPy reference."""
    if dom_eq_ran:
        mat = np.random.rand(3, 3)
    else:
        mat = np.random.rand(4, 3)

    op = MultiplyAndSquareOp(mat)
    right = op.domain.element(np.arange(op.domain.size))
    left = op.range.element(np.arange(op.range.size))
    xarr, x = noise_elements(op.domain)

    rmult_op = OperatorRightVectorMult(op, right)
    lmult_op = OperatorLeftVectorMult(op, left)
    assert not rmult_op.is_linear
    assert not lmult_op.is_linear

    check_call(rmult_op, x, mult_sq_np(mat, right * xarr))
    check_call(lmult_op, x, left * mult_sq_np(mat, xarr))

    # Using operator overloading
    check_call(op * right, x, mult_sq_np(mat, right * xarr))
    check_call(left * op, x, left * mult_sq_np(mat, xarr))


def test_operator_composition(dom_eq_ran):
    """Check operator composition against NumPy reference."""
    if dom_eq_ran:
        mat1 = np.random.rand(3, 3)
        mat2 = np.random.rand(3, 3)
    else:
        mat1 = np.random.rand(5, 4)
        mat2 = np.random.rand(4, 3)

    op1 = MultiplyAndSquareOp(mat1)
    op2 = MultiplyAndSquareOp(mat2)
    xarr, x = noise_elements(op2.domain)

    comp_op = OperatorComp(op1, op2)
    assert not comp_op.is_linear
    check_call(comp_op, x, mult_sq_np(mat1, mult_sq_np(mat2, xarr)))

    # Verify that incorrect order fails
    if not dom_eq_ran:
        with pytest.raises(OpTypeError):
            OperatorComp(op2, op1)


def test_linear_operator_call(dom_eq_ran):
    """Check call of a linear operator against NumPy, and ``is_linear``."""
    if dom_eq_ran:
        mat = np.random.rand(3, 3)
    else:
        mat = np.random.rand(4, 3)

    op = MatrixOperator(mat)
    assert op.is_linear

    xarr, x = noise_elements(op.domain)
    check_call(op, x, np.dot(mat, xarr))


def test_linear_operator_adjoint(dom_eq_ran):
    """Check adjoint of a linear operator against NumPy."""
    if dom_eq_ran:
        mat = np.random.rand(3, 3)
    else:
        mat = np.random.rand(4, 3)

    op = MatrixOperator(mat)
    xarr, x = noise_elements(op.range)
    check_call(op.adjoint, x, np.dot(mat.T, xarr))


def test_linear_operator_addition(dom_eq_ran):
    """Check call and adjoint of a sum of linear operators."""
    if dom_eq_ran:
        mat1 = np.random.rand(3, 3)
        mat2 = np.random.rand(3, 3)
    else:
        mat1 = np.random.rand(4, 3)
        mat2 = np.random.rand(4, 3)

    op1 = MatrixOperator(mat1)
    op2 = MatrixOperator(mat2)
    xarr, x = noise_elements(op1.domain)
    yarr, y = noise_elements(op1.range)

    # Explicit instantiation
    sum_op = OperatorSum(op1, op2)
    assert sum_op.is_linear
    assert sum_op.adjoint.is_linear
    check_call(sum_op, x, np.dot(mat1, xarr) + np.dot(mat2, xarr))
    check_call(sum_op.adjoint, y, np.dot(mat1.T, yarr) + np.dot(mat2.T, yarr))

    # Using operator overloading
    check_call(op1 + op2, x, np.dot(mat1, xarr) + np.dot(mat2, xarr))
    check_call((op1 + op2).adjoint,
               y, np.dot(mat1.T, yarr) + np.dot(mat2.T, yarr))


def test_linear_operator_scaling(dom_eq_ran):
    """Check call and adjoint of a scaled linear operator."""
    if dom_eq_ran:
        mat = np.random.rand(3, 3)
    else:
        mat = np.random.rand(4, 3)

    op = MatrixOperator(mat)
    xarr, x = noise_elements(op.domain)
    yarr, y = noise_elements(op.range)

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    scalars = [-1.432, -1, 0, 1, 3.14]
    for scalar in scalars:
        # Explicit instantiation
        scaled_op = OperatorRightScalarMult(op, scalar)
        assert scaled_op.is_linear
        assert scaled_op.adjoint.is_linear
        check_call(scaled_op, x, scalar * np.dot(mat, xarr))
        check_call(scaled_op.adjoint, y, scalar * np.dot(mat.T, yarr))

        # Using operator overloading
        check_call(scalar * op, x, scalar * np.dot(mat, xarr))
        check_call(op * scalar, x, scalar * np.dot(mat, xarr))
        check_call((scalar * op).adjoint, y, scalar * np.dot(mat.T, yarr))
        check_call((op * scalar).adjoint, y, scalar * np.dot(mat.T, yarr))


def test_linear_right_vector_mult(dom_eq_ran):
    """Check call and adjoint of linear operator x vector."""
    if dom_eq_ran:
        mat = np.random.rand(3, 3)
    else:
        mat = np.random.rand(4, 3)

    op = MatrixOperator(mat)
    (xarr, mul_arr), (x, mul) = noise_elements(op.domain, n=2)
    yarr, y = noise_elements(op.range)

    # Explicit instantiation
    rmult_op = OperatorRightVectorMult(op, mul)
    assert rmult_op.is_linear
    assert rmult_op.adjoint.is_linear
    check_call(rmult_op, x, np.dot(mat, mul_arr * xarr))
    check_call(rmult_op.adjoint, y, mul_arr * np.dot(mat.T, yarr))

    # Using operator overloading
    check_call(op * mul, x, np.dot(mat, mul_arr * xarr))
    check_call((op * mul).adjoint, y, mul_arr * np.dot(mat.T, yarr))


def test_linear_left_vector_mult(dom_eq_ran):
    """Check call and adjoint of vector x linear operator."""
    if dom_eq_ran:
        mat = np.random.rand(3, 3)
    else:
        mat = np.random.rand(4, 3)

    op = MatrixOperator(mat)
    xarr, x = noise_elements(op.domain)
    (yarr, mul_arr), (y, mul) = noise_elements(op.range, n=2)

    # Explicit instantiation
    lmult_op = OperatorLeftVectorMult(op, mul)
    assert lmult_op.is_linear
    assert lmult_op.adjoint.is_linear
    check_call(lmult_op, x, mul_arr * np.dot(mat, xarr))
    check_call(lmult_op.adjoint, y, np.dot(mat.T, mul_arr * yarr))

    # Using operator overloading
    check_call(mul * op, x, mul_arr * np.dot(mat, xarr))
    check_call((mul * op).adjoint, y, np.dot(mat.T, mul_arr * yarr))


def test_linear_operator_composition(dom_eq_ran):
    """Check call and adjoint of linear operator composition."""
    if dom_eq_ran:
        mat1 = np.random.rand(3, 3)
        mat2 = np.random.rand(3, 3)
    else:
        mat1 = np.random.rand(4, 3)
        mat2 = np.random.rand(3, 4)

    op1 = MatrixOperator(mat1)
    op2 = MatrixOperator(mat2)
    xarr, x = noise_elements(op2.domain)
    yarr, y = noise_elements(op1.range)

    # Explicit instantiation
    comp_op = OperatorComp(op1, op2)
    assert comp_op.is_linear
    assert comp_op.adjoint.is_linear
    check_call(comp_op, x, np.dot(mat1, np.dot(mat2, xarr)))
    check_call(comp_op.adjoint, y, np.dot(mat2.T, np.dot(mat1.T, yarr)))

    # Using operator overloading
    check_call(op1 * op2, x, np.dot(mat1, np.dot(mat2, xarr)))
    check_call((op1 * op2).adjoint, y, np.dot(mat2.T, np.dot(mat1.T, yarr)))


def test_type_errors():
    r3 = odl.rn(3)
    r4 = odl.rn(4)

    op = MatrixOperator(np.random.rand(3, 3))
    r3_elem1 = r3.zero()
    r3_elem2 = r3.zero()
    r4_elem1 = r4.zero()
    r4_elem2 = r4.zero()

    # Verify that correct usage works
    op(r3_elem1, r3_elem2)
    op.adjoint(r3_elem1, r3_elem2)

    # Test that erroneous usage raises
    with pytest.raises(OpDomainError):
        op(r4_elem1)

    with pytest.raises(OpDomainError):
        op.adjoint(r4_elem1)

    with pytest.raises(OpRangeError):
        op(r3_elem1, r4_elem1)

    with pytest.raises(OpRangeError):
        op.adjoint(r3_elem1, r4_elem1)

    with pytest.raises(OpDomainError):
        op(r4_elem1, r3_elem1)

    with pytest.raises(OpDomainError):
        op.adjoint(r4_elem1, r3_elem1)

    with pytest.raises(OpDomainError):
        op(r4_elem1, r4_elem2)

    with pytest.raises(OpDomainError):
        op.adjoint(r4_elem1, r4_elem2)


def test_arithmetic(dom_eq_ran):
    """Test that all standard arithmetic works."""
    if dom_eq_ran:
        mat1 = np.random.rand(3, 3)
        mat2 = np.random.rand(3, 3)
        mat3 = np.random.rand(3, 3)
        mat4 = np.random.rand(3, 3)
    else:
        mat1 = np.random.rand(4, 3)
        mat2 = np.random.rand(4, 3)
        mat3 = np.random.rand(3, 3)
        mat4 = np.random.rand(4, 4)

    op = MultiplyAndSquareOp(mat1)
    op2 = MultiplyAndSquareOp(mat2)
    op3 = MultiplyAndSquareOp(mat3)
    op4 = MultiplyAndSquareOp(mat4)
    # Create elements needed for later
    x = noise_element(op.domain)
    y = noise_element(op.domain)
    z = noise_element(op.range)
    scalar = np.pi

    # Simple tests here, more in depth comes later
    check_call(+op, x, op(x))
    check_call(-op, x, -op(x))
    check_call(scalar * op, x, scalar * op(x))
    check_call(scalar * (scalar * op), x, scalar**2 * op(x))
    check_call(op * scalar, x, op(scalar * x))
    check_call((op * scalar) * scalar, x, op(scalar**2 * x))
    check_call(op + op2, x, op(x) + op2(x))
    check_call(op - op2, x, op(x) - op2(x))
    check_call(op * op3, x, op(op3(x)))
    check_call(op4 * op, x, op4(op(x)))
    check_call(z * op, x, z * op(x))
    check_call(z * (z * op), x, (z * z) * op(x))
    check_call(op * y, x, op(x * y))
    check_call((op * y) * y, x, op((y * y) * x))
    check_call(op + z, x, op(x) + z)
    check_call(op - z, x, op(x) - z)
    check_call(z + op, x, z + op(x))
    check_call(z - op, x, z - op(x))
    check_call(op + scalar, x, op(x) + scalar)
    check_call(op - scalar, x, op(x) - scalar)
    check_call(scalar + op, x, scalar + op(x))
    check_call(scalar - op, x, scalar - op(x))


def test_operator_pointwise_product():
    """Check call and adjoint of operator pointwise multiplication."""
    if dom_eq_ran:
        mat1 = np.random.rand(3, 3)
        mat2 = np.random.rand(3, 3)
    else:
        mat1 = np.random.rand(4, 3)
        mat2 = np.random.rand(4, 3)

    op1 = MultiplyAndSquareOp(mat1)
    op2 = MultiplyAndSquareOp(mat2)
    x = noise_element(op1.domain)

    prod_op = odl.OperatorPointwiseProduct(op1, op2)

    # Evaluate
    expected = op1(x) * op2(x)
    check_call(prod_op, x, expected)

    # Derivative
    y = noise_element(op1.domain)
    expected = (op1.derivative(x)(y) * op2(x) +
                op2.derivative(x)(y) * op1(x))
    prod_deriv_op = prod_op.derivative(x)
    assert prod_deriv_op.is_linear
    check_call(prod_deriv_op, y, expected)

    # Derivative Adjoint
    z = noise_element(op1.range)
    expected = (op1.derivative(x).adjoint(z * op2(x)) +
                op2.derivative(x).adjoint(z * op1(x)))
    prod_deriv_adj_op = prod_deriv_op.adjoint
    assert prod_deriv_adj_op.is_linear
    check_call(prod_deriv_adj_op, z, expected)


# FUNCTIONAL TEST
class SumFunctional(Operator):

    """Sum of elements."""

    def __init__(self, domain):
        super(SumFunctional, self).__init__(domain, domain.field, linear=True)

    def _call(self, x):
        return np.sum(x)

    @property
    def adjoint(self):
        return ConstantVector(self.domain)


class ConstantVector(Operator):

    """Vector times a scalar."""

    def __init__(self, domain):
        super(ConstantVector, self).__init__(domain.field, domain, linear=True)

    def _call(self, x):
        return self.range.element(np.ones(self.range.size) * x)

    @property
    def adjoint(self):
        return SumFunctional(self.range)


def test_functional():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    op = SumFunctional(r3)

    assert op(x) == 6


def test_functional_out():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    op = SumFunctional(r3)

    # No out parameter allowed with functionals
    out = 0
    with pytest.raises(TypeError):
        op(x, out=out)


def test_functional_adjoint():
    r3 = odl.rn(3)

    op = SumFunctional(r3)

    assert op.adjoint(3) == r3.element([3, 3, 3])

    x = r3.element([1, 2, 3])
    assert op.adjoint.adjoint(x) == op(x)


def test_functional_addition():
    r3 = odl.rn(3)

    op = SumFunctional(r3)
    op2 = SumFunctional(r3)
    x = r3.element([1, 2, 3])
    y = 1

    # Explicit instantiation
    C = OperatorSum(op, op2)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert C(x) == 2 * np.sum(x)

    # Test adjoint
    assert all_almost_equal(C.adjoint(y), y * 2 * np.ones(3))
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert (op + op2)(x) == 2 * np.sum(x)
    assert all_almost_equal((op + op2).adjoint(y), y * 2 * np.ones(3))


def test_functional_scale():
    r3 = odl.rn(3)

    op = SumFunctional(r3)
    x = r3.element([1, 2, 3])
    y = 1

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    scalars = [-1.432, -1, 0, 1, 3.14]
    for scalar in scalars:
        C = OperatorRightScalarMult(op, scalar)

        assert C.is_linear
        assert C.adjoint.is_linear

        assert C(x) == scalar * np.sum(x)
        assert all_almost_equal(C.adjoint(y), scalar * y * np.ones(3))
        assert all_almost_equal(C.adjoint.adjoint(x), C(x))

        # Using operator overloading
        assert (scalar * op)(x) == scalar * np.sum(x)
        assert (op * scalar)(x) == scalar * np.sum(x)
        assert all_almost_equal((scalar * op).adjoint(y),
                                scalar * y * np.ones(3))
        assert all_almost_equal((op * scalar).adjoint(y),
                                scalar * y * np.ones(3))


def test_functional_left_vector_mult():
    r3 = odl.rn(3)
    r4 = odl.rn(4)

    op = SumFunctional(r3)
    x = r3.element([1, 2, 3])
    y = r4.element([3, 2, 1, 5])

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    C = FunctionalLeftVectorMult(op, y)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(x), y * np.sum(x))
    assert all_almost_equal(C.adjoint(y), y.inner(y) * np.ones(3))
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert all_almost_equal((y * op)(x),
                            y * np.sum(x))
    assert all_almost_equal((y * op).adjoint(y),
                            y.inner(y) * np.ones(3))


def test_functional_right_vector_mult():
    r3 = odl.rn(3)

    op = SumFunctional(r3)
    vec = r3.element([1, 2, 3])
    x = r3.element([4, 5, 6])
    y = 2.0

    # Test a range of scalars (scalar multiplication could implement
    # optimizations for (-1, 0, 1).
    C = OperatorRightVectorMult(op, vec)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(x), np.sum(vec * x))
    assert all_almost_equal(C.adjoint(y), vec * y)
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert all_almost_equal((op * vec)(x),
                            np.sum(vec * x))
    assert all_almost_equal((op * vec).adjoint(y),
                            vec * y)


def test_functional_composition():
    r3 = odl.rn(3)

    op = SumFunctional(r3)
    op2 = ConstantVector(r3)
    x = r3.element([1, 2, 3])
    y = 1

    C = OperatorComp(op2, op)

    assert C.is_linear
    assert C.adjoint.is_linear

    assert all_almost_equal(C(x), np.sum(x) * np.ones(3))
    assert all_almost_equal(C.adjoint(x), np.sum(x) * np.ones(3))
    assert all_almost_equal(C.adjoint.adjoint(x), C(x))

    # Using operator overloading
    assert (op * op2)(y) == y * 3
    assert (op * op2).adjoint(y) == y * 3
    assert all_almost_equal((op2 * op)(x),
                            np.sum(x) * np.ones(3))
    assert all_almost_equal((op2 * op).adjoint(x),
                            np.sum(x) * np.ones(3))


class SumSquaredFunctional(Operator):

    """Sum of the squared elements."""

    def __init__(self, domain):
        super(SumSquaredFunctional, self).__init__(
            domain, domain.field, linear=False)

    def _call(self, x):
        return np.sum(x ** 2)


def test_nonlinear_functional():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    op = SumSquaredFunctional(r3)

    assert op(x) == pytest.approx(np.sum(x ** 2))


def test_nonlinear_functional_out():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    op = SumSquaredFunctional(r3)
    out = op.range.element()

    with pytest.raises(TypeError):
        op(x, out=out)


def test_nonlinear_functional_operators():
    r3 = odl.rn(3)
    x = r3.element([1, 2, 3])

    mat = SumSquaredFunctional(r3)
    mat2 = SumFunctional(r3)

    # Sum
    C = mat + mat2

    assert not C.is_linear
    assert C(x) == pytest.approx(mat(x) + mat2(x))

    # Minus
    C = mat - mat2

    assert not C.is_linear
    assert C(x) == pytest.approx(mat(x) - mat2(x))

    # left mul
    C = 2.0 * mat

    assert not C.is_linear
    assert C(x) == pytest.approx(2.0 * mat(x))

    # right mul
    C = mat * 2.0

    assert not C.is_linear
    assert C(x) == pytest.approx(mat(x * 2.0))

    # right divide
    C = mat / 2.0

    assert not C.is_linear
    assert C(x) == pytest.approx(mat(x / 2.0))


# test functions to dispatch
def f1(x):
    """f1(x)
    False, False
    good
    """


def f2(x, y):
    """f2(x, y)
    False, False
    bad
    """


def f3(x, y, out):
    """f3(x, y, out)
    False, True
    bad
    """


def f4(*args):
    """f4(*args)
    False, False
    bad
    """


def f5(out):
    """f5(out)
    True, False
    bad
    """


def f6(**kwargs):
    """f6(**kwargs)
    False, False
    bad
    """


def f7(*args, **kwargs):
    """f7(*args, **kwargs)
    False, False
    bad
    """


def f8(out=None, param=2):
    """f8(out=None, param=2)
    True, True
    bad
    """


def f9(x, out=None):
    """f9(x, out=None)
    True, True
    good
    """


def f10(x, out=None, param=2):
    """f10(x, out=None, param=2)
    True, True
    bad
    """


def f11(x, out, **kwargs):
    """f11(x, out, **kwargs)
    True, False
    good
    """


def f12(x, y, out=None, param=2, *args, **kwargs):
    """f12(x, y, out=None, param=2, *args, **kwargs)
    True, True
    bad
    """


py3 = sys.version_info.major > 2
if py3:
    exec('''
def f13(x, *, out=None, param=2, **kwargs):
    """f13(x, *, out=None, param=2, **kwargs)
    True, True
    good
    """
''')

    exec('''
def f14(x, *y, out=None, param=2, **kwargs):
    """f14(x, *y, out=None, param=2, **kwargs)
    True, True
    bad
    """
''')


func_params = [eval('f{}'.format(i)) for i in range(1, 13)]
if py3:
    func_params += [eval('f{}'.format(i)) for i in range(13, 15)]
func_ids = [' f{} '.format(i) for i in range(1, 13)]
if py3:
    func_ids += [' f{} '.format(i) for i in range(13, 15)]
func_fixture = pytest.fixture(scope="module", ids=func_ids, params=func_params)


@func_fixture
def func(request):
    return request.param


def test_function_signature(func):

    true_sig = func.__doc__.splitlines()[0].strip()
    sig = _function_signature(func)
    assert true_sig == sig


def test_dispatch_call_args(func):
    # Unbound functions
    true_has, true_opt = eval(func.__doc__.splitlines()[1].strip())
    good = func.__doc__.splitlines()[2].strip() == 'good'

    if good:
        truespec = getargspec(func)
        truespec.args.insert(0, 'self')

        has, opt, spec = _dispatch_call_args(unbound_call=func)

        assert has == true_has
        assert opt == true_opt
        assert spec == truespec
    else:
        with pytest.raises(ValueError):
            _dispatch_call_args(unbound_call=func)


def test_dispatch_call_args_class():

    # Two sneaky classes whose _call method would pass the signature check
    class WithStaticMethod(object):
        @staticmethod
        def _call(x, y, out):
            pass

    class WithClassMethod(object):
        @classmethod
        def _call(cls, x, out=None):
            pass

    with pytest.raises(TypeError):
        _dispatch_call_args(cls=WithStaticMethod)

    with pytest.raises(TypeError):
        _dispatch_call_args(cls=WithClassMethod)


if __name__ == '__main__':
    odl.util.test_file(__file__)
