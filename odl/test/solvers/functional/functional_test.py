# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the Functional class."""

from __future__ import division

import numpy as np
import pytest

import odl
from odl.operator import OpTypeError
from odl.solvers.functional.default_functionals import (
    KullbackLeiblerConvexConj)
from odl.util.testutils import (
    all_almost_equal, dtype_ndigits, dtype_tol, noise_element, simple_fixture)

# --- pytest fixtures --- #


scalar = simple_fixture('scalar', [0.01, 2.7, 10, -2, -0.2, -7.1, 0])
sigma = simple_fixture('sigma', [0.001, 2.7, np.array(0.5), 10])

space_params = ['r10', 'uniform_discr', 'power_space_unif_discr']
space_ids = [' space={} '.format(p) for p in space_params]

# Fixtures for test_functional_quadratic_perturb
linear_term = simple_fixture('linear_term', [False, True])
quadratic_coeff = simple_fixture('quadratic_coeff', [0.0, 2.13])


# --- Unittests --- #


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def space(request, odl_tspace_impl):
    name = request.param.strip()
    impl = odl_tspace_impl

    if name == 'r10':
        return odl.rn(10, impl=impl)
    elif name == 'uniform_discr':
        # Discretization parameters
        return odl.uniform_discr(0, 1, 7, impl=impl)
    elif name == 'power_space_unif_discr':
        # Discretization parameters
        space = odl.uniform_discr(0, 1, 7, impl=impl)
        return odl.ProductSpace(space, 2)


func_params = ['l1 ', 'l2', 'l2^2', 'constant', 'zero', 'ind_unit_ball_1',
               'ind_unit_ball_2', 'ind_unit_ball_pi', 'ind_unit_ball_inf',
               'product', 'quotient', 'kl', 'kl_cc', 'kl_cross_ent',
               'kl_cc_cross_ent', 'huber', 'groupl1', 'bregman_l2squared',
               'bregman_l1', 'indicator_simplex', 'indicator_sum_constraint']

func_ids = [" functional='{}' ".format(p) for p in func_params]

FUNCTIONALS_WITHOUT_DERIVATIVE = (
    odl.solvers.functional.IndicatorLpUnitBall,
    odl.solvers.functional.IndicatorSimplex,
    odl.solvers.functional.IndicatorSumConstraint)


@pytest.fixture(scope="module", ids=func_ids, params=func_params)
def functional(request, space):
    name = request.param.strip()

    if name == 'l1':
        func = odl.solvers.functional.L1Norm(space)
    elif name == 'l2':
        func = odl.solvers.functional.L2Norm(space)
    elif name == 'l2^2':
        func = odl.solvers.functional.L2NormSquared(space)
    elif name == 'constant':
        func = odl.solvers.functional.ConstantFunctional(space, 2)
    elif name == 'zero':
        func = odl.solvers.functional.ZeroFunctional(space)
    elif name == 'ind_unit_ball_1':
        func = odl.solvers.functional.IndicatorLpUnitBall(space, 1)
    elif name == 'ind_unit_ball_2':
        func = odl.solvers.functional.IndicatorLpUnitBall(space, 2)
    elif name == 'ind_unit_ball_pi':
        func = odl.solvers.functional.IndicatorLpUnitBall(space, np.pi)
    elif name == 'ind_unit_ball_inf':
        func = odl.solvers.functional.IndicatorLpUnitBall(space, np.inf)
    elif name == 'product':
        left = odl.solvers.functional.L2Norm(space)
        right = odl.solvers.functional.ConstantFunctional(space, 2)
        func = odl.solvers.functional.FunctionalProduct(left, right)
    elif name == 'quotient':
        dividend = odl.solvers.functional.L2Norm(space)
        divisor = odl.solvers.functional.ConstantFunctional(space, 2)
        func = odl.solvers.functional.FunctionalQuotient(dividend, divisor)
    elif name == 'kl':
        func = odl.solvers.functional.KullbackLeibler(space)
    elif name == 'kl_cc':
        func = odl.solvers.KullbackLeibler(space).convex_conj
    elif name == 'kl_cross_ent':
        func = odl.solvers.functional.KullbackLeiblerCrossEntropy(space)
    elif name == 'kl_cc_cross_ent':
        func = odl.solvers.KullbackLeiblerCrossEntropy(space).convex_conj
    elif name == 'huber':
        func = odl.solvers.Huber(space, gamma=0.1)
    elif name == 'groupl1':
        if isinstance(space, odl.ProductSpace):
            pytest.skip("The `GroupL1Norm` is not supported on `ProductSpace`")
        space = odl.ProductSpace(space, 3)
        func = odl.solvers.GroupL1Norm(space)
    elif name == 'bregman_l2squared':
        point = noise_element(space)
        l2_squared = odl.solvers.L2NormSquared(space)
        subgrad = l2_squared.gradient(point)
        func = odl.solvers.BregmanDistance(l2_squared, point, subgrad)
    elif name == 'bregman_l1':
        point = noise_element(space)
        l1 = odl.solvers.L1Norm(space)
        subgrad = l1.gradient(point)
        func = odl.solvers.BregmanDistance(l1, point, subgrad)
    elif name == 'indicator_simplex':
        diameter = 1.23
        func = odl.solvers.IndicatorSimplex(space, diameter)
    elif name == 'indicator_sum_constraint':
        sum_value = 1.23
        func = odl.solvers.IndicatorSumConstraint(space, sum_value)
    else:
        assert False

    return func


# --- Functional tests --- #


def test_derivative(functional):
    """Test for the derivative of a functional.

    The test checks that the directional derivative in a point is the same as
    the inner product of the gradient and the direction, if the gradient is
    defined.
    """
    if isinstance(functional, FUNCTIONALS_WITHOUT_DERIVATIVE):
        # IndicatorFunction has no derivative
        with pytest.raises(NotImplementedError):
            functional.derivative(functional.domain.zero())
        return

    space = functional.domain
    F = space.ufuncs
    R = space.reduce
    x = noise_element(space)
    y = noise_element(space)

    if (
        isinstance(functional, odl.solvers.KullbackLeibler)
        or isinstance(functional, odl.solvers.KullbackLeiblerCrossEntropy)
    ):
        # This functional is not defined for values <= 0
        x = F.abs(x)
        y = F.abs(y)

    if isinstance(functional, KullbackLeiblerConvexConj):
        # This functional is not defined for values >= 1
        x = x - R.max(x) + 0.99
        y = y - R.max(y) + 0.99

    # Compute a "small" step size according to dtype of space
    step = float(np.sqrt(np.finfo(space.dtype).eps))

    # Numerical test of gradient, only low accuracy can be guaranteed
    assert all_almost_equal(
        (functional(x + step * y) - functional(x)) / step,
        space.inner(y, functional.gradient(x)),
        ndigits=1,
    )

    # Check that derivative and gradient is consistent
    assert all_almost_equal(
        functional.derivative(x)(y), space.inner(y, functional.gradient(x))
    )


def test_arithmetic():
    """Test that standard arithmetic works as expected."""
    space = odl.rn(3)
    functional = odl.solvers.L2Norm(space).translated([1, 2, 3])
    functional2 = odl.solvers.L2NormSquared(space)
    operator = odl.IdentityOperator(space) - space.element([4, 5, 6])
    x = noise_element(space)
    y = noise_element(space)
    scalar = np.pi

    # Simple tests here, more in depth comes later
    assert functional(x) != functional2(x)
    assert (scalar * functional)(x) == scalar * functional(x)
    assert (scalar * (scalar * functional))(x) == scalar ** 2 * functional(x)
    assert (functional * scalar)(x) == functional(scalar * x)
    assert ((functional * scalar) * scalar)(x) == functional(scalar ** 2 * x)
    assert (functional + functional2)(x) == functional(x) + functional2(x)
    assert (functional - functional2)(x) == functional(x) - functional2(x)
    assert (functional * operator)(x) == functional(operator(x))
    assert all_almost_equal((functional * y)(x), functional(y * x))
    assert all_almost_equal(((functional * y) * y)(x), functional((y * y) * x))


def test_left_scalar_mult(space, scalar):
    """Test for right and left multiplication of a functional with a scalar."""
    ndigits = dtype_ndigits(space.dtype)
    rtol = dtype_tol(space.dtype)

    x = noise_element(space)
    func = odl.solvers.functional.L2Norm(space)
    lmul_func = scalar * func

    if scalar == 0:
        assert (scalar * func)(x) == 0
        # Return early in this case as many things are undefined
        return

    assert lmul_func(x) == pytest.approx(scalar * func(x), rel=rtol)
    assert all_almost_equal(
        lmul_func.gradient(x), scalar * func.gradient(x), ndigits
    )
    p = noise_element(space)
    assert all_almost_equal(
        lmul_func.derivative(x)(p),
        scalar * (func.derivative(x))(p),
        ndigits,
    )

    pos_scalar = abs(scalar) + 1e-4
    neg_scalar = -pos_scalar

    with pytest.raises(ValueError):
        # Not a convex functional, should raise
        (neg_scalar * func).convex_conj

    assert all_almost_equal(
        (pos_scalar * func).convex_conj(x),
        pos_scalar * func.convex_conj(x / pos_scalar),
        ndigits,
    )

    sigma = 1.2
    with pytest.raises(ValueError):
        # Not a convex functional, should raise
        (neg_scalar * func).proximal(sigma)

    assert all_almost_equal(
        (pos_scalar * func).proximal(sigma)(x),
        func.proximal(sigma * pos_scalar)(x),
    )


def test_right_scalar_mult(space, scalar):
    """Test for right and left multiplication of a functional with a scalar."""
    ndigits = dtype_ndigits(space.dtype)
    rtol = dtype_tol(space.dtype)

    x = noise_element(space)
    func = odl.solvers.functional.L2NormSquared(space)
    rmul_func = func * scalar

    if scalar == 0:
        # Should yield `func(0)` for any input
        assert all_almost_equal(
            rmul_func(x), func(space.zero()), ndigits
        )
        # Nothing more to do, rest is part of ConstantFunctional test
        return

    assert rmul_func(x) == pytest.approx(func(scalar * x), rel=rtol)

    # Chain rule for gradient: grad[f(c * .)] = c * grad[f](c * .)
    assert all_almost_equal(
        rmul_func.gradient(x), scalar * func.gradient(scalar * x), ndigits,
    )
    # Same for derivative
    p = noise_element(space)
    assert all_almost_equal(
        rmul_func.derivative(x)(p),
        scalar * func.derivative(scalar * x)(p),
        ndigits,
    )

    # Scaling and convex conjugate: [f(c * .)]^* = f^*(1/c * .)
    assert all_almost_equal(
        rmul_func.convex_conj(x), func.convex_conj(x / scalar), ndigits,
    )

    # Scaling and proximal: prox[s * f(c * .)] = 1/c * prox[s*c^2 * f](c * .)
    sigma = 1.2
    assert all_almost_equal(
        rmul_func.proximal(sigma)(x),
        (1.0 / scalar) * func.proximal(sigma * scalar ** 2)(x * scalar),
        ndigits,
    )

    # Verify that for linear functionals, left multiplication is used
    func = odl.solvers.ZeroFunctional(space)
    assert isinstance(func * scalar, odl.solvers.FunctionalLeftScalarMult)


def test_functional_composition(space):
    """Test composition from the right with an operator."""
    ndigits = dtype_ndigits(space.dtype)
    rtol = dtype_tol(space.dtype)
    func = odl.solvers.L2NormSquared(space)
    x = noise_element(space)

    op = odl.operator.ScalingOperator(space, 2.0)
    func_op_comp = func * op
    assert isinstance(func_op_comp, odl.solvers.Functional)

    assert func_op_comp(x) == pytest.approx(func(op(x)), rel=rtol)

    # Chain rule for composition: grad[f o A] = A^* o grad[f] o A
    assert all_almost_equal(
        func_op_comp.gradient(x),
        (op.adjoint * func.gradient * op)(x),
        ndigits,
    )
    # Same for derivative
    p = noise_element(space)
    assert all_almost_equal(
        func_op_comp.derivative(x)(p),
        space.inner((op.adjoint * func.gradient * op)(x), p),
        ndigits,
    )

    wrong_space = odl.uniform_discr(1, 2, 10)
    op_wrong = odl.operator.ScalingOperator(wrong_space, 2.1)

    with pytest.raises(OpTypeError):
        func * op_wrong


def test_functional_sum(space):
    """Test for the sum of two functionals."""
    ndigits = dtype_ndigits(space.dtype)
    rtol = dtype_tol(space.dtype)
    func1 = odl.solvers.L2NormSquared(space)
    func2 = odl.solvers.L2Norm(space)

    func_sum = func1 + func2
    x = noise_element(space)
    p = noise_element(space)

    assert func_sum(x) == pytest.approx(func1(x) + func2(x), rel=rtol)

    # grad[f + g] = grad[f] + grad[g]
    assert all_almost_equal(
        func_sum.gradient(x), func1.gradient(x) + func2.gradient(x), ndigits
    )
    assert (
        func_sum.derivative(x)(p)
        == pytest.approx(
            space.inner(func1.gradient(x), p)
            + space.inner(func2.gradient(x), p),
            rel=rtol,
        )
    )

    op = odl.operator.IdentityOperator(space)
    with pytest.raises(OpTypeError):
        func1 + op

    wrong_space = odl.uniform_discr(1, 2, 10)
    func_wrong_domain = odl.solvers.L2Norm(wrong_space)
    with pytest.raises(OpTypeError):
        func1 + func_wrong_domain



def test_functional_plus_scalar(space):
    """Test for sum of functioanl and scalar."""
    ndigits = dtype_ndigits(space.dtype)
    rtol = dtype_tol(space.dtype)
    func = odl.solvers.L2NormSquared(space)
    scalar = -1.3

    func_scalar_sum = func + scalar
    x = noise_element(space)
    p = noise_element(space)

    assert func_scalar_sum(x) == pytest.approx(func(x) + scalar, rel=rtol)

    # grad[f + c] = grad[f]
    assert all_almost_equal(
        func_scalar_sum.gradient(x), func.gradient(x), ndigits
    )
    assert (
        func_scalar_sum.derivative(x)(p)
        == pytest.approx(space.inner(func.gradient(x), p), rel=rtol)
    )

    # Proximal is unaffected by constant shift
    sigma = 1.2
    assert all_almost_equal(
        func_scalar_sum.proximal(sigma)(x), func.proximal(sigma)(x), ndigits
    )

    # [f + c]^* = f^* - c
    assert (
        func_scalar_sum.convex_conj(x)
        == pytest.approx(func.convex_conj(x) - scalar, rel=rtol)
    )
    assert all_almost_equal(
        func_scalar_sum.convex_conj.gradient(x),
        func.convex_conj.gradient(x),
        ndigits,
    )

    complex_scalar = 1j  # not in space.field
    with pytest.raises(TypeError):
        func + complex_scalar


def test_translation_of_functional(space):
    """Test for the translation of a functional."""
    ndigits = dtype_ndigits(space.dtype)
    transl = noise_element(space)
    func = odl.solvers.L2NormSquared(space)
    func_tr = func.translated(transl)
    x = noise_element(space)

    assert all_almost_equal(func_tr(x), func(x - transl), ndigits)
    assert all_almost_equal(
        func_tr.gradient(x), func.gradient(x - transl), ndigits
    )

    # prox[s * f(. - t)] = t + prox[s * f](. - y)
    sigma = 1.2
    assert all_almost_equal(
        func_tr.proximal(sigma)(x),
        transl + func.proximal(sigma)(x - transl),
        ndigits,
    )

    # [f(. - t)]^* = f^* + <t, .>
    assert all_almost_equal(
        func_tr.convex_conj(x),
        func.convex_conj(x) + space.inner(x, transl),
        ndigits,
    )

    # Test for optimized implementation when translating a translated
    # functional
    transl2 = noise_element(space)
    func_tr_twice = func_tr.translated(transl2)
    assert all_almost_equal(func_tr_twice.translation, transl + transl2)


def test_translation_proximal_stepsizes():
    """Test for stepsize types for proximal of a translated functional."""
    space = odl.rn(2)
    func = odl.solvers.L2NormSquared(space)
    func_tr = func.translated([0.5, 0.5])
    x = space.one()

    y = func_tr.convex_conj.proximal(space.element([0.5, 2.0]))(x)
    y_list = func_tr.convex_conj.proximal([0.5, 2.0])(x)
    expected_result = [0.6, 0.0]
    assert all_almost_equal(y, expected_result)
    assert all_almost_equal(y_list, expected_result)


def test_multiplication_with_vector(space):
    """Test for multiplying a functional with a vector, both left and right."""
    ndigits = dtype_ndigits(space.dtype)
    rtol = dtype_tol(space.dtype)
    x = noise_element(space)
    y = noise_element(space)
    func = odl.solvers.L2NormSquared(space)

    func_times_y = func * y
    assert isinstance(func_times_y, odl.solvers.Functional)
    assert func_times_y(x) == pytest.approx(func(y * x), rel=rtol)

    # Gradient should be 2 * y^2 * x
    assert all_almost_equal(func_times_y.gradient(x), 2 * y * y * x, ndigits)

    # Convex conjugate should be 1/4 * ||x/y||_2^2
    assert func_times_y.convex_conj(x) == pytest.approx(
        1 / 4 * func(x / y), rel=rtol
    )


def test_functional_quadratic_perturb(space, linear_term, quadratic_coeff):
    """Test for the functional ``f(.) + a | . |^2 + <y, .>``."""
    ndigits = dtype_ndigits(space.dtype)
    rtol = dtype_tol(space.dtype)
    func = odl.solvers.L2NormSquared(space)
    x = noise_element(space)

    if linear_term:
        linear_term_arg = linear_term = noise_element(space)
    else:
        linear_term_arg = None
        linear_term = space.zero()

    func_quad_perturb = odl.solvers.FunctionalQuadraticPerturb(
        func, quadratic_coeff, linear_term_arg,
    )

    assert (
        func_quad_perturb(x)
        == pytest.approx(
            func(x)
            + quadratic_coeff * space.inner(x, x)
            + space.inner(x, linear_term),
            rel=rtol,
        )
    )

    # grad[f + a * <., .> + <., u> + c] = grad[f] + 2*a * . + u
    assert all_almost_equal(
        func_quad_perturb.gradient(x),
        func.gradient(x) + 2.0 * quadratic_coeff * x + linear_term,
        ndigits,
    )

    sigma = 1.2
    # prox[s * (f + a * <., .> + <., u> + c)] =
    # = prox[s * alpha * f]((. - s * u) * alpha), alpha = 1 / (2 * s * a + 1)
    # Explicit computation gives
    alpha = 1 / (2 * sigma * quadratic_coeff + 1)
    assert all_almost_equal(
        func_quad_perturb.proximal(sigma)(x),
        func.proximal(sigma * alpha)((x - sigma * linear_term) * alpha),
        ndigits,
    )

    # Convex conjugate only known for zero quadratic term
    if quadratic_coeff == 0:
        # [f + <., u>]^* = f^*(. - u)
        assert func_quad_perturb.convex_conj(x) == pytest.approx(
            func.convex_conj(x - linear_term), rel=rtol
        )


def test_bregman(functional):
    """Test for the Bregman distance of a functional."""
    space = functional.domain
    F = space.ufuncs
    R = space.reduce
    rtol = dtype_tol(space.dtype)

    if isinstance(functional, FUNCTIONALS_WITHOUT_DERIVATIVE):
        # IndicatorFunction has no gradient
        with pytest.raises(NotImplementedError):
            functional.gradient(space.zero())
        return

    y = noise_element(space)
    x = noise_element(space)

    if (
        isinstance(functional, odl.solvers.KullbackLeibler)
        or isinstance(functional, odl.solvers.KullbackLeiblerCrossEntropy)
    ):
        # The functional is not defined for values <= 0
        x = F.abs(x)
        y = F.abs(y)

    if isinstance(functional, KullbackLeiblerConvexConj):
        # The functional is not defined for values >= 1
        x = x - R.max(x) + 0.99
        y = y - R.max(y) + 0.99

    grad = functional.gradient(y)
    quadratic_func = odl.solvers.QuadraticForm(
        space, vector=-grad, constant=-functional(y) + space.inner(grad, y)
    )
    expected_func = functional + quadratic_func

    assert (
        functional.bregman(y, grad)(x)
        == pytest.approx(expected_func(x), rel=rtol)
    )


if __name__ == '__main__':
    odl.util.test_file(__file__)
