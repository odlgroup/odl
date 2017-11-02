# Copyright 2014-2017 The ODL contributors
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
from odl.util.testutils import (all_almost_equal, almost_equal, noise_element,
                                simple_fixture)
from odl.solvers.functional.default_functionals import (
    KullbackLeiblerConvexConj)


# TODO: maybe add tests for if translations etc. belongs to the wrong space.
# These tests don't work as intended now, since casting is possible between
# spaces with the same number of discretization points.


# --- pytest fixtures --- #


scalar = simple_fixture('scalar', [0.01, 2.7, 10, -2, -0.2, -7.1, 0])
sigma = simple_fixture('sigma', [0.001, 2.7, np.array(0.5), 10])

space_params = ['r10', 'uniform_discr']
space_ids = [' space={} '.format(p) for p in space_params]


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def space(request, fn_impl):
    name = request.param.strip()

    if name == 'r10':
        return odl.rn(10, impl=fn_impl)
    elif name == 'uniform_discr':
        # Discretization parameters
        return odl.uniform_discr(0, 1, 7, impl=fn_impl)


func_params = ['l1 ', 'l2', 'l2^2', 'constant', 'zero', 'ind_unit_ball_1',
               'ind_unit_ball_2', 'ind_unit_ball_pi', 'ind_unit_ball_inf',
               'product', 'quotient', 'kl', 'kl_cc', 'kl_cross_ent',
               'kl_cc_cross_ent', 'huber']
func_ids = [" functional='{}' ".format(p) for p in func_params]


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
    else:
        assert False

    return func


# --- functional tests --- #


def test_derivative(functional):
    """Test for the derivative of a functional.

    The test checks that the directional derivative in a point is the same as
    the inner product of the gradient and the direction, if the gradient is
    defined.
    """
    if isinstance(functional, odl.solvers.functional.IndicatorLpUnitBall):
        # IndicatorFunction has no derivative
        with pytest.raises(NotImplementedError):
            functional.derivative(functional.domain.zero())
        return

    x = noise_element(functional.domain)
    y = noise_element(functional.domain)

    if (isinstance(functional, odl.solvers.KullbackLeibler) or
            isinstance(functional, odl.solvers.KullbackLeiblerCrossEntropy)):
        # The functional is not defined for values <= 0
        x = x.ufuncs.absolute()
        y = y.ufuncs.absolute()

    if isinstance(functional, KullbackLeiblerConvexConj):
        # The functional is not defined for values >= 1
        x = x - x.ufuncs.max() + 0.99
        y = y - y.ufuncs.max() + 0.99

    # Compute a "small" step size according to dtype of space
    step = float(np.sqrt(np.finfo(functional.domain.dtype).eps))

    # Numerical test of gradient, only low accuracy can be guaranteed.
    assert all_almost_equal((functional(x + step * y) - functional(x)) / step,
                            y.inner(functional.gradient(x)),
                            places=1)

    # Check that derivative and gradient is consistent
    assert all_almost_equal(functional.derivative(x)(y),
                            y.inner(functional.gradient(x)))


def test_arithmetic():
    """Test that all standard arithmetic works."""
    space = odl.rn(3)

    # Create elements needed for later
    functional = odl.solvers.L2Norm(space).translated([1, 2, 3])
    functional2 = odl.solvers.L2NormSquared(space)
    operator = odl.IdentityOperator(space) - space.element([4, 5, 6])
    x = noise_element(functional.domain)
    y = noise_element(functional.domain)
    scalar = np.pi

    # Simple tests here, more in depth comes later
    assert functional(x) == functional(x)
    assert functional(x) != functional2(x)
    assert (scalar * functional)(x) == scalar * functional(x)
    assert (scalar * (scalar * functional))(x) == scalar**2 * functional(x)
    assert (functional * scalar)(x) == functional(scalar * x)
    assert ((functional * scalar) * scalar)(x) == functional(scalar**2 * x)
    assert (functional + functional2)(x) == functional(x) + functional2(x)
    assert (functional - functional2)(x) == functional(x) - functional2(x)
    assert (functional * operator)(x) == functional(operator(x))
    assert all_almost_equal((y * functional)(x), y * functional(x))
    assert all_almost_equal((y * (y * functional))(x), (y * y) * functional(x))
    assert all_almost_equal((functional * y)(x), functional(y * x))
    assert all_almost_equal(((functional * y) * y)(x), functional((y * y) * x))


def test_left_scalar_mult(space, scalar):
    """Test for right and left multiplication of a functional with a scalar."""
    # Less strict checking for single precision
    places = 3 if space.dtype == np.float32 else 5

    x = noise_element(space)
    func = odl.solvers.functional.L2Norm(space)
    lmul_func = scalar * func

    if scalar == 0:
        assert isinstance(scalar * func, odl.solvers.ZeroFunctional)
        return

    # Test functional evaluation
    assert almost_equal(lmul_func(x), scalar * func(x), places=places)

    # Test gradient of left scalar multiplication
    assert all_almost_equal(lmul_func.gradient(x), scalar * func.gradient(x),
                            places=places)

    # Test derivative of left scalar multiplication
    p = noise_element(space)
    assert all_almost_equal(lmul_func.derivative(x)(p),
                            scalar * (func.derivative(x))(p),
                            places=places)

    # Test convex conjugate. This requires positive scaling to work
    pos_scalar = abs(scalar)
    neg_scalar = -pos_scalar

    with pytest.raises(ValueError):
        (neg_scalar * func).convex_conj

    assert all_almost_equal((pos_scalar * func).convex_conj(x),
                            pos_scalar * func.convex_conj(x / pos_scalar),
                            places=places)

    # Test proximal operator. This requires scaling to be positive.
    sigma = 1.2
    with pytest.raises(ValueError):
        (neg_scalar * func).proximal(sigma)

    assert all_almost_equal((pos_scalar * func).proximal(sigma)(x),
                            func.proximal(sigma * pos_scalar)(x))


def test_right_scalar_mult(space, scalar):
    """Test for right and left multiplication of a functional with a scalar."""
    # Less strict checking for single precision
    places = 3 if space.dtype == np.float32 else 5

    x = noise_element(space)
    func = odl.solvers.functional.L2NormSquared(space)
    rmul_func = func * scalar

    if scalar == 0:
        # expecting the constant functional x -> func(0)
        assert isinstance(rmul_func, odl.solvers.ConstantFunctional)
        assert all_almost_equal(rmul_func(x), func(space.zero()),
                                places=places)
        # Nothing more to do, rest is part of ConstantFunctional test
        return

    # Test functional evaluation
    assert almost_equal(rmul_func(x), func(scalar * x), places=places)

    # Test gradient of right scalar multiplication
    assert all_almost_equal(rmul_func.gradient(x),
                            scalar * func.gradient(scalar * x),
                            places=places)

    # Test derivative of right scalar multiplication
    p = noise_element(space)
    assert all_almost_equal(rmul_func.derivative(x)(p),
                            scalar * func.derivative(scalar * x)(p),
                            places=places)

    # Test convex conjugate conjugate
    assert all_almost_equal(rmul_func.convex_conj(x),
                            func.convex_conj(x / scalar),
                            places=places)

    # Test proximal operator
    sigma = 1.2
    assert all_almost_equal(
        rmul_func.proximal(sigma)(x),
        (1.0 / scalar) * func.proximal(sigma * scalar ** 2)(x * scalar),
        places=places)

    # Verify that for linear functionals, left multiplication is used.
    func = odl.solvers.ZeroFunctional(space)
    assert isinstance(func * scalar, odl.solvers.FunctionalLeftScalarMult)


def test_functional_composition(space):
    """Test composition from the right with an operator."""
    # Less strict checking for single precision
    places = 3 if space.dtype == np.float32 else 5

    func = odl.solvers.L2NormSquared(space)

    # Verify that an error is raised if an invalid operator is used
    # (e.g. wrong range)
    scalar = 2.1
    wrong_space = odl.uniform_discr(1, 2, 10)
    op_wrong = odl.operator.ScalingOperator(wrong_space, scalar)

    with pytest.raises(OpTypeError):
        func * op_wrong

    # Test composition with operator from the right
    op = odl.operator.ScalingOperator(space, scalar)
    func_op_comp = func * op
    assert isinstance(func_op_comp, odl.solvers.Functional)

    x = noise_element(space)
    assert almost_equal(func_op_comp(x), func(op(x)), places=places)

    # Test gradient and derivative with composition from the right
    assert all_almost_equal(func_op_comp.gradient(x),
                            (op.adjoint * func.gradient * op)(x),
                            places=places)

    p = noise_element(space)
    assert all_almost_equal(func_op_comp.derivative(x)(p),
                            (op.adjoint * func.gradient * op)(x).inner(p),
                            places=places)


def test_functional_sum(space):
    """Test for the sum of two functionals."""
    # Less strict checking for single precision
    places = 3 if space.dtype == np.float32 else 5

    func1 = odl.solvers.L2NormSquared(space)
    func2 = odl.solvers.L2Norm(space)

    # Verify that an error is raised if one operand is "wrong"
    op = odl.operator.IdentityOperator(space)
    with pytest.raises(OpTypeError):
        func1 + op

    wrong_space = odl.uniform_discr(1, 2, 10)
    func_wrong_domain = odl.solvers.L2Norm(wrong_space)
    with pytest.raises(OpTypeError):
        func1 + func_wrong_domain

    func_sum = func1 + func2
    x = noise_element(space)
    p = noise_element(space)

    # Test functional evaluation
    assert almost_equal(func_sum(x), func1(x) + func2(x), places=places)

    # Test gradient and derivative
    assert all_almost_equal(func_sum.gradient(x),
                            func1.gradient(x) + func2.gradient(x),
                            places=places)

    assert almost_equal(
        func_sum.derivative(x)(p),
        func1.gradient(x).inner(p) + func2.gradient(x).inner(p),
        places=places)

    # Verify that proximal raises
    with pytest.raises(NotImplementedError):
        func_sum.proximal

    # Test the convex conjugate raises
    with pytest.raises(NotImplementedError):
        func_sum.convex_conj(x)


def test_functional_plus_scalar(space):
    """Test for sum of functioanl and scalar."""
    # Less strict checking for single precision
    places = 3 if space.dtype == np.float32 else 5

    func = odl.solvers.L2NormSquared(space)
    scalar = -1.3

    # Test for scalar not in the field (field of unifor_discr is RealNumbers)
    complex_scalar = 1j
    with pytest.raises(TypeError):
        func + complex_scalar

    func_scalar_sum = func + scalar
    x = noise_element(space)
    p = noise_element(space)

    # Test for evaluation
    assert almost_equal(func_scalar_sum(x), func(x) + scalar, places=places)

    # Test for derivative and gradient
    assert all_almost_equal(func_scalar_sum.gradient(x), func.gradient(x),
                            places=places)

    assert almost_equal(func_scalar_sum.derivative(x)(p),
                        func.gradient(x).inner(p),
                        places=places)

    # Test proximal operator
    sigma = 1.2
    assert all_almost_equal(func_scalar_sum.proximal(sigma)(x),
                            func.proximal(sigma)(x),
                            places=places)

    # Test convex conjugate functional
    assert almost_equal(func_scalar_sum.convex_conj(x),
                        func.convex_conj(x) - scalar,
                        places=places)

    assert all_almost_equal(func_scalar_sum.convex_conj.gradient(x),
                            func.convex_conj.gradient(x),
                            places=places)


def test_translation_of_functional(space):
    """Test for the translation of a functional: (f(. - y))^*."""
    # Less strict checking for single precision
    places = 3 if space.dtype == np.float32 else 5

    # The translation; an element in the domain
    translation = noise_element(space)

    test_functional = odl.solvers.L2NormSquared(space)
    translated_functional = test_functional.translated(translation)
    x = noise_element(space)

    # Test for evaluation of the functional
    expected_result = test_functional(x - translation)
    assert all_almost_equal(translated_functional(x), expected_result,
                            places=places)

    # Test for the gradient
    expected_result = test_functional.gradient(x - translation)
    translated_gradient = translated_functional.gradient
    assert all_almost_equal(translated_gradient(x), expected_result,
                            places=places)

    # Test for proximal
    sigma = 1.2
    # The helper function below is tested explicitly in proximal_utils_test
    expected_result = odl.solvers.proximal_translation(
        test_functional.proximal, translation)(sigma)(x)
    assert all_almost_equal(translated_functional.proximal(sigma)(x),
                            expected_result, places=places)

    # Test for conjugate functional
    # The helper function below is tested explicitly further down in this file
    expected_result = odl.solvers.FunctionalQuadraticPerturb(
        test_functional.convex_conj, linear_term=translation)(x)
    assert all_almost_equal(translated_functional.convex_conj(x),
                            expected_result, places=places)

    # Test for derivative in direction p
    p = noise_element(space)

    # Explicit computation in point x, in direction p: <x/2 + translation, p>
    expected_result = p.inner(test_functional.gradient(x - translation))
    assert all_almost_equal(translated_functional.derivative(x)(p),
                            expected_result, places=places)

    # Test for optimized implementation, when translating a translated
    # functional
    second_translation = noise_element(space)
    double_translated_functional = translated_functional.translated(
        second_translation)

    # Evaluation
    assert almost_equal(double_translated_functional(x),
                        test_functional(x - translation - second_translation),
                        places=places)


def test_multiplication_with_vector(space):
    """Test for multiplying a functional with a vector, both left and right."""
    # Less strict checking for single precision
    places = 3 if space.dtype == np.float32 else 5

    x = noise_element(space)
    y = noise_element(space)
    func = odl.solvers.L2NormSquared(space)

    wrong_space = odl.uniform_discr(1, 2, 10)
    y_other_space = noise_element(wrong_space)

    # Multiplication from the right. Make sure it is a
    # FunctionalRightVectorMult
    func_times_y = func * y
    assert isinstance(func_times_y, odl.solvers.FunctionalRightVectorMult)

    expected_result = func(y * x)
    assert almost_equal(func_times_y(x), expected_result, places=places)

    # Test for the gradient.
    # Explicit calculations: 2*y*y*x
    expected_result = 2.0 * y * y * x
    assert all_almost_equal(func_times_y.gradient(x), expected_result,
                            places=places)

    # Test for convex_conj
    cc_func_times_y = func_times_y.convex_conj
    # Explicit calculations: 1/4 * ||x/y||_2^2
    expected_result = 1.0 / 4.0 * (x / y).norm()**2
    assert almost_equal(cc_func_times_y(x), expected_result, places=places)

    # Make sure that right muliplication is not allowed with vector from
    # another space
    with pytest.raises(TypeError):
        func * y_other_space

    # Multiplication from the left. Make sure it is a FunctionalLeftVectorMult
    y_times_func = y * func
    assert isinstance(y_times_func, odl.FunctionalLeftVectorMult)

    expected_result = y * func(x)
    assert all_almost_equal(y_times_func(x), expected_result, places=places)

    # Now, multiplication with vector from another space is ok (since it is the
    # same as scaling that vector with the scalar returned by the functional).
    y_other_times_func = y_other_space * func
    assert isinstance(y_other_times_func, odl.FunctionalLeftVectorMult)

    expected_result = y_other_space * func(x)
    assert all_almost_equal(y_other_times_func(x), expected_result,
                            places=places)


# Fixtures for test_functional_quadratic_perturb
linear_term = simple_fixture('linear_term', [False, True])
quadratic_coeff = simple_fixture('quadratic_coeff', [0.0, 2.13])


def test_functional_quadratic_perturb(space, linear_term, quadratic_coeff):
    """Test for the functional f(.) + a | . |^2 + <y, .>."""
    # Less strict checking for single precision
    places = 3 if space.dtype == np.float32 else 5

    orig_func = odl.solvers.L2NormSquared(space)

    if linear_term:
        linear_term_arg = None
        linear_term = space.zero()
    else:
        linear_term_arg = linear_term = noise_element(space)

    # Creating the functional ||x||_2^2 and add the quadratic perturbation
    functional = odl.solvers.FunctionalQuadraticPerturb(
        orig_func,
        quadratic_coeff=quadratic_coeff,
        linear_term=linear_term_arg)

    # Create an element in the space, in which to evaluate
    x = noise_element(space)

    # Test for evaluation of the functional
    assert all_almost_equal(functional(x),
                            (orig_func(x) +
                             quadratic_coeff * x.inner(x) +
                             x.inner(linear_term)),
                            places=places)

    # Test for the gradient
    assert all_almost_equal(functional.gradient(x),
                            (orig_func.gradient(x) +
                             2.0 * quadratic_coeff * x +
                             linear_term),
                            places=places)

    # Test for the proximal operator if it exists
    sigma = 1.2
    # Explicit computation gives
    c = 1 / np.sqrt(2 * sigma * quadratic_coeff + 1)
    prox = orig_func.proximal(sigma * c ** 2)
    expected_result = prox((x - sigma * linear_term) * c ** 2)
    assert all_almost_equal(functional.proximal(sigma)(x),
                            expected_result,
                            places=places)

    # Test convex conjugate functional
    if quadratic_coeff == 0:
        expected = orig_func.convex_conj.translated(linear_term)(x)
        assert almost_equal(functional.convex_conj(x),
                            expected,
                            places=places)

    # Test proximal of the convex conjugate
    cconj_prox = odl.solvers.proximal_convex_conj(functional.proximal)
    assert all_almost_equal(
        functional.convex_conj.proximal(sigma)(x),
        cconj_prox(sigma)(x),
        places=places)


if __name__ == '__main__':
    odl.util.test_file(__file__)
