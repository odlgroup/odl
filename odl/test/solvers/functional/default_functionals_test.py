# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the default functionals."""

from __future__ import division
import numpy as np
import scipy.special
import pytest

import odl
from odl.util.testutils import all_almost_equal, noise_element, simple_fixture
from odl.solvers.functional.default_functionals import (
    KullbackLeiblerConvexConj, KullbackLeiblerCrossEntropyConvexConj)


# --- pytest fixtures --- #


scalar = simple_fixture('scalar', [0.01, 2.7, np.array(5.0), 10, -2, -0.2,
                                   -np.array(7.1), 0])
sigma = simple_fixture('sigma', [0.001, 2.7, 10])
exponent = simple_fixture('sigma', [1, 2, 1.5, 2.5, -1.6])


space_params = ['r10', 'uniform_discr', 'power_space_unif_discr']
space_ids = [' space={} '.format(p) for p in space_params]


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def space(request, odl_tspace_impl):
    name = request.param.strip()
    impl = odl_tspace_impl

    if name == 'r10':
        return odl.rn(10, impl=impl)
    elif name == 'uniform_discr':
        return odl.uniform_discr(0, 1, 7, impl=impl)
    elif name == 'power_space_unif_discr':
        # Discretization parameters
        space = odl.uniform_discr(0, 1, 7, impl=impl)
        return odl.ProductSpace(space, 2)

# --- functional tests --- #


def test_L1_norm(space, sigma):
    """Test the L1-norm."""
    sigma = float(sigma)
    func = odl.solvers.L1Norm(space)
    x = noise_element(space)

    # Test functional evaluation
    expected_result = np.abs(x).inner(space.one())
    assert func(x) == pytest.approx(expected_result)

    # Test gradient - expecting sign function
    expected_result = func.domain.element(np.sign(x))
    assert all_almost_equal(func.gradient(x), expected_result)

    # Test proximal - expecting the following:
    #                            |  x_i + sigma, if x_i < -sigma
    #                      z_i = {  0,           if -sigma <= x_i <= sigma
    #                            |  x_i - sigma, if x_i > sigma
    tmp = np.zeros(space.shape)
    orig = x.asarray()
    tmp[orig > sigma] = orig[orig > sigma] - sigma
    tmp[orig < -sigma] = orig[orig < -sigma] + sigma
    expected_result = space.element(tmp)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result)

    # Test convex conjugate - expecting 0 if |x|_inf <= 1, infty else
    func_cc = func.convex_conj
    norm_larger_than_one = 1.1 * x / np.max(np.abs(x))
    assert func_cc(norm_larger_than_one) == np.inf

    norm_less_than_one = 0.9 * x / np.max(np.abs(x))
    assert func_cc(norm_less_than_one) == 0

    norm_equal_to_one = x / np.max(np.abs(x))
    assert func_cc(norm_equal_to_one) == 0

    # Gradient of the convex conjugate (not implemeted)
    with pytest.raises(NotImplementedError):
        func_cc.gradient

    # Test proximal of the convex conjugate - expecting x / max(1, |x|)
    expected_result = x / np.maximum(1, np.abs(x))
    assert all_almost_equal(func_cc.proximal(sigma)(x), expected_result)

    # Verify that the biconjugate is the functional itself
    func_cc_cc = func_cc.convex_conj
    assert isinstance(func_cc_cc, odl.solvers.L1Norm)


def test_indicator_lp_unit_ball(space, sigma, exponent):
    """Test for indicator function on unit ball."""
    x = noise_element(space)
    one_elem = space.one()

    func = odl.solvers.IndicatorLpUnitBall(space, exponent)

    # Test functional evaluation
    p_norm_x = np.power(
        func.domain.element(np.power(np.abs(x), exponent)).inner(one_elem),
        1.0 / exponent)

    norm_larger_than_one = 1.01 * x / p_norm_x
    assert func(norm_larger_than_one) == np.inf

    norm_less_than_one = 0.99 * x / p_norm_x
    assert func(norm_less_than_one) == 0


def test_L2_norm(space, sigma):
    """Test the L2-norm."""
    func = odl.solvers.L2Norm(space)
    x = noise_element(space)
    x_norm = x.norm()

    # Test functional evaluation
    expected_result = np.sqrt((x ** 2).inner(space.one()))
    assert func(x) == pytest.approx(expected_result)

    # Test gradient
    if x_norm > 0:
        expected_result = x / x.norm()
        assert all_almost_equal(func.gradient(x), expected_result)

    # Verify that the gradient at zero is zero
    assert all_almost_equal(func.gradient(func.domain.zero()), space.zero())

    # Test proximal operator - expecting
    # x * (1 - sigma/||x||) if ||x|| > sigma, 0 else
    norm_less_than_sigma = 0.99 * sigma * x / x_norm
    assert all_almost_equal(func.proximal(sigma)(norm_less_than_sigma),
                            space.zero())

    norm_larger_than_sigma = 1.01 * sigma * x / x_norm
    expected_result = (norm_larger_than_sigma *
                       (1.0 - sigma / norm_larger_than_sigma.norm()))
    assert all_almost_equal(func.proximal(sigma)(norm_larger_than_sigma),
                            expected_result)

    # Test convex conjugate
    func_cc = func.convex_conj

    # Test evaluation of the convex conjugate - expecting
    # 0 if ||x|| < 1, infty else
    norm_larger_than_one = 1.01 * x / x_norm
    assert func_cc(norm_larger_than_one) == np.inf

    norm_less_than_one = 0.99 * x / x_norm
    assert func_cc(norm_less_than_one) == 0

    # Gradient of the convex conjugate (not implemeted)
    with pytest.raises(NotImplementedError):
        func_cc.gradient

    # Test the proximal of the convex conjugate - expecting
    # x if ||x||_2 < 1, x/||x|| else
    if x_norm < 1:
        expected_result = x
    else:
        expected_result = x / x_norm
    assert all_almost_equal(func_cc.proximal(sigma)(x), expected_result)

    # Verify that the biconjugate is the functional itself
    func_cc_cc = func_cc.convex_conj
    assert func_cc_cc(x) == pytest.approx(func(x))


def test_L2_norm_squared(space, sigma):
    """Test the squared L2-norm."""
    func = odl.solvers.L2NormSquared(space)
    x = noise_element(space)
    x_norm = x.norm()

    # Test functional evaluation
    expected_result = x_norm ** 2
    assert func(x) == pytest.approx(expected_result)

    # Test gradient
    expected_result = 2.0 * x
    assert all_almost_equal(func.gradient(x), expected_result)

    # Test proximal operator
    expected_result = x / (1 + 2.0 * sigma)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result)

    # Test convex conjugate
    func_cc = func.convex_conj

    # Test evaluation of the convex conjugate
    expected_result = x_norm ** 2 / 4.0
    assert func_cc(x) == pytest.approx(expected_result)

    # Test gradient of the convex conjugate
    expected_result = x / 2.0
    assert all_almost_equal(func_cc.gradient(x), expected_result)

    # Test proximal of the convex conjugate
    expected_result = x / (1 + sigma / 2.0)
    assert all_almost_equal(func_cc.proximal(sigma)(x), expected_result)

    # Verify that the biconjugate is the functional itself
    func_cc_cc = func_cc.convex_conj

    # Check that they evaluate to the same value
    assert func_cc_cc(x) == pytest.approx(func(x))

    # Check that their gradients evaluate to the same value
    assert all_almost_equal(func_cc_cc.gradient(x), func.gradient(x))


def test_constant_functional(space, scalar):
    """Test the constant functional."""
    constant = float(scalar)
    func = odl.solvers.ConstantFunctional(space, constant=scalar)
    x = noise_element(space)

    assert func.constant == constant

    # Test functional evaluation
    assert func(x) == constant

    # Test gradient - expecting zero operator
    assert isinstance(func.gradient, odl.ZeroOperator)

    # Test proximal operator - expecting identity
    sigma = 1.5
    assert isinstance(func.proximal(sigma), odl.IdentityOperator)

    # Test convex conjugate
    func_cc = func.convex_conj

    # Test evaluation of the convex conjugate - expecting
    # -constant if x=0, infty else
    assert func_cc(x) == np.inf
    assert func_cc(space.zero()) == -constant

    # Gradient of the convex conjugate (not implemeted)
    with pytest.raises(NotImplementedError):
        func_cc.gradient

    # Proximal of the convex conjugate - expecting zero operator
    assert isinstance(func_cc.proximal(sigma), odl.ZeroOperator)

    # Verify that the biconjugate is the functional itself
    func_cc_cc = func_cc.convex_conj
    assert isinstance(func_cc_cc, odl.solvers.ConstantFunctional)
    assert func_cc_cc.constant == constant


def test_zero_functional(space):
    """Test the zero functional."""
    zero_func = odl.solvers.ZeroFunctional(space)
    assert isinstance(zero_func, odl.solvers.ConstantFunctional)
    assert zero_func.constant == 0


def test_kullback_leibler(space):
    """Test the kullback leibler functional and its convex conjugate."""
    # The prior needs to be positive
    prior = np.abs(noise_element(space)) + 0.1

    func = odl.solvers.KullbackLeibler(space, prior)

    # The fucntional is only defined for positive elements
    x = np.abs(noise_element(space)) + 0.1
    one_elem = space.one()

    # Evaluation of the functional
    expected_result = (
        x - prior + prior * np.log(prior / x)
    ).inner(one_elem)
    assert func(x) == pytest.approx(expected_result)

    # Check property for prior
    assert all_almost_equal(func.prior, prior)

    # For elements with (a) negative components it should return inf
    x_neg = noise_element(space)
    x_neg = x_neg - x_neg.ufuncs.max()
    assert func(x_neg) == np.inf

    # The gradient
    expected_result = 1 - prior / x
    assert all_almost_equal(func.gradient(x), expected_result)

    # The proximal operator
    sigma = np.random.rand()
    expected_result = odl.solvers.proximal_convex_conj(
        odl.solvers.proximal_convex_conj_kl(space, g=prior))(sigma)(x)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result)

    # The convex conjugate functional
    cc_func = func.convex_conj

    assert isinstance(cc_func, KullbackLeiblerConvexConj)

    # The convex conjugate functional is only finite for elements with all
    # components smaller than 1.
    x = noise_element(space)
    x = x - x.ufuncs.max() + 0.99

    # Evaluation of convex conjugate
    expected_result = - (prior * np.log(1 - x)).inner(one_elem)
    assert cc_func(x) == pytest.approx(expected_result)

    x_wrong = noise_element(space)
    x_wrong = x_wrong - x_wrong.ufuncs.max() + 1.01
    assert cc_func(x_wrong) == np.inf

    # The gradient of the convex conjugate
    expected_result = prior / (1 - x)
    assert all_almost_equal(cc_func.gradient(x), expected_result)

    # The proximal of the convex conjugate
    expected_result = 0.5 * (1 + x - np.sqrt((x - 1) ** 2 + 4 * sigma * prior))
    assert all_almost_equal(cc_func.proximal(sigma)(x), expected_result)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    cc_cc_func = cc_func.convex_conj

    # Check that they evaluate the same
    assert cc_cc_func(x) == pytest.approx(func(x))


def test_kullback_leibler_cross_entorpy(space):
    """Test the kullback leibler cross entropy and its convex conjugate."""
    # The prior needs to be positive
    prior = noise_element(space)
    prior = space.element(np.abs(prior))

    func = odl.solvers.KullbackLeiblerCrossEntropy(space, prior)

    # The fucntional is only defined for positive elements
    x = noise_element(space)
    x = func.domain.element(np.abs(x))
    one_elem = space.one()

    # Evaluation of the functional
    expected_result = ((prior - x + x * np.log(x / prior))
                       .inner(one_elem))
    assert func(x) == pytest.approx(expected_result)

    # Check property for prior
    assert all_almost_equal(func.prior, prior)

    # For elements with (a) negative components it should return inf
    x_neg = noise_element(space)
    x_neg = x_neg - x_neg.ufuncs.max()
    assert func(x_neg) == np.inf

    # The gradient
    expected_result = np.log(x / prior)
    assert all_almost_equal(func.gradient(x), expected_result)

    # The proximal operator
    sigma = np.random.rand()
    prox = odl.solvers.proximal_convex_conj(
        odl.solvers.proximal_convex_conj_kl_cross_entropy(space, g=prior))
    expected_result = prox(sigma)(x)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result)

    # The convex conjugate functional
    cc_func = func.convex_conj

    assert isinstance(cc_func, KullbackLeiblerCrossEntropyConvexConj)

    # The convex conjugate functional is defined for all values of x.
    x = noise_element(space)

    # Evaluation of convex conjugate
    expected_result = (prior * (np.exp(x) - 1)).inner(one_elem)
    assert cc_func(x) == pytest.approx(expected_result)

    # The gradient of the convex conjugate
    expected_result = prior * np.exp(x)
    assert all_almost_equal(cc_func.gradient(x), expected_result)

    # The proximal of the convex conjugate
    expected_result = (x -
                       scipy.special.lambertw(sigma * prior * np.exp(x)).real)
    assert all_almost_equal(cc_func.proximal(sigma)(x), expected_result)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    cc_cc_func = cc_func.convex_conj

    # Check that they evaluate the same
    assert cc_cc_func(x) == pytest.approx(func(x))


def test_quadratic_form(space):
    """Test the quadratic form functional."""
    operator = odl.IdentityOperator(space)
    vector = space.one()
    constant = 0.363
    func = odl.solvers.QuadraticForm(operator, vector, constant)

    x = noise_element(space)

    # Checking that values is stored correctly
    assert func.operator == operator
    assert func.vector == vector
    assert func.constant == constant

    # Evaluation of the functional
    expected_result = x.inner(operator(x)) + vector.inner(x) + constant
    assert func(x) == pytest.approx(expected_result)

    # The gradient
    expected_gradient = 2 * operator(x) + vector
    assert all_almost_equal(func.gradient(x), expected_gradient)

    # The convex conjugate
    assert isinstance(func.convex_conj, odl.solvers.QuadraticForm)

    # Test for linear functional
    func_no_operator = odl.solvers.QuadraticForm(vector=vector,
                                                 constant=constant)
    expected_result = vector.inner(x) + constant
    assert func_no_operator(x) == pytest.approx(expected_result)

    expected_gradient = vector
    assert all_almost_equal(func_no_operator.gradient(x), expected_gradient)

    # The convex conjugate is a translation of the IndicatorZero
    func_no_operator_cc = func_no_operator.convex_conj
    assert isinstance(func_no_operator_cc,
                      odl.solvers.FunctionalTranslation)
    assert isinstance(func_no_operator_cc.functional,
                      odl.solvers.IndicatorZero)
    assert func_no_operator_cc(vector) == -constant
    assert np.isinf(func_no_operator_cc(vector + 2.463))

    # Test with no offset
    func_no_offset = odl.solvers.QuadraticForm(operator, constant=constant)
    expected_result = x.inner(operator(x)) + constant
    assert func_no_offset(x) == pytest.approx(expected_result)


def test_separable_sum(space):
    """Test for the separable sum."""
    l1 = odl.solvers.L1Norm(space)
    l2 = odl.solvers.L2Norm(space)

    x = noise_element(space)
    y = noise_element(space)

    # Initialization and calling
    func = odl.solvers.SeparableSum(l1, l2)
    assert func([x, y]) == pytest.approx(l1(x) + l2(y))

    power_func = odl.solvers.SeparableSum(l1, 5)
    assert power_func([x, x, x, x, x]) == pytest.approx(5 * l1(x))

    # Gradient
    grad = func.gradient([x, y])
    assert grad[0] == l1.gradient(x)
    assert grad[1] == l2.gradient(y)

    # Proximal
    sigma = 1.0
    prox = func.proximal(sigma)([x, y])
    assert prox[0] == l1.proximal(sigma)(x)
    assert prox[1] == l2.proximal(sigma)(y)

    # Convex conjugate
    assert func.convex_conj([x, y]) == l1.convex_conj(x) + l2.convex_conj(y)


def test_moreau_envelope_l1():
    """Test for the Moreau envelope with L1 norm."""
    space = odl.rn(3)

    l1 = odl.solvers.L1Norm(space)

    # Test l1 norm, gives "Huber norm"
    smoothed_l1 = odl.solvers.MoreauEnvelope(l1)
    assert all_almost_equal(smoothed_l1.gradient([0, -0.2, 0.7]),
                            [0, -0.2, 0.7])
    assert all_almost_equal(smoothed_l1.gradient([-3, 2, 10]),
                            [-1, 1, 1])

    # Test with different sigma
    smoothed_l1 = odl.solvers.MoreauEnvelope(l1, sigma=0.5)
    assert all_almost_equal(smoothed_l1.gradient([0, 0.2, 0.7]),
                            [0, 0.4, 1.0])


def test_moreau_envelope_l2_sq(space, sigma):
    """Test for the Moreau envelope with l2 norm squared."""

    # Result is ||x||_2^2 / (1 + 2 sigma)
    # Gradient is x * 2 / (1 + 2 * sigma)
    l2_sq = odl.solvers.L2NormSquared(space)

    smoothed_l2_sq = odl.solvers.MoreauEnvelope(l2_sq, sigma=sigma)
    x = noise_element(space)
    assert all_almost_equal(smoothed_l2_sq.gradient(x),
                            x * 2 / (1 + 2 * sigma))


def test_weighted_separablesum(space):
    """Test for the weighted proximal of a SeparableSum functional."""

    l1 = odl.solvers.L1Norm(space)
    l2 = odl.solvers.L2Norm(space)
    func = odl.solvers.SeparableSum(l1, l2)

    x = func.domain.one()

    sigma = [0.5, 1.0]

    prox = func.proximal(sigma)(x)
    assert all_almost_equal(prox, [l1.proximal(sigma[0])(x[0]),
                                   l2.proximal(sigma[1])(x[1])])


def test_weighted_proximal_L2_norm_squared(space):
    """Test for the weighted proximal of the squared L2 norm"""

    # Define the functional on the space.
    func = odl.solvers.L2NormSquared(space)

    # Set the stepsize as a random element of the spaces
    # with elements between 1 and 10.
    sigma = odl.phantom.uniform_noise(space, 1, 10)

    # Start at the one vector.
    x = space.one()

    # Calculate the proximal point in-place and out-of-place
    p_ip = space.element()
    func.proximal(sigma)(x, out=p_ip)
    p_oop = func.proximal(sigma)(x)

    # Both should contain the same vector now.
    assert all_almost_equal(p_ip, p_oop)

    # Check if the subdifferential inequalities are satisfied.
    # p = prox_{sigma * f}(x) iff (x - p)/sigma = grad f(p)
    assert all_almost_equal(func.gradient(p_ip),
                            (x - p_ip) / sigma)


def test_weighted_proximal_L1_norm_far(space):
    """Test for the weighted proximal of the L1 norm away from zero"""

    # Define the functional on the space.
    func = odl.solvers.L1Norm(space)

    # Set the stepsize as a random element of the spaces
    # with elements between 1 and 10.
    sigma = odl.phantom.noise.uniform_noise(space, 1, 10)

    # Start far away from zero so that the L1 norm will be differentiable
    # at the result.
    x = 100 * space.one()

    # Calculate the proximal point in-place and out-of-place
    p_ip = space.element()
    func.proximal(sigma)(x, out=p_ip)
    p_oop = func.proximal(sigma)(x)

    # Both should contain the same vector now.
    assert all_almost_equal(p_ip, p_oop)

    # Check if the subdifferential inequalities are satisfied.
    # p = prox_{sigma * f}(x) iff (x - p)/sigma = grad f(p)
    assert all_almost_equal(func.gradient(p_ip), (x - p_ip) / sigma)


def test_weighted_proximal_L1_norm_close(space):
    """Test for the weighted proximal of the L1 norm near zero"""

    # Set the space.
    space = odl.rn(5)

    # Define the functional on the space.
    func = odl.solvers.L1Norm(space)

    # Set the stepsize.
    sigma = [0.1, 0.2, 0.5, 1.0, 2.0]

    # Set the starting point.
    x = 0.5 * space.one()

    # Calculate the proximal point in-place and out-of-place
    p_ip = space.element()
    func.proximal(sigma)(x, out=p_ip)
    p_oop = func.proximal(sigma)(x)

    # Both should contain the same vector now.
    assert all_almost_equal(p_ip, p_oop)

    # Check if this equals the expected result.
    expected_result = [0.4, 0.3, 0.0, 0.0, 0.0]
    assert all_almost_equal(expected_result, p_ip)


def test_bregman_functional_no_gradient(space):
    """Test Bregman distance for functional without gradient."""

    ind_func = odl.solvers.IndicatorNonnegativity(space)
    point = np.abs(noise_element(space))
    subgrad = noise_element(space)  # Any element in the domain is ok
    bregman_dist = odl.solvers.BregmanDistance(ind_func, point, subgrad)

    x = np.abs(noise_element(space))

    expected_result = -subgrad.inner(x - point)
    assert all_almost_equal(bregman_dist(x), expected_result)

    # However, since the functional is not differentialbe we cannot call the
    # gradient of the Bregman distance functional
    with pytest.raises(NotImplementedError):
        bregman_dist.gradient


def test_bregman_functional_l2_squared(space, sigma):
    """Test Bregman distance using l2 norm squared as underlying functional."""
    sigma = float(sigma)

    l2_sq = odl.solvers.L2NormSquared(space)
    point = noise_element(space)
    subgrad = l2_sq.gradient(point)
    bregman_dist = odl.solvers.BregmanDistance(l2_sq, point, subgrad)

    expected_func = odl.solvers.L2NormSquared(space).translated(point)

    x = noise_element(space)

    # Function evaluation
    assert all_almost_equal(bregman_dist(x), expected_func(x))

    # Gradient evaluation
    assert all_almost_equal(bregman_dist.gradient(x),
                            expected_func.gradient(x))

    # Convex conjugate
    cc_bregman_dist = bregman_dist.convex_conj
    cc_expected_func = expected_func.convex_conj
    assert all_almost_equal(cc_bregman_dist(x), cc_expected_func(x))

    # Proximal operator
    prox_bregman_dist = bregman_dist.proximal(sigma)
    prox_expected_func = expected_func.proximal(sigma)
    assert all_almost_equal(prox_bregman_dist(x), prox_expected_func(x))


if __name__ == '__main__':
    odl.util.test_file(__file__)
