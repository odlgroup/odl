# Copyright 2014-2016 The ODL development group
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

"""Test for the default functionals."""

# External
import numpy as np
import scipy
import pytest

# Internal
import odl
from odl.util.testutils import all_almost_equal, almost_equal, noise_element
from odl.solvers.functional.default_functionals import (
    KullbackLeiblerConvexConj, KullbackLeiblerCrossEntropyConvexConj)

# --- pytest fixtures --- #


scalar_params = [0.01, 2.7, np.array(5.0), 10, -2, -0.2, -np.array(7.1), 0]
scalar_ids = [' scalar={} '.format(s) for s in scalar_params]


@pytest.fixture(scope='module', params=scalar_params, ids=scalar_ids)
def scalar(request):
    return request.param


space_params = ['r10', 'uniform_discr']
space_ids = [' space = {}'.format(p.ljust(10)) for p in space_params]


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def space(request, fn_impl):
    name = request.param.strip()

    if name == 'r10':
        return odl.rn(10, impl=fn_impl)
    elif name == 'uniform_discr':
        return odl.uniform_discr(0, 1, 7, impl=fn_impl)


sigma_params = [0.001, 2.7, np.array(0.5), 10]
sigma_ids = [' sigma={} '.format(s) for s in sigma_params]


@pytest.fixture(scope='module', params=sigma_params, ids=sigma_ids)
def sigma(request):
    return request.param


exponent_params = [1, 2, 1.5, 2.5, -1.6]
exponent_ids = [' exponent={} '.format(s) for s in exponent_params]


@pytest.fixture(scope='module', params=exponent_params, ids=exponent_ids)
def exponent(request):
    return request.param


# --- functional tests --- #


def test_L1_norm(space, sigma):
    """Test the L1-norm."""
    sigma = float(sigma)
    func = odl.solvers.L1Norm(space)
    x = noise_element(space)

    # Test functional evaluation
    expected_result = (np.abs(x)).inner(space.one())
    assert almost_equal(func(x), expected_result)

    # Test gradient - expecting sign function
    expected_result = np.sign(x)
    assert all_almost_equal(func.gradient(x), expected_result)

    # Test proximal - expecting the following:
    #                            |  x_i + sigma, if x_i < -sigma
    #                      z_i = {  0,           if -sigma <= x_i <= sigma
    #                            |  x_i - sigma, if x_i > sigma
    tmp = np.zeros(space.size)
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
    p_norm_x = np.power(np.power(np.abs(x), exponent).inner(one_elem),
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
    assert almost_equal(func(x), expected_result)

    # Test gradient
    if x_norm > 0:
        expected_result = x / x.norm()
        assert all_almost_equal(func.gradient(x), expected_result)

    # Verify that the gradient at zero raises
    with pytest.raises(ValueError):
        func.gradient(func.domain.zero())

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
    assert almost_equal(func_cc_cc(x), func(x))


def test_L2_norm_squared(space, sigma):
    """Test the squared L2-norm."""
    func = odl.solvers.L2NormSquared(space)
    x = noise_element(space)
    x_norm = x.norm()

    # Test functional evaluation
    expected_result = x_norm ** 2
    assert almost_equal(func(x), expected_result)

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
    assert almost_equal(func_cc(x), expected_result)

    # Test gradient of the convex conjugate
    expected_result = x / 2.0
    assert all_almost_equal(func_cc.gradient(x), expected_result)

    # Test proximal of the convex conjugate
    expected_result = x / (1 + sigma / 2.0)
    assert all_almost_equal(func_cc.proximal(sigma)(x), expected_result)

    # Verify that the biconjugate is the functional itself
    func_cc_cc = func_cc.convex_conj

    # Check that they evaluate to the same value
    assert almost_equal(func_cc_cc(x), func(x))

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
    prior = noise_element(space)
    prior = np.abs(prior)

    func = odl.solvers.KullbackLeibler(space, prior)

    # The fucntional is only defined for positive elements
    x = noise_element(space)
    x = np.abs(x)
    one_elem = space.one()

    # Evaluation of the functional
    expected_result = ((x - prior + prior * np.log(prior / x))
                       .inner(one_elem))
    assert almost_equal(func(x), expected_result)

    # Check property for prior
    assert all_almost_equal(func.prior, prior)

    # For elements with (a) negative components it should return inf
    x_neg = noise_element(space)
    x_neg = x_neg - x_neg.ufunc.max()
    assert func(x_neg) == np.inf

    # The gradient
    expected_result = 1 - prior / x
    assert all_almost_equal(func.gradient(x), expected_result)

    # The proximal operator
    sigma = np.random.rand()
    expected_result = odl.solvers.proximal_cconj(
        odl.solvers.proximal_cconj_kl(space, g=prior))(sigma)(x)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result)

    # The convex conjugate functional
    cc_func = func.convex_conj

    assert isinstance(cc_func, KullbackLeiblerConvexConj)

    # The convex conjugate functional is only finite for elements with all
    # components smaller than 1.
    x = noise_element(space)
    x = x - x.ufunc.max() + 0.99

    # Evaluation of convex conjugate
    expected_result = - (prior * np.log(1 - x)).inner(one_elem)
    assert almost_equal(cc_func(x), expected_result)

    x_wrong = noise_element(space)
    x_wrong = x_wrong - x_wrong.ufunc.max() + 1.01
    assert cc_func(x_wrong) == np.inf

    # The gradient of the convex conjugate
    expected_result = prior / (1 - x)
    assert all_almost_equal(cc_func.gradient(x), expected_result)

    # The proximal of the convex conjugate
    expected_result = 0.5 * (1 + x - np.sqrt((x - 1)**2 + 4 * sigma * prior))
    assert all_almost_equal(cc_func.proximal(sigma)(x), expected_result)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    cc_cc_func = cc_func.convex_conj

    # Check that they evaluate the same
    assert almost_equal(cc_cc_func(x), func(x))


def test_kullback_leibler_cross_entorpy(space):
    """Test the kullback leibler cross entropy and its convex conjugate."""
    # The prior needs to be positive
    prior = noise_element(space)
    prior = np.abs(prior)

    func = odl.solvers.KullbackLeiblerCrossEntropy(space, prior)

    # The fucntional is only defined for positive elements
    x = noise_element(space)
    x = np.abs(x)
    one_elem = space.one()

    # Evaluation of the functional
    expected_result = ((prior - x + x * np.log(x / prior))
                       .inner(one_elem))
    assert almost_equal(func(x), expected_result)

    # Check property for prior
    assert all_almost_equal(func.prior, prior)

    # For elements with (a) negative components it should return inf
    x_neg = noise_element(space)
    x_neg = x_neg - x_neg.ufunc.max()
    assert func(x_neg) == np.inf

    # The gradient
    expected_result = np.log(x / prior)
    assert all_almost_equal(func.gradient(x), expected_result)

    # The proximal operator
    sigma = np.random.rand()
    expected_result = odl.solvers.proximal_cconj(
        odl.solvers.proximal_cconj_kl_cross_entropy(space, g=prior))(sigma)(x)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result)

    # The convex conjugate functional
    cc_func = func.convex_conj

    assert isinstance(cc_func, KullbackLeiblerCrossEntropyConvexConj)

    # The convex conjugate functional is defined for all values of x.
    x = noise_element(space)

    # Evaluation of convex conjugate
    expected_result = (prior * (np.exp(x) - 1)).inner(one_elem)
    assert almost_equal(cc_func(x), expected_result)

    # The gradient of the convex conjugate
    expected_result = prior * np.exp(x)
    assert all_almost_equal(cc_func.gradient(x), expected_result)

    # The proximal of the convex conjugate
    expected_result = x - scipy.special.lambertw(sigma * prior * np.exp(x))
    assert all_almost_equal(cc_func.proximal(sigma)(x), expected_result)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    cc_cc_func = cc_func.convex_conj

    # Check that they evaluate the same
    assert almost_equal(cc_cc_func(x), func(x))


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
    assert almost_equal(func(x), expected_result)

    # Also test with some values as none
    func_no_offset = odl.solvers.QuadraticForm(operator, constant=constant)
    expected_result = x.inner(operator(x)) + constant
    assert almost_equal(func_no_offset(x), expected_result)

    func_no_operator = odl.solvers.QuadraticForm(vector=vector,
                                                 constant=constant)
    expected_result = vector.inner(x) + constant
    assert almost_equal(func_no_operator(x), expected_result)

    # The gradient
    expected_gradient = 2 * operator(x) + vector
    assert all_almost_equal(func.gradient(x), expected_gradient)

    # The convex conjugate
    assert isinstance(func.convex_conj, odl.solvers.QuadraticForm)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
