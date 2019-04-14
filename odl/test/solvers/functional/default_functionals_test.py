# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the default functionals."""

from __future__ import division

import numpy as np
import pytest
import scipy.special

import odl
from odl.util.testutils import all_almost_equal, noise_element, simple_fixture

# --- pytest fixtures --- #


scalar = simple_fixture(
    'scalar', [0.01, 2.7, np.array(5.0), 10, -2, -0.2, -np.array(7.1), 0]
)
sigma = simple_fixture('sigma', [0.001, 2.7, 10])
exponent = simple_fixture('exponent', [1, 2, 1.5, 2.5])


space_params = ['r10', 'Lp', 'Lp ** 2']
space_ids = [' space={} '.format(p) for p in space_params]


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def space(request, odl_tspace_impl):
    name = request.param.strip()
    impl = odl_tspace_impl

    if name == 'r10':
        return odl.rn(10, impl=impl)
    elif name == 'Lp':
        return odl.uniform_discr(0, 1, 7, impl=impl)
    elif name == 'Lp ** 2':
        space = odl.uniform_discr(0, 1, 7, impl=impl)
        return odl.ProductSpace(space, 2)

# --- functional tests --- #


def _test_op(op, x, true_res):
    assert op.domain == op.range

    assert all_almost_equal(op(x), true_res)  # out-of-place
    out = op.range.element()
    op(x, out=out)
    assert all_almost_equal(out, true_res)  # in-place
    y = op.domain.copy(x)
    op(y, out=y)
    assert all_almost_equal(y, true_res)  # in-place, aliased


def test_L1_norm(space, sigma):
    """Test the L1-norm."""
    func = odl.solvers.L1Norm(space)
    x = noise_element(space)
    F = space.ufuncs
    R = space.reduce
    sigma = float(sigma)

    # Evaluation
    assert func(x) == pytest.approx(space.inner(F.abs(x), space.one()))

    # Gradient
    func_grad_x = F.sign(x)
    _test_op(func.gradient, x, func_grad_x)

    # Proximal
    func_prox_x = F.sign(x) * F.maximum(F.abs(x) - sigma, 0)
    _test_op(func.proximal(sigma), x, func_prox_x)

    # CC evaluation
    # NB: L1 CC is the indicator of the inf-norm unit ball, not covered by
    # `test_indicator_lp_unit_ball`
    func_cc = func.convex_conj
    inf_norm_x = R.max(F.abs(x))
    # Elements with norm > 1 --> inf
    for c in [2.0, 1.1, 1 + 1e-5]:
        norm_gt_1 = (c / inf_norm_x) * x
        assert func_cc(norm_gt_1) == np.inf, 'c={}'.format(c)

    # Elements with norm < 1 --> 0
    for c in [0.0, 0.9, 1 - 1e-5]:
        norm_lt_1 = (c / inf_norm_x) * x
        assert func_cc(norm_lt_1) == 0, 'c={}'.format(c)

    # CC gradient not implemented

    # CC proximal
    func_cc_prox_x = x / F.maximum(1, F.abs(x))
    _test_op(func_cc.proximal(sigma), x, func_cc_prox_x)

    # Biconjugate
    func_cc_cc = func_cc.convex_conj
    assert func_cc_cc(x) == pytest.approx(func(x))


def test_indicator_lp_unit_ball(space, sigma, exponent):
    """Test for indicator function on unit ball."""
    func = odl.solvers.IndicatorLpUnitBall(space, exponent)
    x = noise_element(space)
    F = space.ufuncs

    # Evaluation
    p_norm_x = np.power(
        space.inner(F.power(F.abs(x), exponent), space.one()), 1 / exponent
    )
    for c in [2.0, 1.1, 1 + 1e-5]:
        norm_gt_1 = (c / p_norm_x) * x
        assert func(norm_gt_1) == np.inf, 'c={}'.format(c)

    for c in [0.0, 0.9, 1 - 1e-5]:
        norm_lt_1 = (c / p_norm_x) * x
        assert func(norm_lt_1) == 0, 'c={}'.format(c)

    # Gradient not implemented

    # Proximal
    if exponent in {2, float('inf')}:
        func_prox_x = x if p_norm_x <= 1 else x / p_norm_x
        _test_op(func.proximal(sigma), x, func_prox_x)


def test_L2_norm(space, sigma):
    """Test the L2-norm."""
    func = odl.solvers.L2Norm(space)
    x = noise_element(space)
    x_norm = space.norm(x)
    zero = space.zero()

    # Evaluation
    assert func(x) == pytest.approx(np.sqrt(space.inner(x ** 2, space.one())))

    # Gradient
    func_grad_x = x / x_norm
    _test_op(func.gradient, x, func_grad_x)
    assert all_almost_equal(func.gradient(zero), zero)

    # Proximal
    # x * (1 - sigma/||x||) if ||x|| > sigma, else 0
    for c in [2.0, 1.1, 1 + 1e-5]:
        # ||y|| > sigma
        y = (c * sigma / x_norm) * x
        func_prox_y = y * (1 - sigma / space.norm(y))
        _test_op(func.proximal(sigma), y, func_prox_y)

    for c in [0.0, 0.9, 1 - 1e-5]:
        # ||y|| < sigma
        y = (c * sigma / x_norm) * x
        func_prox_y = zero
        _test_op(func.proximal(sigma), y, func_prox_y)

    # CC evaluation
    # 0 if ||x|| < 1, else infty
    func_cc = func.convex_conj
    for c in [2.0, 1.1, 1 + 1e-5]:
        # ||y|| > 1
        y = (c / x_norm) * x
        assert func_cc(y) == np.inf

    for c in [0.0, 0.9, 1 - 1e-5]:
        # ||y|| < 1
        y = (c / x_norm) * x
        assert func_cc(y) == 0

    # CC gradient not implemented

    # CC proximal
    # x if ||x||_2 < 1, else x/||x||
    for c in [2.0, 1.1, 1 + 1e-5]:
        # ||y|| > 1
        y = (c / x_norm) * x
        func_cc_prox_y = x / x_norm
        _test_op(func_cc.proximal(sigma), y, func_cc_prox_y)

    for c in [0.0, 0.9, 1 - 1e-5]:
        # ||y|| < 1
        y = (c / x_norm) * x
        func_cc_prox_y = y
        _test_op(func_cc.proximal(sigma), y, func_cc_prox_y)

    func_cc_cc = func_cc.convex_conj
    assert func_cc_cc(x) == pytest.approx(func(x))


def test_L2_norm_squared(space, sigma):
    """Test the squared L2-norm."""
    func = odl.solvers.L2NormSquared(space)
    x = noise_element(space)
    x_norm = space.norm(x)

    # Evaluation
    assert func(x) == pytest.approx(x_norm ** 2)

    # Gradient
    func_grad_x = 2 * x
    _test_op(func.gradient, x, func_grad_x)

    # Proximal
    func_prox_x = x / (1 + 2 * sigma)
    _test_op(func.proximal(sigma), x, func_prox_x)

    # CC Evaluation
    func_cc = func.convex_conj
    assert func_cc(x) == pytest.approx(x_norm ** 2 / 4)

    # CC Gradient
    func_cc_grad_x = x / 2
    _test_op(func_cc.gradient, x, func_cc_grad_x)

    # CC Proximal
    func_prox_cc_x = x / (1 + sigma / 2)
    _test_op(func_cc.proximal(sigma), x, func_prox_cc_x)

    # Biconjugate
    func_cc_cc = func_cc.convex_conj
    assert func_cc_cc(x) == pytest.approx(func(x))


def test_constant_functional(space, scalar):
    """Test the constant functional."""
    constant = float(scalar)
    func = odl.solvers.ConstantFunctional(space, constant=scalar)
    x = noise_element(space)
    sigma = 1.5

    assert func(x) == constant
    assert isinstance(func.gradient, odl.ZeroOperator)
    assert isinstance(func.proximal(sigma), odl.IdentityOperator)

    # CC Evaluation
    # -constant if x = 0, else infty
    func_cc = func.convex_conj
    assert func_cc(x) == np.inf
    assert func_cc(space.zero()) == -constant

    # CC Gradient not implemented

    # CC Proximal
    assert isinstance(func_cc.proximal(sigma), odl.ZeroOperator)

    # Biconjugate
    func_cc_cc = func_cc.convex_conj
    assert isinstance(func_cc_cc, odl.solvers.ConstantFunctional)
    assert func_cc_cc.constant == constant


def test_zero_functional(space):
    """Test the zero functional."""
    zero_func = odl.solvers.ZeroFunctional(space)
    assert zero_func(space.one()) == 0


def test_kullback_leibler(space):
    """Test the Kullback-Leibler functional and its convex conjugate."""
    F = space.ufuncs
    R = space.reduce
    prior = F.abs(noise_element(space)) + 0.1  # must be positive
    func = odl.solvers.KullbackLeibler(space, prior)
    x = F.abs(noise_element(space)) + 0.1  # must be positive
    one = space.one()
    sigma = 1.2

    # Evaluation
    assert func(x) == pytest.approx(
        space.inner((x - prior + prior * F.log(prior / x)), one)
    )
    # If any component is nonpositive, the result should be infinity
    assert func(-x) == np.inf
    if not isinstance(space, odl.ProductSpace):
        y = space.copy(x)
        y[0] = 0
        assert func(y) == np.inf
    # Points where `prior` is 0 should contribute 0
    if not isinstance(space, odl.ProductSpace):
        prior2 = space.zero()
        prior2[:prior2.shape[0] // 2] = 2
        # Fraction of nonzero elements in prior2
        nz_frac = (prior2.shape[0] // 2) / prior2.shape[0]
        func2 = odl.solvers.KullbackLeibler(space, prior2)
        # Where prior is 1, we integrate y - 2 + 2 * log(2 / y),
        # elsewhere we integrate y
        # Testing with y = 1
        assert func2(one) == pytest.approx(
            (-1 + 2 * np.log(2)) * space.inner(one, one) * nz_frac
            + space.inner(one, one) * (1 - nz_frac)
        )

    # Gradient
    func_grad_x = 1 - prior / x
    _test_op(func.gradient, x, func_grad_x)

    # Proximal
    func_prox_x = (
        x - sigma + F.sqrt((x - sigma) ** 2 + 4 * sigma * prior)
    ) / 2
    _test_op(func.proximal(sigma), x, func_prox_x)

    # CC evaluation
    func_cc = func.convex_conj
    # integral of -prior * log(1 - x) if x < 1 everywhere, otherwise infinity
    y = 0.99 * x / R.max(x)  # max(y) < 1
    assert func_cc(y) == pytest.approx(
        -space.inner(prior * F.log(1 - y), one)
    )
    y = 1.01 * x / R.max(x)  # max(y) > 1
    assert func_cc(y) == np.inf

    # CC gradient
    y = 0.99 * x / R.max(x)
    func_cc_grad_y = prior / (1 - y)
    _test_op(func_cc.gradient, y, func_cc_grad_y)

    # CC proximal
    y = 0.99 * x / R.max(x)
    func_cc_prox_y = (y + 1 - F.sqrt((y - 1) ** 2 + 4 * sigma * prior)) / 2
    _test_op(func_cc.proximal(sigma), y, func_cc_prox_y)

    # Biconjugate
    func_cc_cc = func_cc.convex_conj
    assert func_cc_cc(x) == pytest.approx(func(x))


def test_kullback_leibler_cross_entropy(space):
    """Test the kullback leibler cross entropy and its convex conjugate."""
    F = space.ufuncs
    prior = F.abs(noise_element(space)) + 0.1  # must be positive
    func = odl.solvers.KullbackLeiblerCrossEntropy(space, prior)
    x = F.abs(noise_element(space)) + 0.1  # must be positive
    one = space.one()
    zero = space.zero()
    sigma = 1.2

    # Evaluation
    assert func(x) == pytest.approx(
        space.inner(prior - x + x * F.log(x / prior), one)
    )
    # At x=0, the expected value is the integral of the prior
    assert func(zero) == pytest.approx(space.inner(prior, one))
    # If any component is negative, the result should be infinity
    assert func(-x) == np.inf
    # If any prior component is 0, the result should be infinity
    if not isinstance(space, odl.ProductSpace):
        prior2 = space.copy(x)
        prior2[0] = 0
        func2 = odl.solvers.KullbackLeiblerCrossEntropy(space, prior2)
        assert func2(x) == np.inf

    # Gradient
    func_grad_x = prior - 1 + F.log(x / prior)
    _test_op(func.gradient, x, func_grad_x)

    # Proximal
    # sigma * W(prior * exp(x / sigma) / sigma)
    if isinstance(space, odl.ProductSpace):
        arg = prior * F.exp(x / sigma) / sigma
        func_prox_x = sigma * space.apply(scipy.special.lambertw, arg).real
    else:
        func_prox_x = sigma * scipy.special.lambertw(
            prior * F.exp(x / sigma) / sigma
        ).real
    _test_op(func.proximal(sigma), x, func_prox_x)

    # CC evaluation
    # integral of prior * (exp(x) - 1)
    func_cc = func.convex_conj
    x = noise_element(space)  # convex conjugate is defined for any x
    assert func_cc(x) == pytest.approx(
        space.inner(prior * (F.exp(x) - 1), one)
    )

    # CC gradient
    func_cc_grad_x = prior * F.exp(x)
    _test_op(func_cc.gradient, x, func_cc_grad_x)

    # CC proximal
    # x - W(prior * exp(x) * sigma)
    if isinstance(space, odl.ProductSpace):
        arg = sigma * prior * F.exp(x)
        func_cc_prox_x = x - space.apply(scipy.special.lambertw, arg).real
    else:
        func_cc_prox_x = x - scipy.special.lambertw(
            sigma * prior * F.exp(x)
        ).real
    _test_op(func_cc.proximal(sigma), x, func_cc_prox_x)

    # Biconjugate
    func_cc_cc = func_cc.convex_conj
    x = F.abs(noise_element(space)) + 0.1  # need positive again
    assert func_cc_cc(x) == pytest.approx(func(x))


def test_quadratic_form(space):
    """Test the quadratic form functional."""
    # TODO: move this to largescale tests
    if False:
        # Non-symmetric operator
        mat = np.eye(space.size)
        mat[0, 1] = 1
        operator = odl.MatrixOperator(mat, domain=space, range=space)

        mat_sym = (mat + mat.T) / 2
        mat_sym_inv = np.linalg.inv(mat_sym)
        sym_inv_op = odl.MatrixOperator(mat_sym_inv, domain=space, range=space)
        kwargs['operator_sym_inv'] = sym_inv_op

        def prox_inv_fact(sigma):
            minv = np.linalg.inv(np.eye(space.size) + sigma * mat_sym)
            return odl.MatrixOperator(minv, domain=space, range=space)

        kwargs['operator_prox_inv_fact'] = prox_inv_fact

    I = odl.IdentityOperator(space)
    vector = noise_element(space)
    constant = np.random.rand()

    def prox_inv_fact(sigma):
        return odl.ScalingOperator(space, 1 / (1 + sigma))

    x = noise_element(space)
    sigma = 1.2

    # Quadratic form with operator, vector and constant

    func = odl.solvers.QuadraticForm(
        space, operator=I, vector=vector, constant=constant,
        operator_sym_inv=I, operator_prox_inv_fact=prox_inv_fact
    )

    # Evaluation
    assert func(x) == pytest.approx(
        space.inner(x, x) + space.inner(x, vector) + constant
    )

    # Gradient
    func_grad_x = 2 * x + vector
    _test_op(func.gradient, x, func_grad_x)

    # Proximal
    func_prox_x = (x - sigma * vector) / (1 + 2 * sigma)
    _test_op(func.proximal(sigma), x, func_prox_x)

    # CC evaluation
    func_cc = func.convex_conj
    assert func_cc(x) == pytest.approx(
        space.inner(x - vector, x - vector) / 4 - constant
    )

    # CC gradient has nothing special

    # CC proximal
    # Same as above, but with vector -> -vector/2 and sigma -> sigma/2
    # in the denominator
    func_cc_prox_x = (x + sigma * vector / 2) / (1 + sigma / 2)
    _test_op(func_cc.proximal(sigma), x, func_cc_prox_x)

    # Quadratic form without operator, i.e., an affine functional

    func_affine = odl.solvers.QuadraticForm(
        space, vector=vector, constant=constant
    )
    # Evaluation
    assert func_affine(x) == pytest.approx(space.inner(x, vector) + constant)

    # Gradient
    func_affine_grad_x = vector
    _test_op(func_affine.gradient, x, func_affine_grad_x)

    # Proximal
    func_affine_prox_x = x - sigma * vector
    _test_op(func_affine.proximal(sigma), x, func_affine_prox_x)

    # CC evaluation
    # Translation of IndicatorZero by `vector` with offset `-constant`
    func_affine_cc = func_affine.convex_conj
    assert func_affine_cc(vector) == -constant
    assert func_affine_cc(vector + 1) == float('inf')

    # CC gradient not implemented

    # CC prox
    # projection onto the point set `{vector}`
    func_affine_cc_prox_x = vector
    _test_op(func_affine_cc.proximal(sigma), x, func_affine_cc_prox_x)

    # Quadratic form without vector

    func_no_vec = odl.solvers.QuadraticForm(
        space, I, constant=constant, operator_sym_inv=I,
        operator_prox_inv_fact=prox_inv_fact

    )

    # Evaluation
    assert func_no_vec(x) == pytest.approx(space.inner(x, x) + constant)

    # Gradient
    func_no_vec_grad_x = 2 * x
    _test_op(func_no_vec.gradient, x, func_no_vec_grad_x)

    # Proximal
    func_no_vec_prox_x = x / (1 + 2 * sigma)
    _test_op(func_no_vec.proximal(sigma), x, func_no_vec_prox_x)

    # CC evaluation
    func_no_vec_cc = func_no_vec.convex_conj
    assert func_no_vec_cc(x) == pytest.approx(space.inner(x, x) / 4 - constant)

    # CC gradient has nothing special

    # CC proximal
    # Same as above, but with sigma/2 in the denominator
    func_no_vec_cc_prox_x = x / (1 + sigma / 2)
    _test_op(func_no_vec_cc.proximal(sigma), x, func_no_vec_cc_prox_x)


def test_separable_sum(space):
    """Test for the separable sum."""
    l1 = odl.solvers.L1Norm(space)
    l2 = odl.solvers.L2Norm(space)

    x = noise_element(space)
    y = noise_element(space)

    # Evaluation
    func = odl.solvers.SeparableSum(l1, l2)
    assert func([x, y]) == pytest.approx(l1(x) + l2(y))
    power_func = odl.solvers.SeparableSum(l1, 5)
    assert power_func([x, x, x, x, x]) == pytest.approx(5 * l1(x))

    # Gradient
    func_grad_xy = [l1.gradient(x), l2.gradient(y)]
    _test_op(func.gradient, [x, y], func_grad_xy)

    # Proximal
    sigma = 1.2
    func_prox_xy = [l1.proximal(sigma)(x), l2.proximal(sigma)(y)]
    _test_op(func.proximal(sigma), [x, y], func_prox_xy)

    # CC evaluation
    assert func.convex_conj([x, y]) == pytest.approx(
        l1.convex_conj(x) + l2.convex_conj(y)
    )

    # CC is a SeparableSum of the convex conjugates, remaining test cases
    # are thus covered


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
    """Test for the Moreau envelope with squared L2 norm."""

    # Result is ||x||_2^2 / (1 + 2 * sigma)
    # Gradient is x * 2 / (1 + 2 * sigma)
    l2_sq = odl.solvers.L2NormSquared(space)

    smoothed_l2_sq = odl.solvers.MoreauEnvelope(l2_sq, sigma=sigma)
    x = noise_element(space)
    assert all_almost_equal(
        smoothed_l2_sq.gradient(x), x * 2 / (1 + 2 * sigma)
    )


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
    """Test for the weighted proximal of the squared L2 norm."""
    func = odl.solvers.L2NormSquared(space)
    sigma = odl.phantom.uniform_noise(space, 1, 10)
    x = space.one()

    # Check if the subdifferential inequalities are satisfied:
    # p = prox_{sigma * f}(x) iff (x - p)/sigma = grad f(p)
    prox = func.proximal(sigma)(x)
    assert all_almost_equal(
        func.gradient(prox), (x - prox) / sigma
    )

    prox_ip = space.element()
    func.proximal(sigma)(x, out=prox_ip)
    assert all_almost_equal(prox, prox_ip)


def test_weighted_proximal_L1_norm_far(space):
    """Test for the weighted proximal of the L1 norm away from zero."""
    func = odl.solvers.L1Norm(space)
    sigma = odl.phantom.noise.uniform_noise(space, 1, 10)
    x = 100 * space.one()  # no problem with differentiability

    # Check if the subdifferential inequalities are satisfied:
    # p = prox_{sigma * f}(x) iff (x - p)/sigma = grad f(p)
    prox = func.proximal(sigma)(x)
    assert all_almost_equal(
        func.gradient(prox), (x - prox) / sigma
    )

    prox_ip = space.element()
    func.proximal(sigma)(x, out=prox_ip)
    assert all_almost_equal(prox, prox_ip)


def test_weighted_proximal_L1_norm_close(space):
    """Test for the weighted proximal of the L1 norm near zero"""
    space = odl.rn(5)
    func = odl.solvers.L1Norm(space)
    sigma = [0.1, 0.2, 0.5, 1.0, 2.0]
    x = 0.5 * space.one()

    prox = func.proximal(sigma)(x)
    expected_result = [0.4, 0.3, 0.0, 0.0, 0.0]
    assert all_almost_equal(prox, expected_result)

    prox_ip = space.element()
    func.proximal(sigma)(x, out=prox_ip)
    assert all_almost_equal(prox, prox_ip)


def test_bregman_functional_no_gradient(space):
    """Test Bregman distance for functional without gradient."""
    F = space.ufuncs
    ind_func = odl.solvers.IndicatorNonnegativity(space)
    point = np.abs(noise_element(space))
    subgrad = noise_element(space)  # Any element in the domain is ok
    bregman_dist = odl.solvers.BregmanDistance(ind_func, point, subgrad)

    x = F.abs(noise_element(space))

    expected_result = space.inner(-subgrad, x - point)
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
