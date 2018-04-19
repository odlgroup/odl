# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the factory functions to create proximal operators."""

from __future__ import division

import numpy as np
import pytest
import scipy.special

import odl
from odl.solvers.nonsmooth.proximal_operators import (
    proximal_separable_sum, proximal_box_constraint, proximal_const_func,
    proximal_convex_conj_kl, proximal_convex_conj_kl_cross_entropy,
    proximal_convex_conj_l1, proximal_convex_conj_l1_l2,
    proximal_convex_conj_l2_squared, proximal_l2)
from odl.util.testutils import all_almost_equal, noise_element

# Places for the accepted error when comparing results
HIGH_ACC = 8
LOW_ACC = 4


def test_proximal_const_func():
    """Proximal factory for the constnat mapping G(x) = c."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Element in the image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    # Factory function returning the proximal operator
    prox_factory = proximal_const_func(space)

    # Initialize proximal operator of G (with an unused parameter)
    prox = prox_factory(None)

    # prox_tau[G](x) = x = identity operator
    assert isinstance(prox, odl.IdentityOperator)

    # Optimal point of the auxiliary minimization problem prox_tau[G]
    x_opt = prox(x)

    # Identity map
    assert x == x_opt


def test_proximal_box_constraint():
    """Proximal factory for indicator function for non-negativity."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Element in the image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    for lower in [None, -2, -2.0 * space.one()]:
        for upper in [None, 2, 2.0 * space.one()]:
            # Factory function returning the proximal operator
            prox_factory = proximal_box_constraint(space,
                                                   lower=lower, upper=upper)
            prox = prox_factory(1.0)
            result = prox(x).asarray()

            # Create reference
            lower_np = -np.inf if lower is None else lower
            upper_np = np.inf if upper is None else upper
            result_np = np.minimum(np.maximum(x, lower_np), upper_np).asarray()

            # Verify equal result
            assert all_almost_equal(result_np, result)


def test_proximal_separable_sum():
    """Validate ``proximal_separable_sum``."""
    space = odl.uniform_discr(0, 1, 10)
    prox_factory = proximal_const_func(space)
    prox = prox_factory(1)

    sum_prox_factory = proximal_separable_sum(prox_factory, prox_factory)
    sum_prox = sum_prox_factory(1)

    assert isinstance(sum_prox, odl.Operator)

    x = noise_element(sum_prox.domain)
    out = sum_prox.range.element()
    true_result = x

    assert all_almost_equal(prox(x), true_result)
    prox(x, out=out)
    assert all_almost_equal(out, true_result)


def test_proximal_l2_wo_data():
    """Proximal factory for the L2-norm."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Factory function returning the proximal operator
    lam = 2.0
    prox_factory = proximal_l2(space, lam=lam)

    # Initialize the proximal operator
    sigma = 3.0
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Elements
    x = space.element(np.arange(-5, 5))
    x_small = x * 0.5 * lam * sigma / x.norm()
    x_big = x * 2.0 * lam * sigma / x.norm()

    # Explicit computation:
    x_small_opt = x_small * 0
    x_big_opt = (1 - lam * sigma / x_big.norm()) * x_big

    assert all_almost_equal(prox(x_small), x_small_opt, HIGH_ACC)
    assert all_almost_equal(prox(x_big), x_big_opt, HIGH_ACC)


def test_proximal_l2_with_data():
    """Proximal factory for the L2-norm with data term."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Create data
    g = space.element(np.arange(-5, 5))

    # Factory function returning the proximal operator
    lam = 2.0
    prox_factory = proximal_l2(space, lam=lam, g=g)

    # Initialize the proximal operator
    sigma = 3.0
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Elements
    x = space.element(np.arange(-5, 5))
    x_small = g + x * 0.5 * lam * sigma / x.norm()
    x_big = g + x * 2.0 * lam * sigma / x.norm()

    # Explicit computation:
    x_small_opt = g
    const = lam * sigma / (x_big - g).norm()
    x_big_opt = (1 - const) * x_big + const * g

    assert all_almost_equal(prox(x_small), x_small_opt, HIGH_ACC)
    assert all_almost_equal(prox(x_big), x_big_opt, HIGH_ACC)


def test_proximal_convconj_l2_sq_wo_data():
    """Proximal factory for the convex conjugate of the L2-norm."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)

    # Create an element in the image space
    x_arr = np.arange(-5, 5)
    x = space.element(x_arr)

    # Factory function returning the proximal operator
    lam = 2
    prox_factory = proximal_convex_conj_l2_squared(space, lam=lam)

    # Initialize the proximal operators
    sigma = 0.25 * space.one()
    sigmav = sigma * space.one()
    prox = prox_factory(sigma)
    proxv = prox_factory(sigmav)

    assert isinstance(prox, odl.Operator)
    assert isinstance(proxv, odl.Operator)

    # Allocate output elements
    x_out = space.element()
    x_outv = space.element()

    # Optimal point returned by the proximal operator
    prox(x, x_out)
    proxv(x, x_outv)

    # Explicit computation: x / (1 + sigma / (2 * lambda))
    x_verify = x / (1 + sigma / (2 * lam))

    assert all_almost_equal(x_out, x_verify, HIGH_ACC)
    assert all_almost_equal(x_outv, x_verify, HIGH_ACC)


def test_proximal_convconj_l2_sq_with_data():
    """Proximal factory for the convex conjugate of the L2-norm."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Create an element in the image space
    x_arr = np.arange(-5, 5)
    x = space.element(x_arr)

    # Create data
    g = space.element(-2 * x_arr)

    # Factory function returning the proximal operator
    lam = 2
    prox_factory = proximal_convex_conj_l2_squared(space, lam=lam, g=g)

    # Initialize the proximal operator
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate output element
    x_out = space.element()

    # Optimal point returned by the proximal operator
    prox(x, x_out)

    # Explicit computation: (x - sigma * g) / (1 + sigma / (2 * lambda))
    x_verify = (x - sigma * g) / (1 + sigma / (2 * lam))

    assert all_almost_equal(x_out, x_verify, HIGH_ACC)


def test_proximal_convconj_l1_simple_space_without_data():
    """Proximal factory for the convex conjugate of the L1-norm."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Image element
    x_arr = np.arange(-5, 5)
    x = space.element(x_arr)

    # Factory function returning the proximal operator
    lam = 2
    prox_factory = proximal_convex_conj_l1(space, lam=lam)

    # Initialize the proximal operator of F^*
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Apply the proximal operator returning its optimal point
    # Explicit computation: x / max(lam, |x|)
    denom = np.maximum(lam, np.sqrt(x_arr ** 2))
    x_exact = lam * x_arr / denom

    # Using out
    x_opt = space.element()
    x_result = prox(x, x_opt)
    assert x_result is x_opt
    assert all_almost_equal(x_opt, x_exact, HIGH_ACC)

    # Without out
    x_result = prox(x)
    assert all_almost_equal(x_result, x_exact, HIGH_ACC)

    # With aliased out
    x_result = prox(x, x)
    assert all_almost_equal(x_result, x_exact, HIGH_ACC)


def test_proximal_convconj_l1_simple_space_with_data():
    """Proximal factory for the convex conjugate of the L1-norm."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)
    x_arr = np.arange(-5, 5)
    x = space.element(x_arr)

    # RHS data
    g_arr = np.arange(10, 0, -1)
    g = space.element(g_arr)

    # Factory function returning the proximal operator
    lam = 2
    prox_factory = proximal_convex_conj_l1(space, lam=lam, g=g)

    # Initialize the proximal operator of F^*
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Apply the proximal operator returning its optimal point
    x_opt = space.element()
    prox(x, x_opt)

    # Explicit computation: (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = np.maximum(lam, np.abs(x_arr - sigma * g_arr))
    x0_verify = lam * (x_arr - sigma * g_arr) / denom

    assert all_almost_equal(x_opt, x0_verify, HIGH_ACC)


def test_proximal_convconj_l1_product_space():
    """Proximal factory for the convex conjugate of the L1-norm using
    product spaces."""

    # Product space for matrix of operators
    op_domain = odl.ProductSpace(odl.uniform_discr(0, 1, 10), 2)

    # Element in the product space where the proximal operator is evaluated
    x0_arr = np.arange(-5, 5)
    x1_arr = np.arange(10, 0, -1)
    x = op_domain.element([x0_arr, x1_arr])

    # Create a data element in the product space
    g0_arr = x1_arr.copy()
    g1_arr = x0_arr.copy()
    g = op_domain.element([g0_arr, g1_arr])

    # Factory function returning the proximal operator
    lam = 2
    prox_factory = proximal_convex_conj_l1_l2(op_domain, lam=lam, g=g)

    # Initialize the proximal operator
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Apply the proximal operator returning its optimal point
    x_opt = prox(x)

    # Explicit computation: (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = np.maximum(lam,
                       np.sqrt((x0_arr - sigma * g0_arr) ** 2 +
                               (x1_arr - sigma * g1_arr) ** 2))
    x_verify = lam * (x - sigma * g) / denom

    # Compare components
    assert all_almost_equal(x_verify, x_opt)


def test_proximal_convconj_kl_simple_space():
    """Test for proximal factory for the convex conjugate of KL divergence."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Element in image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    # Data
    g = space.element(np.arange(10, 0, -1))

    # Factory function returning the proximal operator
    lam = 2
    prox_factory = proximal_convex_conj_kl(space, lam=lam, g=g)

    # Initialize the proximal operator of F^*
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate an output element
    x_opt = space.element()

    # Apply the proximal operator returning its optimal point
    prox(x, x_opt)

    # Explicit computation:
    x_verify = (lam + x - np.sqrt((x - lam) ** 2 + 4 * lam * sigma * g)) / 2

    assert all_almost_equal(x_opt, x_verify, HIGH_ACC)


def test_proximal_convconj_kl_product_space():
    """Test for product spaces in proximal for conjugate of KL divergence"""

    # Product space for matrix of operators
    op_domain = odl.ProductSpace(odl.uniform_discr(0, 1, 10), 2)

    # Element in the product space where the proximal operator is evaluated
    x0_arr = np.arange(-5, 5)
    x1_arr = np.arange(10, 0, -1)
    x = op_domain.element([x0_arr, x1_arr])

    # Element in the product space with given data
    g0_arr = x1_arr.copy()
    g1_arr = x0_arr.copy()
    g = op_domain.element([g0_arr, g1_arr])

    # Factory function returning the proximal operator
    lam = 2
    prox_factory = proximal_convex_conj_kl(op_domain, lam=lam, g=g)

    # Initialize the proximal operator
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate an output element
    x_opt = op_domain.element()

    # Apply the proximal operator returning its optimal point
    prox(x, x_opt)

    # Explicit computation:
    x_verify = (lam + x - np.sqrt((x - lam) ** 2 + 4 * lam * sigma * g)) / 2

    # Compare components
    assert all_almost_equal(x_verify, x_opt)


def test_proximal_convconj_kl_cross_entropy():
    """Test for proximal of convex conjugate of cross entropy KL divergence."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Data
    g = space.element(np.arange(10, 0, -1))

    # Factory function returning the proximal operator
    lam = 2
    prox_factory = proximal_convex_conj_kl_cross_entropy(space, lam=lam, g=g)

    # Initialize the proximal operator of F^*
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Element in image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    prox_val = prox(x)

    # Explicit computation:
    x_verify = x - lam * scipy.special.lambertw(
        sigma / lam * g * np.exp(x / lam)).real

    assert all_almost_equal(prox_val, x_verify, HIGH_ACC)

    # Test in-place evaluation
    x_inplace = space.element()
    prox(x, out=x_inplace)

    assert all_almost_equal(x_inplace, x_verify, HIGH_ACC)


def test_proximal_arg_scaling():
    """Test for proximal argument scaling."""

    # Set the underlying space.
    space = odl.uniform_discr(0, 1, 10)

    # Set the functional and the prox factory.
    func = odl.solvers.L2NormSquared(space)
    prox_factory = odl.solvers.proximal_l2_squared(space)

    # Set the point where the proximal operator will be evaluated.
    x = space.one()

    # Set the scaling parameters.
    for alpha in [2, odl.phantom.noise.uniform_noise(space, 1, 10)]:
        # Scale the proximal factories
        prox_scaled = odl.solvers.proximal_arg_scaling(prox_factory, alpha)

        # Set the step size.
        for sigma in [2, odl.phantom.noise.uniform_noise(space, 1, 10)]:
            # Evaluation of the proximals
            p = prox_scaled(sigma)(x)

            # Now we know that p = Prox_{sigma g}(x) where g(x) = f(alpha x),
            # i.e., (x - p)/sigma = grad g(p) = alpha * grad f(alpha p).
            lhs = (x - p) / sigma
            rhs = alpha * func.gradient(alpha * p)
            assert all_almost_equal(lhs, rhs)


if __name__ == '__main__':
    odl.util.test_file(__file__)
