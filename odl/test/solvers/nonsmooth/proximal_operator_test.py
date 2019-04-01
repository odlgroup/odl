# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the factory functions to create proximal operators."""

from __future__ import division

import numpy as np
import scipy.special

import odl
from odl.solvers.nonsmooth.proximal_operators import (
    combine_proximals, proximal_box_constraint, proximal_const_func,
    proximal_convex_conj_kl, proximal_convex_conj_kl_cross_entropy,
    proximal_convex_conj_l1, proximal_convex_conj_l1_l2,
    proximal_convex_conj_l2_squared, proximal_l2)
from odl.util.testutils import all_almost_equal, simple_fixture

# Places for the accepted error when comparing results
HIGH_ACC = 8
LOW_ACC = 4


# --- pytest fixtures --- #

lower = simple_fixture('lower', [None, -2, 'elem'])
upper = simple_fixture('upper', [None, 2, 'elem'])
with_g = simple_fixture('with_g', [False, True])
sigma = simple_fixture('sigma', [0.5, 1.2, 'elem'])
scaling = simple_fixture('scaling', [2.5, 'elem'])


# --- Unittests --- #


def test_prox_const_func():
    """Test proximal of the constant functional."""
    space = odl.uniform_discr(0, 1, 10)
    x = space.element(np.arange(-5, 5))

    prox = proximal_const_func(space)(1.0)
    prox_x = prox(x)
    assert all_almost_equal(prox_x, x)


def test_prox_box_constraint(lower, upper):
    """Test proximal of the box constraint indicator."""
    space = odl.uniform_discr(0, 1, 10)
    x = space.element(np.arange(-5, 5))

    if lower == 'elem':
        lower = -2.0 * space.one()

    if upper == 'elem':
        upper = 2.0 * space.one()

    prox = proximal_box_constraint(space, lower, upper)(1.0)
    prox_x = prox(x)

    lower_np = -np.inf if lower is None else lower
    upper_np = np.inf if upper is None else upper
    true_prox_x = np.minimum(np.maximum(x, lower_np), upper_np)

    assert all_almost_equal(prox_x, true_prox_x)


def test_combine_proximals():
    """Test function to combine proximal factory functions.

    The combine function makes use of the separable sum property of proximal
    operators.
    """
    space = odl.uniform_discr(0, 1, 10)
    # Combination works at the level of factories
    prox_fact = proximal_const_func(space)
    combined_prox_fact = combine_proximals(prox_fact, prox_fact)

    prox = combined_prox_fact(1.0)

    I = odl.IdentityOperator(space)
    true_prox = odl.DiagonalOperator(I, I)
    x = true_prox.domain.element([np.arange(-5, 5), np.arange(-5, 5)])
    assert all_almost_equal(prox(x), true_prox(x))
    out = true_prox.range.element()
    prox(x, out=out)
    assert all_almost_equal(out, true_prox(x))


def test_prox_l2(with_g):
    """Test proximal of the L2 norm."""
    space = odl.uniform_discr(0, 1, 10)
    lam = 2.0
    sigma = 3.0

    if with_g:
        g = space.element(np.arange(-5, 5))
        prox = proximal_l2(space, lam, g)(sigma)
    else:
        g = space.zero()
        prox = proximal_l2(space, lam)(sigma)

    x = space.element(np.arange(-5, 5))
    x_norm = space.norm(x)
    x_small = g + x * 0.5 * lam * sigma / x_norm
    x_large = g + x * 2.0 * lam * sigma / x_norm

    true_prox_x_small = g
    const = lam * sigma / space.norm(x_large - g)
    true_prox_x_large = (1 - const) * x_large + const * g

    assert all_almost_equal(prox(x_small), true_prox_x_small, HIGH_ACC)
    assert all_almost_equal(prox(x_large), true_prox_x_large, HIGH_ACC)

    out = space.element()
    prox(x_large, out=out)
    assert all_almost_equal(out, true_prox_x_large, HIGH_ACC)
    prox(x_large, out=x_large)
    assert all_almost_equal(x_large, true_prox_x_large, HIGH_ACC)


def test_prox_cconj_l2_sq(with_g):
    """Test proximal of the squared L2 norm convex conjugate."""
    space = odl.uniform_discr(0, 10, 10)
    lam = 2
    sigma = 0.25
    sigma_elem = sigma * space.one()

    if with_g:
        g = space.element(-2 * np.arange(-5, 5))
        prox = proximal_convex_conj_l2_squared(space, lam, g)(sigma)
        prox_elem = proximal_convex_conj_l2_squared(space, lam, g)(sigma_elem)
    else:
        g = space.zero()
        prox = proximal_convex_conj_l2_squared(space, lam)(sigma)
        prox_elem = proximal_convex_conj_l2_squared(space, lam)(sigma_elem)

    x = space.element(np.arange(-5, 5))
    true_prox_x = (x - sigma * g) / (1 + sigma / (2 * lam))

    assert all_almost_equal(prox(x), true_prox_x, HIGH_ACC)
    out = space.element()
    prox(x, out=out)
    assert all_almost_equal(out, true_prox_x, HIGH_ACC)
    prox(x, out=x)
    assert all_almost_equal(x, true_prox_x, HIGH_ACC)

    x = space.element(np.arange(-5, 5))
    assert all_almost_equal(prox_elem(x), true_prox_x, HIGH_ACC)
    out = space.element()
    prox_elem(x, out=out)
    assert all_almost_equal(out, true_prox_x, HIGH_ACC)
    prox_elem(x, out=x)
    assert all_almost_equal(x, true_prox_x, HIGH_ACC)


def test_prox_conv_l1(with_g):
    """Test proximal of the L1 norm convex conjugate."""
    space = odl.uniform_discr(0, 1, 10)
    F = space.ufuncs
    lam = 2
    sigma = 0.25

    if with_g:
        g = space.element(np.arange(10, 0, -1))
        prox = proximal_convex_conj_l1(space, lam, g)(sigma)
    else:
        g = space.zero()
        prox = proximal_convex_conj_l1(space, lam)(sigma)

    x = space.element(np.arange(-5, 5))
    true_prox_x = lam * (x - sigma * g) / F.maximum(lam, F.abs(x - sigma * g))

    assert all_almost_equal(prox(x), true_prox_x, HIGH_ACC)
    out = space.element()
    prox(x, out=out)
    assert all_almost_equal(out, true_prox_x, HIGH_ACC)
    prox(x, out=x)
    assert all_almost_equal(x, true_prox_x, HIGH_ACC)


def test_prox_cconj_l1_l2():
    """Test proximal of the L1-L2 norm convex conjugate."""
    pspace = odl.ProductSpace(odl.uniform_discr(0, 1, 10), 2)
    Fb = pspace[0].ufuncs
    lam = 2
    sigma = 0.25

    x = pspace.element([np.arange(-5, 5), np.arange(10, 0, -1)])
    g = pspace.copy(x)[::-1]

    prox = proximal_convex_conj_l1_l2(pspace, lam, g)(sigma)

    # (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = Fb.maximum(lam, Fb.hypot(*(x - sigma * g)))
    true_prox_x = [lam * (xi - sigma * gi) / denom for xi, gi in zip(x, g)]

    assert all_almost_equal(prox(x), true_prox_x, HIGH_ACC)
    out = pspace.element()
    prox(x, out=out)
    assert all_almost_equal(out, true_prox_x, HIGH_ACC)
    prox(x, out=x)
    assert all_almost_equal(x, true_prox_x, HIGH_ACC)


def test_prox_cconj_kl():
    """Test proximal of the KL divergence convex conjugate."""
    space = odl.uniform_discr(0, 1, 10)
    F = space.ufuncs
    lam = 2
    sigma = 0.25
    g = space.element(np.arange(10, 0, -1))
    prox = proximal_convex_conj_kl(space, lam, g)(sigma)

    x = space.element(np.arange(-5, 5))
    true_prox_x = (lam + x - F.sqrt((x - lam) ** 2 + 4 * lam * sigma * g)) / 2

    assert all_almost_equal(prox(x), true_prox_x, HIGH_ACC)
    out = space.element()
    prox(x, out=out)
    assert all_almost_equal(out, true_prox_x, HIGH_ACC)
    prox(x, out=x)
    assert all_almost_equal(x, true_prox_x, HIGH_ACC)


def test_proximal_convconj_kl_product_space():
    """Test for product spaces in proximal for conjugate of KL divergence"""
    pspace = odl.ProductSpace(odl.uniform_discr(0, 1, 10), 2)
    F = pspace.ufuncs
    lam = 2
    sigma = 0.25

    x = pspace.element([np.arange(-5, 5), np.arange(10, 0, -1)])
    g = pspace.copy(x)[::-1]
    prox = proximal_convex_conj_kl(pspace, lam, g)(sigma)
    true_prox_x = (lam + x - F.sqrt((x - lam) ** 2 + 4 * lam * sigma * g)) / 2

    assert all_almost_equal(prox(x), true_prox_x, HIGH_ACC)
    out = pspace.element()
    prox(x, out=out)
    assert all_almost_equal(out, true_prox_x, HIGH_ACC)
    prox(x, out=x)
    assert all_almost_equal(x, true_prox_x, HIGH_ACC)


def test_proximal_convconj_kl_cross_entropy():
    """Test for proximal of convex conjugate of cross entropy KL divergence."""
    space = odl.uniform_discr(0, 1, 10)
    F = space.ufuncs
    lam = 2
    sigma = 0.25
    g = space.element(np.arange(10, 0, -1))
    prox = proximal_convex_conj_kl_cross_entropy(space, lam, g)(sigma)

    x = space.element(np.arange(-5, 5))
    true_prox_x = x - lam * scipy.special.lambertw(
        sigma / lam * g * F.exp(x / lam)
    ).real

    assert all_almost_equal(prox(x), true_prox_x, HIGH_ACC)
    out = space.element()
    prox(x, out=out)
    assert all_almost_equal(out, true_prox_x, HIGH_ACC)
    prox(x, out=x)
    assert all_almost_equal(x, true_prox_x, HIGH_ACC)


def test_proximal_arg_scaling(sigma, scaling):
    """Test for proximal argument scaling."""
    space = odl.uniform_discr(0, 1, 10)
    func = odl.solvers.L2NormSquared(space)

    if sigma == 'elem':
        sigma = odl.phantom.noise.uniform_noise(space, 1, 10)

    if scaling == 'elem':
        scaling = odl.phantom.noise.uniform_noise(space, 1, 10)

    # Scaling happens at the level of factories
    prox_fact = odl.solvers.proximal_l2_squared(space)
    prox_scal = odl.solvers.proximal_arg_scaling(prox_fact, scaling)(sigma)

    x = space.one()
    prox_x = prox_scal(x)

    # Check that p = Prox_{sigma g}(x) where g(x) = f(scaling * x),
    # i.e., (x - p)/sigma = grad g(p) = scaling * grad f(scaling p).
    lhs = (x - prox_x) / sigma
    rhs = scaling * func.gradient(scaling * prox_x)
    assert all_almost_equal(lhs, rhs)


if __name__ == '__main__':
    odl.util.test_file(__file__)
