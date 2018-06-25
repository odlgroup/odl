# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the utility functions for proximal operators."""

from __future__ import division
import numpy as np
import pytest

import odl
from odl.solvers.nonsmooth.proximal_operators import (
    proximal_arg_scaling, proximal_composition,
    proximal_quadratic_perturbation, proximal_translation,
    proximal_l2_squared)
from odl.util.testutils import all_almost_equal, noise_element, simple_fixture

# Number of digits for the accepted error when comparing results
NDIGITS = 8


# --- pytest fixtures --- #


scalar_params = [0.01, 2.7, np.array(5.0), 10, -2, -0.2, -np.array(7.1), 0]
scalar = simple_fixture('scalar', scalar_params)
nonneg_scalar = simple_fixture('nonneg_scalar',
                               [s for s in scalar_params if s >= 0])
pos_scalar = simple_fixture('pos_scalar',
                            [s for s in scalar_params if s > 0])
sigma = simple_fixture('sigma', [0.001, 2.7, np.array(0.5), 10])


# --- proximal utils tests --- #


def test_proximal_arg_scaling(scalar, sigma):
    """Test for the proximal of scaling: ``prox[F(. * a)]``."""
    sigma = float(sigma)
    space = odl.uniform_discr(0, 1, 10)
    lam = 1.2
    prox_factory = proximal_l2_squared(space, lam=lam)

    scaling_param = scalar
    prox = proximal_arg_scaling(prox_factory, scaling_param)(sigma)

    x = noise_element(space)
    # works for scaling_param == 0, too
    expected_result = x / (2 * sigma * lam * scaling_param ** 2 + 1)

    assert all_almost_equal(prox(x), expected_result, ndigits=NDIGITS)


def test_proximal_translation(sigma):
    """Test for the proximal of a translation: ``prox[F(. - g)]``."""
    sigma = float(sigma)
    space = odl.uniform_discr(0, 1, 10)
    lam = 1.2
    prox_factory = proximal_l2_squared(space, lam=lam)

    translation = noise_element(space)
    prox = proximal_translation(prox_factory, translation)(sigma)

    x = noise_element(space)
    expected_result = ((x + 2 * sigma * lam * translation) /
                       (1 + 2 * sigma * lam))

    assert all_almost_equal(prox(x), expected_result, ndigits=NDIGITS)


def test_proximal_quadratic_perturbation(nonneg_scalar, sigma):
    """Test for the proximal of quadratic perturbation."""
    sigma = float(sigma)
    space = odl.uniform_discr(0, 1, 10)
    lam = 1.2
    prox_factory = proximal_l2_squared(space, lam=lam)

    # parameter for the quadratic perturbation, needs to be non-negative
    a = nonneg_scalar

    # Test without linear term
    if a != 0:
        with pytest.raises(ValueError):
            # negative values not allowed
            proximal_quadratic_perturbation(prox_factory, -a)

    prox = proximal_quadratic_perturbation(prox_factory, a)(sigma)
    x = noise_element(space)
    expected_result = x / (2 * sigma * (lam + a) + 1)
    assert all_almost_equal(prox(x), expected_result, ndigits=NDIGITS)

    # Test with linear term
    u = noise_element(space)
    prox = proximal_quadratic_perturbation(prox_factory, a, u)(sigma)
    expected_result = (x - sigma * u) / (2 * sigma * (lam + a) + 1)
    assert all_almost_equal(prox(x), expected_result, ndigits=NDIGITS)


def test_proximal_composition(pos_scalar, sigma):
    """Test for proximal of composition with semi-orthogonal linear operator.

    This test is for ``prox[f * L](x)``, where ``L`` is a linear operator such
    that ``L * L.adjoint = mu * IdentityOperator``. Specifically, ``L`` is
    taken to be ``scal * IdentityOperator``, since this is equivalent to
    scaling of the argument.
    """
    sigma = float(sigma)
    space = odl.uniform_discr(0, 1, 10)
    prox_factory = proximal_l2_squared(space)

    # The semi-orthogonal linear operator
    scal = pos_scalar
    L = odl.ScalingOperator(space, scal)
    mu = scal ** 2  # L = scal * I => L * L.adjoint = scal ** 2 * I
    prox_factory_composition = proximal_composition(prox_factory, L, mu)
    prox = prox_factory_composition(sigma)

    assert isinstance(prox, odl.Operator)

    x = space.element(np.arange(-5, 5))
    prox_x = prox(x)
    equiv_prox = proximal_arg_scaling(prox_factory, scal)(sigma)
    expected_result = equiv_prox(x)
    assert all_almost_equal(prox_x, expected_result, ndigits=NDIGITS)


if __name__ == '__main__':
    odl.util.test_file(__file__)
