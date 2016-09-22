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

"""Tests for the utility functions for proximal operators."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
import odl.solvers as odls

from odl.util.testutils import all_almost_equal, noise_element

# Places for the accepted error when comparing results
PLACES = 8


def test_proximal_translation():
    """Test for the proximal of a translation: prox[F(. - g)]"""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Element in the image space where the proximal operator is evaluated
    translation = noise_element(space)

    # Factory function returning the proximal operators
    lam = float(np.random.randn(1))
    prox_factory = odls.proximal_l2_squared(space, lam=lam)

    # Initialize proximal operators
    step_size = float(np.random.rand(1))  # Non-negative step size
    prox = odls.proximal_translation(prox_factory, translation)(step_size)

    # Create an element in the space, in which to evaluate the proximals
    x = noise_element(space)

    # Explicit computation:
    expected_result = ((x + 2 * step_size * lam * translation) /
                       (1 + 2 * step_size * lam))

    assert all_almost_equal(prox(x), expected_result, places=PLACES)


def test_proximal_arg_scaling():
    """Test for the proximal of scaling: prox[F(. * a)]"""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Factory function returning the proximal operators
    lam = float(np.random.randn(1))
    prox_factory = odls.proximal_l2_squared(space, lam=lam)

    # Initialize proximal operators
    step_size = float(np.random.rand(1))   # Non-negative step size
    scaling_param = float(np.random.randn(1))
    prox = odls.proximal_arg_scaling(prox_factory, scaling_param)(step_size)

    # Create an element in the space, in which to evaluate the proximals
    x = noise_element(space)

    # Explicit computation:
    expected_result = x / (2 * step_size * lam * scaling_param ** 2 + 1)

    assert all_almost_equal(prox(x), expected_result, places=PLACES)


def test_proximal_arg_scaling_zero():
    """Test for the proximal of scaling: prox[F(. * a)] when a = 0"""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Factory function returning the proximal operators
    prox_factory = odls.proximal_l1(space, lam=1)

    # Initialize proximal operators
    scaling_param = 0.0
    step_size = float(np.random.rand(1))   # Non-negative step size
    prox = odls.proximal_arg_scaling(prox_factory, scaling_param)(step_size)

    # Create an element in the space, in which to evaluate the proximals
    x = noise_element(space)

    # Check that the scaling with zero returns proximal facotry for the
    # proximal_const_func, which results in the identity operator
    assert all_almost_equal(prox(x), x, places=PLACES)


def test_proximal_quadratic_perturbation_quadratic():
    """Test for the proximal of quadratic perturbation with quadratic term"""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # The parameter for the quadratic perturbation
    a = float(np.random.rand(1))  # This needs to be non-negative
    lam = float(np.random.randn(1))

    # Factory function returning the proximal operators
    prox_factory = odls.proximal_l2_squared(space, lam=lam)

    # Verify fail with negative a
    with pytest.raises(ValueError):
        odls.proximal_quadratic_perturbation(prox_factory, -a)

    # Initialize proximal operators
    step_size = float(np.random.rand(1))  # Non-negative step size
    prox = odls.proximal_quadratic_perturbation(prox_factory, a)(step_size)

    # Create an element in the space, in which to evaluate the proximals
    x = noise_element(space)

    # Explicit computation:
    expected_result = x / (2 * step_size * (lam + a) + 1)

    assert all_almost_equal(prox(x), expected_result, places=PLACES)


def test_proximal_quadratic_perturbation_linear_and_quadratic():
    """Test for the proximal of quadratic perturbation with both terms"""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # The parameter for the quadratic perturbation
    a = float(np.random.rand(1))  # This needs to be non-negative
    u = noise_element(space)
    lam = float(np.random.randn(1))

    # Factory function returning the proximal operators
    prox_factory = odls.proximal_l2_squared(space, lam=lam)

    # Initialize proximal operators
    step_size = float(np.random.rand(1))  # Non-negative step size
    prox = odls.proximal_quadratic_perturbation(prox_factory,
                                                a, u)(step_size)

    # Create an element in the space, in which to evaluate the proximals
    x = noise_element(space)

    # Explicit computation:
    expected_result = (x - step_size * u) / (2 * step_size * (lam + a) + 1)

    assert all_almost_equal(prox(x), expected_result, places=PLACES)


def test_proximal_compositio():
    """Test for proximal of composition with semi-orthogonal linear operator.

    This test is for ``prox[f * L](x)``, where ``L`` is a linear operator such
    that ``L * L.adjoint = mu * IdentityOperator``. Specifically, ``L`` is
    taken to be ``scal * IdentityOperator``, since this is equivalent to
    scaling of the argument.
    """
    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Factory function returning the proximal operator
    prox_factory = odls.proximal_l2_squared(space)

    # The semi-orthogonal linear operator
    scal = np.random.rand()
    L = odl.ScalingOperator(space, scal)
    mu = scal**2  # L = scal * I => L * L.adjoint = scal**2 * I

    # Initialize the proximal factory and operator for for F*L
    prox_factory_composition = odls.proximal_composition(prox_factory, L, mu)

    sigma = 0.25
    prox = prox_factory_composition(sigma)

    assert isinstance(prox, odl.Operator)

    # Element in image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    prox_val = prox(x)

    # Explicit computation:
    prox_verify = odls.proximal_arg_scaling(prox_factory, scal)(sigma)
    expected_result = prox_verify(x)

    assert all_almost_equal(prox_val, expected_result, places=PLACES)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
