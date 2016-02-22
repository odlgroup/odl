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

"""Tests for the factory functions to create proximal operators."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
from odl.solvers.advanced.proximal_operators import (
    combine_proximals, proximal_zero, proximal_nonnegativity,
    proximal_convexconjugate_l1, proximal_convexconjugate_l2,
    proximal_convexconjugate_kl)
from odl.util.testutils import all_almost_equal


# Places for the accepted error when comparing results
PLACES = 8


def test_proximal_zero():
    """Proximal factory for the zero mapping G(x) = 0."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)

    # Element in the image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    # Factory function returning the proximal operator
    make_prox = proximal_zero(space)

    # Initialize proximal operator of G (with an unused parameter)
    prox = make_prox(None)

    # prox_tau[G](x) = x = identity operator
    assert isinstance(prox, odl.IdentityOperator)

    # Optimal point of the auxiliary minimization problem prox_tau[G]
    x_opt = prox(x)

    # Identity map
    assert x == x_opt


def test_proximal_nonnegativity():
    """Proximal factory for indicator function for non-negativity ."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)

    # Element in the image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    # Allocate output vector
    out = space.element()

    # Factory function returning the proximal operator
    make_prox = proximal_nonnegativity(space)

    # Initialize proximal operator of G with sigma
    prox = make_prox(None)

    # Optimal point returned by the proximal operator
    prox(x, out)

    # prox_tau[G](x) = non-negativity thresholding
    assert all(out.asarray() >= 0)


def test_combine_proximal():
    """Function to combine proximal factory functions.

    The combine function makes use of the separable sum property of proximal
    operators."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)

    # Factory function returning the proximal operator
    make_prox = proximal_zero(space)

    # Combine factory function of proximal operators
    combined_make_prox = combine_proximals([make_prox, make_prox])

    # Initialize combine proximal operator
    prox = combined_make_prox(1)

    assert isinstance(prox, odl.Operator)

    # Explicit construction of the combine proximal operator
    prox_verify = odl.ProductSpaceOperator(
        [[odl.IdentityOperator(space), None],
         [None, odl.IdentityOperator(space)]])

    # Create an element in the domain of the operator
    x = prox_verify.domain.element([np.arange(-5, 5), np.arange(-5, 5)])

    # Allocate output vector
    out = prox_verify.range.element()

    # Apply explicitly constructed and factory-function-combined proximal
    # operators
    assert prox(x) == prox_verify(x)

    # Test output argument
    assert prox(x, out) == prox_verify(x)

    # Identity mapping
    assert out == x


def test_proximal_factory_convconj_l2_wo_data():
    """Proximal factory for the convex conjugate of the L2-norm."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)

    # Element in the image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    # Factory function returning the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l2(space, lam=lam)

    # Initialize the proximal operator
    sigma = 0.5
    prox = make_prox(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate output vector
    x_opt = space.element()

    # Optimal point returned by the proximal operator
    prox(x, x_opt)

    # Explicit computation: (x - sigma * g) / (1 + sigma / lambda)
    x_verify = x / (1 + sigma / lam)

    assert all_almost_equal(x_opt, x_verify, PLACES)


def test_proximal_factory_convconj_l2_with_data():
    """Proximal factory for the convex conjugate of the L2-norm."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)

    # Create an element in the image space
    x_arr = np.arange(-5, 5)
    x = space.element(x_arr)

    # Create data
    g = space.element(-2 * x_arr)

    # Factory function returning the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l2(space, lam=lam, g=g)

    # Initialize the proximal operator
    sigma = 0.5
    prox = make_prox(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate output vector
    x_out = space.element()

    # Optimal point returned by the proximal operator
    prox(x, x_out)

    # Explicit computation: (x - sigma * g) / (1 + sigma / lambda)
    x_verify = (x - sigma * g) / (1 + sigma / lam)

    assert all_almost_equal(x_out, x_verify, PLACES)


def test_proximal_factory_convconj_l1_simple_space_without_data():
    """Proximal factory for the convex conjugate of the L1-semi-norm."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)

    # Image vector
    x_arr = np.arange(-5, 5)
    x = space.element(x_arr)

    # Factory function returning the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l1(space, lam=lam)

    # Initialize the proximal operator of F^*
    sigma = 0.5
    prox = make_prox(sigma)

    assert isinstance(prox, odl.Operator)

    # Apply the proximal operator returning its optimal point
    x_opt = space.element()
    prox(x, x_opt)

    # Explicit computation: x / max(lam, |x|)
    denom = np.maximum(lam * np.ones(x_arr.shape), np.sqrt(x_arr ** 2))
    x_verify = lam * x_arr / denom

    assert all_almost_equal(x_opt, x_verify, PLACES)


def test_proximal_factory_convconj_l1_simple_space_with_data():
    """Proximal factory for the convex conjugate of the L1-semi-norm."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)
    x_arr = np.arange(-5, 5)
    x = space.element(x_arr)

    # RHS data
    g_arr = np.arange(10, 0, -1)
    g = space.element(g_arr)

    # Factory function returning the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l1(space, lam=lam, g=g)

    # Initialize the proximal operator of F^*
    sigma = 0.5
    prox = make_prox(sigma)

    assert isinstance(prox, odl.Operator)

    # Apply the proximal operator returning its optimal point
    x_opt = space.element()
    prox(x, x_opt)

    # Explicit computation: (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = np.maximum(lam * np.ones(x_arr.shape),
                       np.sqrt((x_arr - sigma * g_arr) ** 2))
    x0_verify = lam * (x_arr - sigma * g_arr) / denom

    assert all_almost_equal(x_opt, x0_verify, PLACES)


def test_proximal_factory_convconj_l1_product_space():
    """Proximal factory for the convex conjugate of the L1-semi-norm using
    product spaces."""

    # Product space for matrix of operators
    op_domain = odl.ProductSpace(odl.uniform_discr(0, 10, 10), 2)

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
    make_prox = proximal_convexconjugate_l1(op_domain, lam=lam, g=g)

    # Initialize the proximal operator
    sigma = 0.5
    prox = make_prox(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate output vector
    x_opt = op_domain.element()

    # Apply the proximal operator returning its optimal point
    prox(x, x_opt)

    # Explicit computation: (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = np.maximum(
        lam * np.ones(x0_arr.shape),
        np.sqrt((x0_arr - sigma * g0_arr) ** 2 +
                (x1_arr - sigma * g1_arr) ** 2))
    x0_verify = lam * (x0_arr - sigma * g0_arr) / denom
    x1_verify = lam * (x1_arr - sigma * g1_arr) / denom

    # Compare components
    assert all_almost_equal(x0_verify, x_opt[0])
    assert all_almost_equal(x1_verify, x_opt[1])


def test_proximal_factory_convconj_kl_simple_space():
    """Proximal factory for the convex conjugate of KL divergence."""

    # Image space
    space = odl.uniform_discr(0, 10, 10)

    # Element in image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    # Data
    g = space.element(np.arange(10, 0, -1))

    # Factory function returning the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_kl(space, lam=lam, g=g)

    # Initialize the proximal operator of F^*
    sigma = 0.5
    prox = make_prox(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate an output vector
    x_opt = space.element()

    # Apply the proximal operator returning its optimal point
    prox(x, x_opt)

    # Explicit computation:
    # 1 / 2 * (lam_X + x - sqrt((x - lam_X)^2 + 4 * lam * sigma * g)
    x_verify = (lam + x - np.sqrt((x - lam) ** 2 + 4 * lam * sigma * g)) / 2

    assert all_almost_equal(x_opt, x_verify, PLACES)


def test_proximal_factory_convconj_kl_product_space():
    """Proximal factory for the convex conjugate of the KL divergence using
    product spaces."""

    # Product space for matrix of operators
    op_domain = odl.ProductSpace(odl.uniform_discr(0, 10, 10), 2)

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
    make_prox = proximal_convexconjugate_kl(op_domain, lam=lam, g=g)

    # Initialize the proximal operator
    sigma = 0.5
    prox = make_prox(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate an output vector
    x_opt = op_domain.element()

    # Apply the proximal operator returning its optimal point
    prox(x, x_opt)

    # Explicit computation:
    # 1 / 2 * (lam_X + x - sqrt((x - lam_X)^2 + 4 * lam * sigma * g)
    x0_verify = (lam + x0_arr - np.sqrt((x0_arr - lam) ** 2 + 4 * lam *
                                        sigma * g0_arr)) / 2
    x1_verify = (lam + x1_arr - np.sqrt((x1_arr - lam) ** 2 + 4 * lam *
                                        sigma * g1_arr)) / 2

    # Compare components
    assert all_almost_equal(x0_verify, x_opt[0])
    assert all_almost_equal(x1_verify, x_opt[1])


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
