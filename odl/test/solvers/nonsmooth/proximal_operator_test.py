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
import scipy.special

# Internal
import odl
from odl.solvers.nonsmooth.proximal_operators import (
    combine_proximals, proximal_const_func,
    proximal_box_constraint, proximal_nonnegativity,
    proximal_cconj_l1,
    proximal_l2,
    proximal_cconj_l2_squared,
    proximal_cconj_kl, proximal_cconj_kl_cross_entropy)
from odl.util.testutils import all_almost_equal


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


def test_proximal_nonnegativity():
    """Proximal factory for indicator function for non-negativity."""

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Element in the image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    # Factory function returning the proximal operator
    prox_factory = proximal_nonnegativity(space)

    # Initialize proximal operator of G (with an unused parameter)
    prox = prox_factory(1.0)

    # Optimal point returned by the proximal operator
    result = prox(x)

    # prox_tau[G](x) = non-negativity thresholding
    assert all(result.asarray() >= 0)


def test_combine_proximal():
    """Function to combine proximal factory functions.

    The combine function makes use of the separable sum property of proximal
    operators.
    """

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Factory function returning the proximal operator
    prox_factory = proximal_const_func(space)

    # Combine factory function of proximal operators
    combined_prox_factory = combine_proximals(prox_factory, prox_factory)

    # Initialize combine proximal operator
    prox = combined_prox_factory(1)

    assert isinstance(prox, odl.Operator)

    # Explicit construction of the combine proximal operator
    prox_verify = odl.ProductSpaceOperator(
        [[odl.IdentityOperator(space), None],
         [None, odl.IdentityOperator(space)]])

    # Create an element in the domain of the operator
    x = prox_verify.domain.element([np.arange(-5, 5), np.arange(-5, 5)])

    # Allocate output element
    out = prox_verify.range.element()

    # Apply explicitly constructed and factory-function-combined proximal
    # operators
    assert prox(x) == prox_verify(x)

    # Test output argument
    assert prox(x, out) == prox_verify(x)

    # Identity mapping
    assert out == x


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
    prox_factory = proximal_cconj_l2_squared(space, lam=lam)

    # Initialize the proximal operator
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Allocate output element
    x_out = space.element()

    # Optimal point returned by the proximal operator
    prox(x, x_out)

    # Explicit computation: x / (1 + sigma / (2 * lambda))
    x_verify = x / (1 + sigma / (2 * lam))

    assert all_almost_equal(x_out, x_verify, HIGH_ACC)


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
    prox_factory = proximal_cconj_l2_squared(space, lam=lam, g=g)

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
    prox_factory = proximal_cconj_l1(space, lam=lam)

    # Initialize the proximal operator of F^*
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Apply the proximal operator returning its optimal point
    x_opt = space.element()
    prox(x, x_opt)

    # Explicit computation: x / max(lam, |x|)
    denom = np.maximum(lam, np.sqrt(x_arr ** 2))
    x_verify = lam * x_arr / denom

    assert all_almost_equal(x_opt, x_verify, HIGH_ACC)


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
    prox_factory = proximal_cconj_l1(space, lam=lam, g=g)

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
    prox_factory = proximal_cconj_l1(op_domain, lam=lam, g=g, isotropic=True)

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
    prox_factory = proximal_cconj_kl(space, lam=lam, g=g)

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
    prox_factory = proximal_cconj_kl(op_domain, lam=lam, g=g)

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
    prox_factory = proximal_cconj_kl_cross_entropy(space, lam=lam, g=g)

    # Initialize the proximal operator of F^*
    sigma = 0.25
    prox = prox_factory(sigma)

    assert isinstance(prox, odl.Operator)

    # Element in image space where the proximal operator is evaluated
    x = space.element(np.arange(-5, 5))

    prox_val = prox(x)

    # Explicit computation:
    x_verify = x - lam * scipy.special.lambertw(
        sigma / lam * g * np.exp(x / lam))

    assert all_almost_equal(prox_val, x_verify, HIGH_ACC)

    # Test in-place evaluation
    x_inplace = space.element()
    prox(x, out=x_inplace)

    assert all_almost_equal(x_inplace, x_verify, HIGH_ACC)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
