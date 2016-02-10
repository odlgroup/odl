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
    proximal_convexconjugate_l1, proximal_convexconjugate_l2)
from odl.util.testutils import all_almost_equal


# Places for the accepted error when comparing results
PLACES = 8


def test_proximal_zero():
    """Proximal factory for the zero mapping G(x) = 0."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x_data = x_space.element(np.arange(-5, 5))

    # Factory function for the proximal operator
    make_prox = proximal_zero(x_space)

    # Initialize proximal operator of G (with an unused parameter)
    prox_op = make_prox(None)

    # prox_tau[G](x) = x = identity operator
    assert isinstance(prox_op, odl.IdentityOperator)

    # Optimal point of the auxiliary minimization problem prox_tau[G]
    x_opt = prox_op(x_data)

    # Identity map
    assert x_data == x_opt


def test_proximal_nonnegativity():
    """Proximal factory for indicator function for non-negativity ."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x_data = x_space.element(np.arange(-5, 5))
    out = x_space.element()

    # Factory function for the proximal operator
    make_prox = proximal_nonnegativity(x_space)

    # Initialize proximal operator of G with sigma
    prox_op = make_prox(None)

    # Optimal point returned by the proximal operator
    prox_op(x_data, out)

    # prox_tau[G](x) = non-negativity thresholding
    assert all(out.asarray() >= 0)


def test_combine_proximal():
    """Function to combine proximal factory functions.

    The combine function makes use of the separable sum property of proximal
    operators."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)

    # Factory function for the proximal operator
    make_prox = proximal_zero(x_space)

    # Combine factory function of proximal operators
    prox_factory = combine_proximals([make_prox, make_prox])

    # Initialize combine proximal operator
    prox_op = prox_factory(1)

    assert isinstance(prox_op, odl.Operator)

    # Explicit construction of the combine proximal operator
    prox_op_verify = odl.ProductSpaceOperator(
        [[odl.IdentityOperator(x_space), None],
         [None, odl.IdentityOperator(x_space)]])

    # Create an element in the domain of the operator
    x = prox_op_verify.domain.element([np.arange(-5, 5), np.arange(-5, 5)])

    # Create an element in the range of the operator to store the result
    out = prox_op_verify.range.element()

    # Apply explicitly constructed and factory-function-combined proximal
    # operators
    assert prox_op(x) == prox_op_verify(x)

    # Test output argument
    assert prox_op(x, out) == prox_op_verify(x)

    # Identity mapping
    assert out == x


def test_proximal_factory_convconj_l2():
    """Proximal factory for the convex conjugate of the L2-norm."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)

    # Create an element in the image space
    x0 = np.arange(-5, 5)
    x_data = x_space.element(x0)

    # Create data
    g_data = x_space.element(-2 * x0)

    # Factory function for the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l2(x_space, lam=lam, g=g_data)

    # Initialize the proximal operator
    sigma = 0.5
    prox_op = make_prox(sigma)

    assert isinstance(prox_op, odl.Operator)

    # Optimal point returned by the proximal operator
    x_out = x_space.element()
    prox_op(x_data, x_out)

    # Explicit computation: (x - sigma * g) / (1 + sigma / lambda)
    x_verify = (x_data - sigma * g_data) / (1 + sigma / lam)

    assert all_almost_equal(x_out, x_verify, PLACES)


def test_proximal_factory_convconj_l1_simple_space():
    """Proximal factory for the convex conjugate of the L1-semi-norm."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x0 = np.arange(-5, 5)
    x_data = x_space.element(x0)

    # RHS data
    g0 = np.arange(10, 0, -1)
    g_data = x_space.element(g0)

    # Factory function for the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l1(x_space, lam=lam, g=g_data)

    # Initialize the proximal operator of F^*
    sigma = 0.5
    prox_op = make_prox(sigma)

    assert isinstance(prox_op, odl.Operator)

    # Apply the proximal operator returning its optimal point
    x_opt = x_space.element()
    prox_op(x_data, x_opt)

    # Explicit computation: (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = np.maximum(lam * np.ones(x0.shape),
                       np.sqrt((x0 - sigma * g0) ** 2))
    x0_verify = lam * (x0 - sigma * g0) / denom

    assert all_almost_equal(x_opt, x0_verify, PLACES)


def test_proximal_factory_convconj_l1_product_space():
    """Proximal factory for the convex conjugate of the L1-semi-norm using
    product spaces."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)

    # Product space for matrix of operators
    op_domain = odl.ProductSpace(x_space, 2)

    # Create and element in the product space
    x0 = np.arange(-5, 5)
    x1 = np.arange(10, 0, -1)
    x_data = op_domain.element([x0, x1])

    # Create a data element in the product space
    g0 = x1.copy()
    g1 = x0.copy()
    g_data = op_domain.element([g0, g1])

    # Factory function for the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l1(op_domain, lam=lam, g=g_data)

    # Initialize the proximal operator
    sigma = 0.5
    prox_op = make_prox(sigma)

    assert isinstance(prox_op, odl.Operator)

    # Apply the proximal operator returning its optimal point
    x_out = op_domain.element()
    prox_op(x_data, x_out)

    # Explicit computation: (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = np.maximum(
        lam * np.ones(x0.shape),
        np.sqrt((x0 - sigma * g0) ** 2 + (x1 - sigma * g1) ** 2))
    x0_verify = lam * (x0 - sigma * g0) / denom
    x1_verify = lam * (x1 - sigma * g1) / denom

    # Compare components
    assert all_almost_equal(x0_verify, x_out[0])
    assert all_almost_equal(x1_verify, x_out[1])


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
