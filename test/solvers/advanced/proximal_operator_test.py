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

"""Test for the Chambolle-Pock solver."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
from odl.solvers.advanced.proximal_operators import (combine_proximals,
    proximal_zero, proximal_nonnegativity, proximal_convexconjugate_l1,
    proximal_convexconjugate_l2)
from odl.util.testutils import all_almost_equal, all_equal

# TODO: updated doc

def test_proximal_zero():
    """Proximal factory for G(x) = 0."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x_data = x_space.element(np.arange(-5, 5))

    # Factory function for the proximal operator
    make_prox = proximal_zero(x_space)

    # Initialize proximal operator of G with parameter 1
    prox_op = make_prox(1)

    # prox_tau[G](x) = x = identity operator
    assert isinstance(prox_op, odl.IdentityOperator)

    # Optimal point of the auxiliary minimization problem prox_tau[G]
    x_opt = prox_op(x_data)

    # Identity map
    assert x_data == x_opt


def test_proximal_nonnegativity():
    """Proximal factory for G(x) = ind_P(x)."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x_data = x_space.element(np.arange(-5, 5))
    out = x_space.element()

    # Factory function for the proximal operator
    make_prox = proximal_nonnegativity(x_space)

    # Initialize proximal operator of G with sigma
    prox_op = make_prox(0)

    # Optimal point returned by the proximal operator
    prox_op(x_data, out)

    # prox_tau[G](x) = non-negativity thresholding
    assert all(out.asarray() >= 0)


def test_combine_proximal():
    """Function to combine proximal factory functions.

    The combine function make use of the separable sum property of proximal
    operators."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)

    # Factory function for the proximal operator
    make_prox = proximal_zero(x_space)

    # Combine factory function of proximal operators
    prox_factory = combine_proximals([make_prox, make_prox])

    # Initialize combine proximal operator
    prox_op  = prox_factory(1)

    # Explicit construction of the combine proximal operator
    prox_op_verify = odl.ProductSpaceOperator(
            [[odl.IdentityOperator(x_space), None],
            [None, odl.IdentityOperator(x_space)]])

    # Create an element in the domain of the operator
    x = prox_op_verify.domain.element([np.arange(-5, 5), np.arange(-5, 5)])

    # Create an element in the range of the operator to store the result
    out = prox_op_verify.range.element()

    # Apply explicitly contructed and factory-function-combined proximal
    # operators
    assert prox_op(x) == prox_op_verify(x)

    # Test output argument
    assert prox_op(x, out) == prox_op_verify(x)

    assert out == x

PRECISION = 8

# TODO: improve test documentation
def test_proximal_factory_l2():
    """Test of factory functions creating the proximal operator instances."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x0 = np.arange(-5, 5)
    x_data = x_space.element(x0)
    g = x_space.element(-2 * x0)

    # Factory function for the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l2(x_space, lam=lam, g=g)

    # Proximal operator of F^*
    sigma = 0.5
    prox_op = make_prox(sigma)

    # Optimal point returned by the proximal operator
    x_opt = x_space.element()
    prox_op(x_data, x_opt)

    # Explicit computation: (x - sigma * g) / (1 + sigma / lambda)
    x_verify = (x_data - sigma * g) / (1 + sigma / lam)

    assert all_almost_equal(x_opt, x_verify, PRECISION)


# TODO: improve test documentation
def test_proximal_factory_l1_simple_space():
    """Test of factory functions creating the proximal operator instances."""

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x0 = np.arange(-5, 5)
    x_data = x_space.element(x0)

    # RHS data
    g0 = np.arange(10, 0, -1)
    g = x_space.element(g0)

    # Factory function for the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l1(x_space, lam=lam, g=g)

    # Proximal operator of F^*
    sigma = 0.5
    prox_op = make_prox(sigma)

    # Optimal point returned by the proximal operator
    x_opt = x_space.element()
    prox_op(x_data, x_opt)

    # Explicit computation: (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = np.maximum(lam * np.ones(x0.shape),
                       np.sqrt((x0 - sigma * g0) ** 2))
    x0_verify = lam * (x0 - sigma * g0) / denom

    assert all_almost_equal(x_opt, x0_verify, PRECISION)


def test_proximal_factory_l1_product_space():

    # Image space
    x_space = odl.uniform_discr(0, 10, 10)
    x0 = np.arange(-5, 5)
    x1 = np.arange(10, 0, -1)

    # Product space for matrix of operators
    op_domain = odl.ProductSpace(x_space, 2)

    # Product space element
    x_data = op_domain.element([x0, x1])

    # RHS data
    g0 = x1.copy()
    g1 = x0.copy()
    g = op_domain.element([g0, g1])

    # Factory function for the proximal operator
    lam = 2
    make_prox = proximal_convexconjugate_l1(op_domain, lam=lam, g=g)

    # Proximal operator of F^*
    sigma = 0.5
    prox_op = make_prox(sigma)

    # Optimal point returned by the proximal operator
    x_opt = op_domain.element()
    prox_op(x_data, x_opt)

    # Explicit computation: (x - sigma * g) / max(lam, |x - sigma * g|)
    denom = np.maximum(
        lam * np.ones(x0.shape),
        np.sqrt((x0 - sigma * g0) ** 2 + (x1 - sigma * g1) ** 2))
    x0_verify = lam * (x0 - sigma * g0) / denom
    x1_verify = lam * (x1 - sigma * g1) / denom

    # Compare components
    assert all_almost_equal(x0_verify, x_opt[0])
    assert all_almost_equal(x1_verify, x_opt[1])


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
