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

"""Test for the Functional class."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
from odl.util.testutils import all_almost_equal

# Places for the accepted error when comparing results
PLACES = 8

# Discretization parameters
n = 3

# Discretized spaces
space = odl.uniform_discr([0, 0], [1, 1], [n, n])

# phantom = odl.util.shepp_logan(space, modified=True)*5+1

# LogPhantom=np.log(phantom)


# l1func = L1Norm(space)
# l1prox = l1func.proximal(sigma=1.5)
# l1conjFun = l1func.conjugate_functional


def test_derivative():
    # Verify that the derivative does indeed work as expected

    x = space.element(np.random.standard_normal((n, n)))

    y = space.element(np.random.standard_normal((n, n)))
    epsK = 1e-8

    F = odl.solvers.functional.L2Norm(space)

    # Numerical test of gradient
    assert all_almost_equal((F(x+epsK*y)-F(x))/epsK,
                            y.inner(F.gradient(x)),
                            places=PLACES/2)

    # Check that derivative and gradient is consistent
    assert all_almost_equal(F.derivative(x)(y),
                            y.inner(F.gradient(x)),
                            places=PLACES)


def test_scalar_multiplication_call():
    # Verify that the left and right scalar multiplication does
    # indeed work as expected

    x = space.element(np.random.standard_normal((n, n)))

    scal = np.random.standard_normal()
    F = odl.solvers.functional.L2Norm(space)

    # Evaluation of right and left scalar multiplication
    assert all_almost_equal((F*scal)(x), (F)(scal*x),
                            places=PLACES)

    assert all_almost_equal((scal*F)(x), scal*(F(x)),
                            places=PLACES)

    # Test gradient of right and left scalar multiplication
    assert all_almost_equal((scal*F).gradient(x), scal*(F.gradient(x)),
                            places=PLACES)

    assert all_almost_equal((F*scal).gradient(x), scal*(F.gradient(scal*x)),
                            places=PLACES)


def test_scalar_multiplication_conjugate_functional():
    # Verify that conjugate functional of right and left scalar multiplication
    # work as intended

    x = space.element(np.random.standard_normal((n, n)))

    scal = np.abs(np.random.standard_normal())

    F = odl.solvers.functional.L2Norm(space)

    assert all_almost_equal((scal*F).conjugate_functional(x),
                            scal*(F.conjugate_functional(x/scal)),
                            places=PLACES)

    assert all_almost_equal((F*scal).conjugate_functional(x),
                            (F.conjugate_functional(x/scal)),
                            places=PLACES)


# TODO: test prox functinoality for scaling

# def test_prox:
    # Verify that the left and right scalar multiplication does indeed work as expected

#    x = space.element(np.random.standard_normal((n,n)))

#    scal=np.random.standard_normal()
#    F=odl.solvers.functional.L1Norm(space)

    #make some tests that check that prox work.

    #assert all_almost_equal((F*scal)(x), (F)(scal*x),
    #                        places=PLACES)

    #assert all_almost_equal((scal*F)(x), scal*(F(x)),
    #                        places=PLACES)

# TODO: implement translation for prox and conjugate functionals + tests

# TODO: Test that prox and conjugate functionals are not returned for negative left scaling.

# TODO: Move tests from convex_conjugate_utils_test to here!!!

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
