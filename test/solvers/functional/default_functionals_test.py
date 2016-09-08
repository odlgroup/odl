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

"""Test for the default functionals."""

# External
import numpy as np
import pytest

# Internal
import odl
from odl.util.testutils import all_almost_equal, almost_equal, noise_element

# Places for the accepted error when comparing results
PLACES = 8


def test_L1_norm():
    """Test the L1-norm."""

    n = 10
    space = odl.rn(n)
    func = odl.solvers.L1Norm(space)
    x = noise_element(space)

    # Evaluation of the functional
    expected_result = np.sum(np.abs(x))
    assert almost_equal(func(x), expected_result, places=PLACES)

    # The gradient is the sign-function
    expected_result = np.sign(x)
    assert all_almost_equal(func.gradient(x), expected_result,
                            places=PLACES)

    # The proximal operator
    sigma = np.random.rand()
    # Explicit computation:      |  x_i - sigma, x_i > sigma
    #                      z_i = {  0,           -sigma <= x_i <= sigma
    #                            |  x_i + sigma,     x_i < -sigma
    tmp = space.zero()
    orig = x.asarray()
    tmp[orig > sigma] = orig[orig > sigma] - sigma
    tmp[orig < -sigma] = orig[orig < -sigma] + sigma
    expected_result = space.element(tmp)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result,
                            places=PLACES)

    # The convex conjugate functional
    cc_func = func.convex_conj

    # Evaluation of convex conjugate
    # Explicit calculation: 0 if |x|_1 < 1, infty else
    norm_larger_than_one = 2.0 * space.one()
    assert cc_func(norm_larger_than_one) == np.inf

    norm_less_than_one = 0.9 * 1.0 / n * space.one()
    assert cc_func(norm_less_than_one) == 0

    # The gradient of the convex conjugate (not implemeted)
    with pytest.raises(NotImplementedError):
        cc_func.gradient

    # The proximal of the convex conjugate
    # Explicit computation: x / max(1, |x|)
    expected_result = x / np.max((space.one(), np.abs(x)), axis=0)
    assert all_almost_equal(cc_func.proximal(sigma)(x), expected_result,
                            places=PLACES)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    expected_functional = odl.solvers.L1Norm(space)
    cc_cc_func = cc_func.convex_conj
    assert isinstance(cc_cc_func, type(expected_functional))


def test_L2_norm():
    """Test the L2-norm."""

    n = 10
    space = odl.rn(n)
    func = odl.solvers.L2Norm(space)
    x = noise_element(space)

    # Evaluation of the functional
    expected_result = np.sqrt(np.sum(x**2))
    assert almost_equal(func(x), expected_result, places=PLACES)

    # The gradient
    if x.norm() == 0:
        # In this case the gradient is not defined
        with pytest.raises(ValueError):
            func.gradient
    else:
        expected_result = x / x.norm()
        assert all_almost_equal(func.gradient(x), expected_result,
                                places=PLACES)

    # The proximal operator
    sigma = np.random.rand()
    # Explicit computation: x * (1 - sigma/||x||) if ||x|| > 1, 0 else
    norm_less_than_sigma = 0.9 * sigma / n * space.one()
    assert all_almost_equal(func.proximal(sigma)(norm_less_than_sigma),
                            space.zero(), places=PLACES)

    norm_larger_than_sigma = 1.1 * x.norm() * x
    expected_result = (norm_larger_than_sigma *
                       (1.0 - sigma / norm_larger_than_sigma.norm()))
    assert all_almost_equal(func.proximal(sigma)(norm_larger_than_sigma),
                            expected_result, places=PLACES)

    # The convex conjugate functional
    cc_func = func.convex_conj

    # Evaluation of convex conjugate
    # Explicit calculation: 0 if ||x|| < 1, infty else
    norm_larger_than_one = 2.0 * space.one()
    assert cc_func(norm_larger_than_one) == np.inf

    norm_less_than_one = 0.9 * 1.0 / n * space.one()
    assert cc_func(norm_less_than_one) == 0

    # The gradient of the convex conjugate (not implemeted)
    with pytest.raises(NotImplementedError):
        cc_func.gradient

    # The proximal of the convex conjugate
    # Explicit computation: x/||x||
    expected_result = x / x.norm()
    assert all_almost_equal(cc_func.proximal(sigma)(x), expected_result,
                            places=PLACES)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    expected_functional = odl.solvers.L2Norm(space)
    cc_cc_func = cc_func.convex_conj
    assert isinstance(cc_cc_func, type(expected_functional))


def test_L2_norm_squared():
    """Test the squared L2-norm."""

    n = 10
    space = odl.rn(n)
    func = odl.solvers.L2NormSquared(space)
    x = noise_element(space)

    # Evaluation of the functional
    expected_result = np.sum(x**2)
    assert almost_equal(func(x), expected_result, places=PLACES)

    # The gradient
    expected_result = 2.0 * x
    assert all_almost_equal(func.gradient(x), expected_result,
                            places=PLACES)

    # The proximal operator
    sigma = np.random.rand()
    # Explicit computation: x / (1 + 2*sigma)
    expected_result = x / (1 + 2.0 * sigma)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result,
                            places=PLACES)

    # The convex conjugate functional
    cc_func = func.convex_conj

    # Evaluation of convex conjugate
    # Explicit calculation: ||x||^2 / 4
    expected_result = x.norm()**2 / 4.0
    assert almost_equal(cc_func(x), expected_result, places=PLACES)

    # The gradient of the convex conjugate (not implemeted)
    # Explicit calculation: x / 2
    expected_result = x / 2.0
    assert all_almost_equal(cc_func.gradient(x), expected_result,
                            places=PLACES)

    # The proximal of the convex conjugate
    # Explicit computation: x / (1 + sigma/2)
    expected_result = x / (1 + sigma / 2.0)
    assert all_almost_equal(cc_func.proximal(sigma)(x), expected_result,
                            places=PLACES)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    cc_cc_func = cc_func.convex_conj

    # Modulo scaling back and forth, check that it is in fact squared L2-norm
    assert isinstance(cc_cc_func.orig_cconj_f.operator,
                      odl.solvers.L2NormSquared)

    # Check that they evaluate the same
    assert almost_equal(cc_cc_func(x), func(x), places=PLACES)

    # Check that they evaluate the gradient the same
    assert all_almost_equal(cc_cc_func.gradient(x), func.gradient(x),
                            places=PLACES)


def test_constant_functional():
    """Test the constant functional."""

    n = 10
    space = odl.rn(n)
    constant = np.random.randn()
    func = odl.solvers.ConstantFunctional(space, constant=constant)
    x = noise_element(space)

    # Checking that constant is stored correctly
    assert func.constant == constant

    # Evaluation of the functional
    expected_result = constant
    assert almost_equal(func(x), expected_result, places=PLACES)

    # The gradient
    # Given by the zero-operator
    assert isinstance(func.gradient, odl.ZeroOperator)

    # The proximal operator
    sigma = np.random.rand()
    # This is the identity operator
    assert isinstance(func.proximal(sigma), odl.IdentityOperator)

    # The convex conjugate functional
    cc_func = func.convex_conj

    # Evaluation of convex conjugate
    # Explicit calculation: -constant if x=0, infty else
    if not x.norm() == 0:
        assert cc_func(x) == np.inf
    else:
        assert cc_func(x) == -constant

    assert cc_func(space.zero()) == -constant

    # The gradient of the convex conjugate (not implemeted)
    with pytest.raises(NotImplementedError):
        cc_func.gradient

    # The proximal of the convex conjugate
    sigma = np.random.rand()
    # This is the zero operator
    assert isinstance(cc_func.proximal(sigma), odl.ZeroOperator)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    expected_functional = odl.solvers.ConstantFunctional(space, constant)
    cc_cc_func = cc_func.convex_conj
    assert isinstance(cc_cc_func, type(expected_functional))


def test_zero_functional():
    """Test the constant functional."""

    n = 10
    space = odl.rn(n)
    func = odl.solvers.ZeroFunctional(space)
    x = noise_element(space)

    # Check that the constant is stored correctly in the ConstantFunctional
    assert func.constant == 0

    # Evaluation of the functional
    assert func(x) == 0

    # The gradient
    # Given by the zero-operator
    assert isinstance(func.gradient, odl.ZeroOperator)

    # The proximal operator
    sigma = np.random.rand()
    # This is the identity operator
    assert isinstance(func.proximal(sigma), odl.IdentityOperator)

    # The convex conjugate functional
    cc_func = func.convex_conj

    # Evaluation of convex conjugate
    # Explicit calculation: 0 if x=0, infty else
    if not x.norm() == 0:
        assert cc_func(x) == np.inf
    else:
        assert cc_func(x) == 0

    assert cc_func(space.zero()) == 0

    # The gradient of the convex conjugate (not implemeted)
    with pytest.raises(NotImplementedError):
        cc_func.gradient

    # The proximal of the convex conjugate
    sigma = np.random.rand()
    # This is the zero operator
    assert isinstance(cc_func.proximal(sigma), odl.ZeroOperator)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    expected_functional = odl.solvers.ConstantFunctional(space, 0)
    cc_cc_func = cc_func.convex_conj
    assert (isinstance(cc_cc_func, type(expected_functional)) and
            cc_cc_func.constant == 0)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
