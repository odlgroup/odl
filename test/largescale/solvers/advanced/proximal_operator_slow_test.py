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
from odl.solvers.advanced.proximal_operators import (
    proximal_l1, proximal_cconj_l1,
    proximal_l2, proximal_cconj_l2,
    proximal_l2_squared, proximal_cconj_l2_squared,
    proximal_cconj_kl, proximal_cconj_kl_cross_entropy)
from odl.util.testutils import (noise_element, all_almost_equal)

pytestmark = odl.util.skip_if_no_largescale


step_params = [0.1, 1.0, 10.0]
step_ids = [' stepsize = {} '.format(p) for p in step_params]


@pytest.fixture(scope="module", ids=step_ids, params=step_params)
def stepsize(request):
    return request.param


offset_params = [False, True]
offset_ids = [' offset = {} '.format(str(p).ljust(5)) for p in offset_params]


@pytest.fixture(scope="module", ids=offset_ids, params=offset_params)
def offset(request):
    return request.param


def make_offset(g, stepsize, convex_conjugate):
    """Decorator that adds an optional offset with stepsize."""
    def offset_function(function):
        if g is None and not convex_conjugate:
            return lambda x: stepsize * function(x)
        elif g is None and convex_conjugate:
            return lambda x: stepsize * function(x)
        elif g is not None and not convex_conjugate:
            return lambda x: stepsize * function(x - g)
        elif g is not None and convex_conjugate:
            return lambda x: stepsize * (function(x) + x.inner(g))
        else:
            assert False
    return offset_function

prox_params = ['l1 ', 'l1_dual',
               'l2', 'l2_dual',
               'l2^2', 'l2^2_dual',
               'kl_dual', 'kl_cross_ent_dual']
prox_ids = [' f = {}'.format(p.ljust(10)) for p in prox_params]


@pytest.fixture(scope="module", ids=prox_ids, params=prox_params)
def proximal_and_function(request, stepsize, offset):
    """Return a proximal factory and the corresponding function."""
    name = request.param.strip()

    space = odl.uniform_discr(0, 1, 2)

    if offset:
        g = noise_element(space)
    else:
        g = None

    if name == 'l1':
        @make_offset(g, stepsize=stepsize, convex_conjugate=False)
        def l1_norm(x):
            return np.abs(x).inner(x.space.one())

        prox = proximal_l1(space, g=g)

        return prox(stepsize), l1_norm

    if name == 'l1_dual':
        @make_offset(g, stepsize=stepsize, convex_conjugate=True)
        def l1_norm_dual(x):
            return 0.0 if np.max(np.abs(x)) <= 1.0 else np.Infinity

        prox = proximal_cconj_l1(space, g=g)

        return prox(stepsize), l1_norm_dual

    elif name == 'l2':
        @make_offset(g, stepsize=stepsize, convex_conjugate=False)
        def l2_norm(x):
            return x.norm()

        prox = proximal_l2(space, g=g)

        return prox(stepsize), l2_norm

    elif name == 'l2_dual':
        @make_offset(g, stepsize=stepsize, convex_conjugate=True)
        def l2_norm_dual(x):
            # numerical margin
            return 0.0 if x.norm() < 1.00001 else np.Infinity

        prox = proximal_cconj_l2(space, g=g)

        return prox(stepsize), l2_norm_dual

    elif name == 'l2^2':
        @make_offset(g, stepsize=stepsize, convex_conjugate=False)
        def l2_norm_squared(x):
            return x.norm() ** 2

        prox = proximal_l2_squared(space, g=g)

        return prox(stepsize), l2_norm_squared

    elif name == 'l2^2_dual':
        @make_offset(g, stepsize=stepsize, convex_conjugate=True)
        def l2_norm_squared_dual(x):
            return (1.0 / 4.0) * x.norm() ** 2

        prox = proximal_cconj_l2_squared(space, g=g)

        return prox(stepsize), l2_norm_squared_dual

    elif name == 'kl_dual':
        if g is not None:
            g = np.abs(g)

        def kl_divergence_dual(x):
            if np.greater_equal(x, 1):
                return np.Infinity
            else:
                one_element = x.space.one()
                if g is None:
                    return stepsize * one_element.inner(
                        np.log(one_element - x))
                else:
                    return stepsize * one_element.inner(
                        g * np.log(one_element - x))

        prox = proximal_cconj_kl(space, g=g)

        return prox(stepsize), kl_divergence_dual

    elif name == 'kl_cross_ent_dual':
        if g is not None:
            g = np.abs(g)

        def kl_divergence_cross_entropy_dual(x):
            one_element = x.space.one()
            if g is None:
                return stepsize * one_element.inner(np.exp(x) - one_element)
            else:
                return stepsize * one_element.inner(
                    g * (np.exp(x) - one_element))

        prox = proximal_cconj_kl_cross_entropy(space, g=g)

        return prox(stepsize), kl_divergence_cross_entropy_dual

    else:
        assert False


def proximal_objective(function, x, y):
    """Calculate the objective function of the proximal optimization problem"""
    return function(y) + (1.0 / 2.0) * (x - y).norm() ** 2


def test_proximal_defintion(proximal_and_function):
    """Test the defintion of the proximal:

        prox[f](x) = argmin_y {f(y) + 1/2 ||x-y||^2}

    Hence we expect for all x in the domain of the proximal

        x* = prox[f](x)

        f(x*) + 1/2 ||x-x*||^2 < f(y) + 1/2 ||x-y||^2
    """

    proximal, function = proximal_and_function

    assert proximal.domain == proximal.range

    x = noise_element(proximal.domain) * 10
    f_x = proximal_objective(function, x, x)
    prox_x = proximal(x)
    f_prox_x = proximal_objective(function, x, prox_x)

    assert f_prox_x <= f_x

    for i in range(100):
        y = noise_element(proximal.domain)
        f_y = proximal_objective(function, x, y)

        assert f_prox_x <= f_y


def test_proximal_cconj_kl_cross_entropy_solving_opt_problem():
    """Test for proximal operator of conjguate of 2nd kind KL-divergecen.

    The test solves the problem

        min_x lam*KL(x | g) + 1/2||x-a||^2_2,

    where g is the nonnegative prior, and a is any vector.  Explicit solution
    to this problem is given by

        x = lam*W(g*e^(a/lam)/lam),

    where W is the Lambert W function.
    """

    # Image space
    space = odl.uniform_discr(0, 1, 10)

    # Data
    g = space.element(np.arange(10, 0, -1))
    a = space.element(np.arange(4, 14, 1))

    # Creating and assembling linear operators and proximals
    id_op = odl.IdentityOperator(space)
    lin_ops = [id_op, id_op]
    lam_kl = 2.3
    prox_cc_g = [odl.solvers.proximal_cconj_kl_cross_entropy(space, lam=lam_kl,
                                                             g=g),
                 odl.solvers.proximal_cconj_l2_squared(space, lam=1.0 / 2.0,
                                                       g=a)]
    prox_f = odl.solvers.proximal_zero(space)

    # Staring point
    x = space.zero()

    odl.solvers.douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops,
                                    tau=2.1, sigma=[0.4, 0.4], niter=100)

    # Explicit solution: x = W(g * exp(a)), where W is the Lambert W function.
    x_verify = lam_kl * scipy.special.lambertw(
        (1.0 / lam_kl) * g * np.exp(a / lam_kl))
    assert all_almost_equal(x, x_verify, places=6)

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v --largescale'))
