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


dual_params = [False, True]
dual_ids = [' offset = {} '.format(str(p).ljust(5)) for p in offset_params]


@pytest.fixture(scope="module", ids=dual_ids, params=dual_params)
def dual(request):
    return request.param


func_params = ['l1', 'l2', 'l2^2', 'kl', 'kl_cross_ent']
func_ids = [' f = {}'.format(p.ljust(10)) for p in func_params]


@pytest.fixture(scope="module", ids=func_ids, params=func_params)
def functional(request, offset, dual, stepsize):
    """Return functional whose proximal should be tested."""
    name = request.param.strip()

    space = odl.uniform_discr(0, 1, 2)

    if name == 'l1':
        func = odl.solvers.L1Norm(space)
    elif name == 'l2':
        func = odl.solvers.L2Norm(space)
    elif name == 'l2^2':
        func = odl.solvers.L2NormSquared(space)
    elif name == 'kl':
        func = odl.solvers.KullbackLeibler(space)
    elif name == 'kl_cross_ent':
        func = odl.solvers.KullbackLeiblerCrossEntropy(space)
    else:
        assert False

    if offset:
        g = noise_element(space)
        if name.startswith('kl'):
            g = np.abs(g)
        func = func.translated(g)

    if dual:
        func = func.convex_conj

    return func


# Margin of error
EPS = 1e-6


def proximal_objective(function, x, y):
    """Calculate the objective function of the proximal optimization problem"""
    return function(y) + (1.0 / 2.0) * (x - y).norm() ** 2


def test_proximal_defintion(functional, stepsize):
    """Test the defintion of the proximal:

        prox[f](x) = argmin_y {f(y) + 1/2 ||x-y||^2}

    Hence we expect for all x in the domain of the proximal

        x* = prox[f](x)

        f(x*) + 1/2 ||x-x*||^2 <= f(y) + 1/2 ||x-y||^2
    """
    proximal = functional.proximal(stepsize)

    assert proximal.domain == proximal.range

    x = noise_element(proximal.domain) * 10
    f_x = proximal_objective(stepsize * functional, x, x)
    prox_x = proximal(x)
    f_prox_x = proximal_objective(stepsize * functional, x, prox_x)

    assert f_prox_x <= f_x + EPS

    for i in range(100):
        y = noise_element(proximal.domain)
        f_y = proximal_objective(stepsize * functional, x, y)

        if not f_prox_x <= f_y + EPS:
            print(functional, x, y, f_prox_x, f_y)

        assert f_prox_x <= f_y + EPS


def test_proximal_cconj_kl_cross_entropy_solving_opt_problem():
    """Test for proximal operator of conjguate of 2nd kind KL-divergence.

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
    kl_ce = odl.solvers.KullbackLeiblerCrossEntropy(space, prior=g)
    g_funcs = [lam_kl * kl_ce,
               0.5 * odl.solvers.L2NormSquared(space).translated(a)]
    f = odl.solvers.ZeroFunctional(space)

    # Staring point
    x = space.zero()

    odl.solvers.douglas_rachford_pd(x, f, g_funcs, lin_ops,
                                    tau=2.1, sigma=[0.4, 0.4], niter=100)

    # Explicit solution: x = W(g * exp(a)), where W is the Lambert W function.
    x_verify = lam_kl * scipy.special.lambertw(
        (g / lam_kl) * np.exp(a / lam_kl))
    assert all_almost_equal(x, x_verify, places=6)

if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v', '--largescale'])
