# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the factory functions to create proximal operators."""

from __future__ import division
import numpy as np
import pytest
import scipy.special
import odl
from odl.util.testutils import (noise_element, all_almost_equal,
                                simple_fixture)
from odl.solvers.functional.functional import FunctionalDefaultConvexConjugate


# --- pytest fixtures --- #


pytestmark = odl.util.skip_if_no_largescale

stepsize = simple_fixture('stepsize', [0.1, 1.0, 10.0])
linear_offset = simple_fixture('linear_offset', ['none', True])
quadratic_offset = simple_fixture('quadratic_offset', [False, True])
dual = simple_fixture('dual', [False, True])


func_params = ['l1', 'l2', 'l2^2', 'kl', 'kl_cross_ent', 'const',
               'groupl1-1', 'groupl1-2',
               'nuclearnorm-1-1', 'nuclearnorm-1-2', 'nuclearnorm-1-inf',
               'quadratic', 'linear']

func_ids = [' f = {} '.format(p.ljust(10)) for p in func_params]


@pytest.fixture(scope="module", ids=func_ids, params=func_params)
def functional(request, linear_offset, quadratic_offset, dual):
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
    elif name == 'const':
        func = odl.solvers.ConstantFunctional(space, constant=2)
    elif name.startswith('groupl1'):
        exponent = float(name.split('-')[1])
        space = odl.ProductSpace(space, 2)
        func = odl.solvers.GroupL1Norm(space, exponent=exponent)
    elif name.startswith('nuclearnorm'):
        outer_exp = float(name.split('-')[1])
        singular_vector_exp = float(name.split('-')[2])

        space = odl.ProductSpace(odl.ProductSpace(space, 2), 3)
        func = odl.solvers.NuclearNorm(space,
                                       outer_exp=outer_exp,
                                       singular_vector_exp=singular_vector_exp)
    elif name == 'quadratic':
        func = odl.solvers.QuadraticForm(operator=odl.IdentityOperator(space),
                                         vector=space.one(), constant=0.623)
    elif name == 'linear':
        func = odl.solvers.QuadraticForm(vector=space.one(), constant=0.623)
    else:
        assert False

    if quadratic_offset:
        if linear_offset:
            g = noise_element(space)
            if name.startswith('kl'):
                g = np.abs(g)
        else:
            g = None

        quadratic_term = 1.32
        func = odl.solvers.FunctionalQuadraticPerturb(
                  func, quadratic_term=quadratic_term, linear_term=g)
    elif linear_offset:
        g = noise_element(space)
        if name.startswith('kl'):
            g = np.abs(g)
        func = func.translated(g)

    if dual:
        func = func.convex_conj

    return func


# Margin of error
EPS = 1e-6


# --- Functional tests --- #


def proximal_objective(functional, x, y):
    """Objective function of the proximal optimization problem."""
    return functional(y) + (1.0 / 2.0) * (x - y).norm() ** 2


def test_proximal_defintion(functional, stepsize):
    """Test the defintion of the proximal:

        prox[f](x) = argmin_y {f(y) + 1/2 ||x-y||^2}

    Hence we expect for all x in the domain of the proximal

        x* = prox[f](x)

        f(x*) + 1/2 ||x-x*||^2 <= f(y) + 1/2 ||x-y||^2
    """
    if isinstance(functional, FunctionalDefaultConvexConjugate):
        pytest.skip('functional has no call method')
        return

    # No implementation of the proximal for convex conj of
    # FunctionalQuadraticPerturb unless the quadratic term is 0.
    if (isinstance(functional, odl.solvers.FunctionalQuadraticPerturb) and
            functional.quadratic_term != 0):
        pytest.skip('functional has no proximal')
        return

    # No implementation of the proximal for quardartic form
    if isinstance(functional, odl.solvers.QuadraticForm):
        pytest.skip('functional has no proximal')
        return

    # No implementation of the proximal for translations of quardartic form
    if (isinstance(functional, odl.solvers.FunctionalTranslation) and
            isinstance(functional.functional, odl.solvers.QuadraticForm)):
        pytest.skip('functional has no proximal')
        return

    # No implementation of the proximal for convex conj of quardartic form,
    # except if the quadratic part is 0.
    if (isinstance(functional, odl.solvers.FunctionalQuadraticPerturb) and
            isinstance(functional.functional, odl.solvers.QuadraticForm) and
            functional.functional.operator is not None):
        pytest.skip('functional has no proximal')
        return

    proximal = functional.proximal(stepsize)

    assert proximal.domain == functional.domain
    assert proximal.range == functional.domain

    for _ in range(100):
        x = noise_element(proximal.domain) * 10
        prox_x = proximal(x)
        f_prox_x = proximal_objective(stepsize * functional, x, prox_x)

        y = noise_element(proximal.domain)
        f_y = proximal_objective(stepsize * functional, x, y)

        if not f_prox_x <= f_y + EPS:
            print(repr(functional), x, y, prox_x, f_prox_x, f_y)

        assert f_prox_x <= f_y + EPS


def convex_conj_objective(functional, x, y):
    """CObjective function of the convex conjugate problem."""
    return x.inner(y) - functional(x)


def test_convex_conj_defintion(functional):
    """Test the defintion of the convex conjugate:

        f^*(y) = sup_x {<x, y> - f(x)}

    Hence we expect for all x in the domain of the proximal

        <x, y> - f(x) <= f^*(y)
    """
    if isinstance(functional, FunctionalDefaultConvexConjugate):
        pytest.skip('functional has no call')
        return

    f_convex_conj = functional.convex_conj
    if isinstance(f_convex_conj, FunctionalDefaultConvexConjugate):
        pytest.skip('functional has no convex conjugate')
        return

    for _ in range(100):
        y = noise_element(functional.domain)
        f_convex_conj_y = f_convex_conj(y)

        x = noise_element(functional.domain)
        lhs = x.inner(y) - functional(x)

        if not lhs <= f_convex_conj_y + EPS:
            print(repr(functional), repr(f_convex_conj), x, y, lhs,
                  f_convex_conj_y)

        assert lhs <= f_convex_conj_y + EPS


def test_proximal_convex_conj_kl_cross_entropy_solving_opt_problem():
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
