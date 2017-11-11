# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for ADMM."""

from __future__ import division
import odl
from odl.solvers import admm_linearized, admm_precon_nonlinear, Callback
from odl.solvers.nonsmooth.admm import (
    admm_linearized_simple, admm_precon_nonlinear_simple)

from odl.util.testutils import all_almost_equal, noise_element


def test_admm_lin_input_handling():
    """Test to see that input is handled correctly."""

    space = odl.uniform_discr(0, 1, 10)

    L = odl.ZeroOperator(space)
    f = g = odl.solvers.ZeroFunctional(space)

    # Check that the algorithm runs. With the above operators and functionals,
    # the algorithm should not modify the initial value.
    x0 = noise_element(space)
    x = x0.copy()
    niter = 3

    admm_linearized(x, f, g, L, tau=1.0, sigma=1.0, niter=niter)

    assert x == x0

    # Check that a provided callback is actually called
    class CallbackTest(Callback):
        was_called = False

        def __call__(self, *args, **kwargs):
            self.was_called = True

    callback = CallbackTest()
    assert not callback.was_called
    admm_linearized(x, f, g, L, tau=1.0, sigma=1.0, niter=niter,
                    callback=callback)
    assert callback.was_called


def test_admm_lin_l1():
    """Verify that the correct value is returned for l1 dist optimization.

    Solves the optimization problem

        min_x ||x - data_1||_1 + 0.5 ||x - data_2||_1

    which has optimum value data_1 since the first term dominates.
    """
    space = odl.rn(5)

    L = odl.IdentityOperator(space)

    data_1 = odl.util.testutils.noise_element(space)
    data_2 = odl.util.testutils.noise_element(space)

    f = odl.solvers.L1Norm(space).translated(data_1)
    g = 0.5 * odl.solvers.L1Norm(space).translated(data_2)

    x = space.zero()
    admm_linearized(x, f, g, L, tau=1.0, sigma=2.0, niter=10)

    assert all_almost_equal(x, data_1, places=2)


def test_admm_lin_vs_simple():
    """Check ``admm_linearized`` against the simple implementation."""
    space = odl.rn(5)
    L = odl.ScalingOperator(space, 2)
    y = L(odl.util.testutils.noise_element(space))
    f = odl.solvers.L2NormSquared(space).translated(y)
    g = 0.5 * odl.solvers.L1Norm(space)

    x_simple = space.zero()
    admm_linearized_simple(x_simple, f, g, L, tau=1.0, sigma=2.0, niter=10)
    x_optim = space.zero()
    admm_linearized(x_optim, f, g, L, tau=1.0, sigma=2.0, niter=10)

    assert all_almost_equal(x_optim, x_simple)


def test_admm_nonlin_affine_l1():
    """Verify that the correct value is returned for l1 dist optimization.

    Solves the optimization problem

        min_x ||L(x) - y_1||_1 + 0.5 ||L(x) - y_2||_1

    with ``L(x) = x + 1``, which has optimum value ``x_1 = y_1 - 1`` since the
    first term dominates.
    """
    space = odl.rn(5)

    L = odl.IdentityOperator(space) + space.one()

    x1 = odl.util.testutils.noise_element(space)
    x2 = odl.util.testutils.noise_element(space)
    y1 = L(x1)
    y2 = L(x2)

    f = odl.solvers.L1Norm(space).translated(y1)
    g = 0.5 * odl.solvers.L1Norm(space).translated(y2)

    x = space.zero()
    admm_precon_nonlinear(x, f, g, L, delta=1.0, niter=10)

    assert all_almost_equal(x, x, places=2)


def test_admm_nonlin_vs_simple():
    """Check ``admm_precon_nonlinear`` against the simple implementation."""
    space = odl.rn(5)
    L = odl.ufunc_ops.sin(space) * odl.ScalingOperator(space, 2)
    y = L(odl.util.testutils.noise_element(space))
    f = odl.solvers.L2NormSquared(space).translated(y)
    g = 0.5 * odl.solvers.L1Norm(space)

    x_simple = space.zero()
    admm_precon_nonlinear_simple(x_simple, f, g, L, delta=1.0, niter=2)
    x_optim = space.zero()
    admm_precon_nonlinear(x_optim, f, g, L, delta=1.0, niter=2)

    assert all_almost_equal(x_optim, x_simple)


if __name__ == '__main__':
    odl.util.test_file(__file__)
