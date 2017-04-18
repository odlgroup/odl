# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the smooth solvers."""

from __future__ import division
import pytest
import odl
from odl.operator import OpNotImplementedError


nonlinear_cg_beta = odl.util.testutils.simple_fixture('nonlinear_cg_beta',
                                                      ['FR', 'PR', 'HS', 'DY'])


@pytest.fixture(scope="module", params=['l2_squared', 'l2_squared_scaled',
                                        'rosenbrock'])
def functional(request):
    """functional with optimum 0 at 0."""
    name = request.param

    # TODO: quadratic (#606) functionals

    if name == 'l2_squared':
        space = odl.rn(3)
        return odl.solvers.L2NormSquared(space)
    elif name == 'l2_squared_scaled':
        space = odl.uniform_discr(0, 1, 3)
        scaling = odl.MultiplyOperator(space.element([1, 2, 3]),
                                       domain=space)
        return odl.solvers.L2NormSquared(space) * scaling
    elif name == 'rosenbrock':
        # Moderately ill-behaved rosenbrock functional.
        rosenbrock = odl.solvers.RosenbrockFunctional(odl.rn(2), scale=2)

        # Center at zero
        return rosenbrock.translated([-1, -1])
    else:
        assert False


@pytest.fixture(scope="module", params=['constant', 'backtracking'])
def functional_and_linesearch(request, functional):
    """Return functional with optimum 0 at 0 and a line search."""
    name = request.param

    if name == 'constant':
        return functional, 1.0
    else:
        return functional, odl.solvers.BacktrackingLineSearch(functional)


@pytest.fixture(scope="module", params=['first', 'second'])
def broyden_impl(request):
    return request.param


def test_newton_solver(functional_and_linesearch):
    """Test the Newton solver."""
    functional, line_search = functional_and_linesearch

    try:
        # Test if derivative exists
        functional.gradient.derivative(functional.domain.zero())
    except OpNotImplementedError:
        return

    # Solving the problem
    x = functional.domain.one()
    odl.solvers.newtons_method(functional, x, tol=1e-6,
                               line_search=line_search)

    # Assert x is close to the optimum at [1, 1]
    assert functional(x) < 1e-3


def test_bfgs_solver(functional_and_linesearch):
    """Test the BFGS quasi-Newton solver."""
    functional, line_search = functional_and_linesearch

    x = functional.domain.one()
    odl.solvers.bfgs_method(functional, x, tol=1e-6,
                            line_search=line_search)

    assert functional(x) < 1e-3


def test_lbfgs_solver(functional_and_linesearch):
    """Test limited memory BFGS quasi-Newton solver."""
    functional, line_search = functional_and_linesearch

    x = functional.domain.one()
    odl.solvers.bfgs_method(functional, x, tol=1e-6,
                            line_search=line_search, num_store=5)

    assert functional(x) < 1e-3


def test_broydens_method(broyden_impl, functional_and_linesearch):
    """Test the ``broydens_method`` quasi-Newton solver."""
    functional, line_search = functional_and_linesearch

    x = functional.domain.one()
    odl.solvers.broydens_method(functional, x, tol=1e-6,
                                line_search=line_search, impl=broyden_impl)

    assert functional(x) < 1e-3


def test_steepest_descent(functional):
    """Test the ``steepest_descent`` solver."""
    line_search = odl.solvers.BacktrackingLineSearch(functional)

    x = functional.domain.one()
    odl.solvers.steepest_descent(functional, x, tol=1e-6,
                                 line_search=line_search)

    assert functional(x) < 1e-3


def test_conjguate_gradient_nonlinear(functional, nonlinear_cg_beta):
    """Test the ``conjugate_gradient_nonlinear`` solver."""
    line_search = odl.solvers.BacktrackingLineSearch(functional)

    x = functional.domain.one()
    odl.solvers.conjugate_gradient_nonlinear(functional, x, tol=1e-6,
                                             line_search=line_search,
                                             beta_method=nonlinear_cg_beta)

    assert functional(x) < 1e-3


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
