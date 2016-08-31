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

"""Standardized tests for `Operator`'s."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

import numpy as np

from odl.diagnostics.examples import samples
from odl.operator import power_method_opnorm
from odl.util.testutils import FailCounter


__all__ = ('OperatorTest',)


class OperatorTest(object):

    """Automated tests for `Operator` implementations.

    This class allows users to automatically test various
    features of an Operator such as linearity, the adjoint definition and
    definition of the derivative.
    """

    def __init__(self, operator, operator_norm=None, verbose=True, tol=1e-5):
        """Initialize a new instance.

        Parameters
        ----------
        operator : `Operator`
            The operator to run tests on
        operator_norm : float, optional
            The norm of the operator, used for error estimates. If
            ``None`` is given, the norm is estimated during
            initialization.
        verbose : bool, optional
            If ``True``, print additional info text.
        tol : float, optional
            Tolerance parameter used as a base for the actual tolerance
            in the tests. Depending on the expected accuracy, the actual
            tolerance used in a test can be a factor times this number.
        """
        self.operator = operator
        self.verbose = False
        if operator_norm is None:
            self.operator_norm = self.norm()
        else:
            self.operator_norm = float(operator_norm)

        self.verbose = bool(verbose)
        self.tol = float(tol)

    def log(self, message):
        """Print message if ``self.verbose == True``."""
        if self.verbose:
            print(message)

    def norm(self):
        """Estimate the operator norm of the operator.

        The norm is estimated by calculating

        ``A(x).norm() / x.norm()``

        for some nonzero ``x``

        Returns
        -------
        norm : float
            Estimate of operator norm

        References
        ----------
        Wikipedia article on `Operator norm
        <https://en.wikipedia.org/wiki/Operator_norm>`_.
        """
        self.log('\n== Calculating operator norm ==\n')

        operator_norm = max(power_method_opnorm(self.operator, maxiter=2,
                                                xstart=x)
                            for name, x in samples(self.operator.domain)
                            if name != 'Zero')

        self.log('Norm is at least: {}'.format(operator_norm))
        self.operator_norm = operator_norm
        return operator_norm

    def self_adjoint(self):
        """Verify ``<Ax, y> == <x, Ay>``."""
        left_inner_vals = []
        right_inner_vals = []

        with FailCounter(
                test_name='Verifying the identity <Ax, y> = <x, Ay>',
                err_msg='error = |<Ax, y> - <x, Ay>| / ||A|| ||x|| ||y||',
                logger=self.log) as counter:

            for [name_x, x], [name_y, y] in samples(self.operator.domain,
                                                    self.operator.range):
                x_norm = x.norm()
                y_norm = y.norm()

                l_inner = self.operator(x).inner(y)
                r_inner = x.inner(self.operator(y))

                denom = self.operator_norm * x_norm * y_norm
                error = 0 if denom == 0 else abs(l_inner - r_inner) / denom

                if error > self.tol:
                    counter.fail('x={:25s} y={:25s} : error={:6.5f}'
                                 ''.format(name_x, name_y, error))

                left_inner_vals.append(l_inner)
                right_inner_vals.append(r_inner)

        scale = np.polyfit(left_inner_vals, right_inner_vals, 1)[0]
        self.log('\nThe adjoint seems to be scaled according to:')
        self.log('(x, Ay) / (Ax, y) = {}. Should be 1.0'.format(scale))

    def _adjoint_definition(self):
        """Verify ``<Ax, y> == <x, A^* y>``."""
        left_inner_vals = []
        right_inner_vals = []

        with FailCounter(
                test_name='Verifying the identity <Ax, y> = <x, A^T y>',
                err_msg='error = |<Ax, y< - <x, A^* y>| / ||A|| ||x|| ||y||',
                logger=self.log) as counter:

            for [name_x, x], [name_y, y] in samples(self.operator.domain,
                                                    self.operator.range):
                x_norm = x.norm()
                y_norm = y.norm()

                l_inner = self.operator(x).inner(y)
                r_inner = x.inner(self.operator.adjoint(y))

                denom = self.operator_norm * x_norm * y_norm
                error = 0 if denom == 0 else abs(l_inner - r_inner) / denom

                if error > self.tol:
                    counter.fail('x={:25s} y={:25s} : error={:6.5f}'
                                 ''.format(name_x, name_y, error))

                left_inner_vals.append(l_inner)
                right_inner_vals.append(r_inner)

        scale = np.polyfit(left_inner_vals, right_inner_vals, 1)[0]
        self.log('\nThe adjoint seems to be scaled according to:')
        self.log('(x, A^T y) / (Ax, y) = {}. Should be 1.0'.format(scale))

    def _adjoint_of_adjoint(self):
        """Verify ``(A^*)^* == A``"""
        try:
            self.operator.adjoint.adjoint
        except AttributeError:
            print('A^* has no adjoint')
            return

        if self.operator.adjoint.adjoint is self.operator:
            self.log('(A^*)^* == A')
            return

        with FailCounter(
                test_name='\nVerifying the identity Ax = (A^*)^* x',
                err_msg='error = ||Ax - (A^*)^* x|| / ||A|| ||x||',
                logger=self.log) as counter:
            for [name_x, x] in self.operator.domain.examples:
                opx = self.operator(x)
                op_adj_adj_x = self.operator.adjoint.adjoint(x)

                denom = self.operator_norm * x.norm()
                if denom == 0:
                    error = 0
                else:
                    error = (opx - op_adj_adj_x).norm() / denom

                if error > self.tol:
                    counter.fail('x={:25s} : error={:6.5f}'
                                 ''.format(name_x, error))

    def adjoint(self):
        """Verify that `Operator.adjoint` works appropriately.

        References
        ----------
        Wikipedia article on `Adjoint
        <https://en.wikipedia.org/wiki/Adjoint>`_.
        """
        try:
            self.operator.adjoint
        except NotImplementedError:
            print('Operator has no adjoint')
            return

        self.log('\n== Verifying operator adjoint ==\n')

        domain_range_ok = True
        if self.operator.domain != self.operator.adjoint.range:
            print('*** ERROR: A.domain != A.adjoint.range ***')
            domain_range_ok = False

        if self.operator.range != self.operator.adjoint.domain:
            print('*** ERROR: A.range != A.adjoint.domain ***')
            domain_range_ok = False

        if domain_range_ok:
            self.log('Domain and range of adjoint are OK.')
        else:
            print('Domain and range of adjoint are not OK, exiting.')
            return

        self._adjoint_definition()
        self._adjoint_of_adjoint()

    def _derivative_convergence(self):
        """Verify that the derivative is a first-order approximation.

        The code verifies if

            ``||A(x+c*p) - A(x) - A'(x)(c*p)|| / c = o(c)``

        for ``c --> 0``.
        """
        with FailCounter(
                test_name='Verifying that derivative is a first-order '
                          'approximation',
                err_msg="error = inf_c ||A(x+c*p)-A(x)-A'(x)(c*p)|| / c",
                logger=self.log) as counter:
            for [name_x, x], [name_dx, dx] in samples(self.operator.domain,
                                                      self.operator.domain):
                # Precompute some values
                deriv = self.operator.derivative(x)
                derivdx = deriv(dx)
                opx = self.operator(x)

                c = 1e-4  # initial step
                derivative_ok = False

                minerror = float('inf')
                while c > 1e-14:
                    exact_step = self.operator(x + dx * c) - opx
                    expected_step = c * derivdx
                    err = (exact_step - expected_step).norm() / c

                    # Need to be slightly more generous here due to possible
                    # numerical instabilities.
                    # TODO: perform more tests to find a good threshold here.
                    if err < 10 * self.tol:
                        derivative_ok = True
                        break
                    else:
                        minerror = min(minerror, err)

                    c /= 10.0

                if not derivative_ok:
                    counter.fail('x={:15s} p={:15s}, error={}'
                                 ''.format(name_x, name_dx, minerror))

    def derivative(self):
        """Verify that `Operator.derivative` works appropriately.

        The code verifies if

            ``||A(x+c*p) - A(x) - A'(x)(c*p)|| / c = o(c)``

        for ``c --> 0`` using a selection of elements ``x`` and ``p``.

        References
        ----------
        Wikipedia article on `Derivative
        <https://en.wikipedia.org/wiki/Derivative>`_.
        Wikipedia article on `Frechet derivative
        <https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative>`_.
        """
        self.log('\n== Verifying operator derivative  ==')

        try:
            deriv = self.operator.derivative(self.operator.domain.zero())

            if not deriv.is_linear:
                print('Derivative is not a linear operator')
                return
        except NotImplementedError:
            print('Operator has no derivative')
            return

        if self.operator.is_linear and deriv is self.operator:
            self.log('A is linear and A.derivative is A')
            return

        self._derivative_convergence()

    def _scale_invariance(self):
        """Verify ``A(c*x) = c * A(x)``."""
        with FailCounter(
                test_name='Verifying homogeneity under scalar multiplication',
                err_msg='error = ||A(c*x)-c*A(x)|| / |c| ||A|| ||x||',
                logger=self.log) as counter:
            for [name_x, x], [_, scale] in samples(self.operator.domain,
                                                   self.operator.domain.field):
                opx = self.operator(x)
                scaled_opx = self.operator(scale * x)

                denom = self.operator_norm * scale * x.norm()
                error = (0 if denom == 0
                         else (scaled_opx - opx * scale).norm() / denom)

                if error > self.tol:
                    counter.fail('x={:25s} scale={:7.2f} error={:6.5f}'
                                 ''.format(name_x, scale, error))

    def _addition_invariance(self):
        """Verify ``A(x+y) = A(x) + A(y)``."""
        with FailCounter(
                test_name='Verifying distributivity under vector addition',
                err_msg='error = ||A(x+y) - A(x) - A(y)|| / '
                        '||A||(||x|| + ||y||)',
                logger=self.log) as counter:
            for [name_x, x], [name_y, y] in samples(self.operator.domain,
                                                    self.operator.domain):
                opx = self.operator(x)
                opy = self.operator(y)
                opxy = self.operator(x + y)

                denom = self.operator_norm * (x.norm() + y.norm())
                error = (0 if denom == 0
                         else (opxy - opx - opy).norm() / denom)

                if error > self.tol:
                    counter.fail('x={:25s} y={:25s} error={:6.5f}'
                                 ''.format(name_x, name_y, error))

    def linear(self):
        """Verify that the operator is actually linear."""
        if not self.operator.is_linear:
            print('Operator is not linear')
            return

        self.log('\n== Verifying operator linearity ==\n')

        # Test if zero gives zero
        result = self.operator(self.operator.domain.zero())
        result_norm = result.norm()
        if result_norm != 0.0:
            print("||A(0)||={:6.5f}. Should be 0.0000".format(result_norm))

        self._scale_invariance()
        self._addition_invariance()

    def run_tests(self):
        """Run all tests on this operator."""
        print('\n== RUNNING ALL TESTS ==')
        print('Operator = {}'.format(self.operator))

        self.norm()

        if self.operator.is_linear:
            self.linear()
            self.adjoint()
        else:
            self.derivative()

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.operator)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.operator)

if __name__ == '__main__':
    # pylint: disable=wrong-import-position

    import odl
    space = odl.uniform_discr([0, 0], [1, 1], [3, 3])
    # Linear operator
    I = odl.IdentityOperator(space)
    OperatorTest(I, verbose=False).run_tests()

    # Nonlinear operator op(x) = x**4
    op = odl.PowerOperator(space, 4)
    OperatorTest(op).run_tests()
