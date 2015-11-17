# Copyright 2014, 2015 The ODL development group
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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

# External
import numpy as np

# Internal
from odl.diagnostics.examples import vector_examples, samples
from odl.util.testutils import FailCounter


__all__ = ('OperatorTest',)


class OperatorTest(object):
    """ Automated tests for :class:`~odl.Operator`'s

    This class allows users to automatically test various
    features of an Operator such as linearity and the
    adjoint definition.
    """

    def __init__(self, operator, operator_norm=None):
        """Create a new instance

        Parameters
        ----------
        operator : :class:`~odl.Operator`
            The operator to run tests on
        operator_norm : `float`
            The norm of the operator, used for error estimates
            can be estimated otherwise.
        """
        self.operator = operator
        self.operator_norm = operator_norm

    def norm(self):
        """Estimate the operator norm of the operator.

        The norm is estimated by calculating

        ``A(x).norm() / x.norm()``

        for some nonzero ``x``

        Returns
        -------
        norm : `float`
            Estimate of operator norm

        References
        ----------
        Wikipedia article on `Operator norm
        <https://en.wikipedia.org/wiki/Operator_norm>`_.
        """
        print('\n== Calculating operator norm ==\n')

        operator_norm = 0.0
        for [n_x, x] in samples(self.operator.domain):
            result = self.operator(x)
            x_norm = x.norm()
            estimate = 0 if x_norm == 0 else result.norm() / x_norm

            operator_norm = max(operator_norm, estimate)

        print('Norm is at least: {}'.format(operator_norm))
        self.operator_norm = operator_norm
        return operator_norm

    def _adjoint_definition(self):
        """Verify (Ax, y) = (x, A^T y)"""
        print('\nVerifying the identity (Ax, y) = (x, A^T y)')

        Axy_vals = []
        xAty_vals = []

        with FailCounter('error = ||(Ax, y) - (x, A^T y)|| / '
                         '||A|| ||x|| ||y||') as counter:
            for [n_x, x], [n_y, y] in samples(self.operator.domain,
                                              self.operator.range):
                x_norm = x.norm()
                y_norm = y.norm()

                Axy = self.operator(x).inner(y)
                xAty = x.inner(self.operator.adjoint(y))

                denom = self.operator_norm * x_norm * y_norm
                error = 0 if denom == 0 else abs(Axy - xAty) / denom

                if error > 0.00001:
                    counter.fail('x={:25s} y={:25s} : error={:6.5f}'
                                 ''.format(n_x, n_y, error))

                Axy_vals.append(Axy)
                xAty_vals.append(xAty)

        scale = np.polyfit(Axy_vals, xAty_vals, 1)[0]
        print('\nThe adjoint seems to be scaled according to:')
        print('(x, A^T y) / (Ax, y) = {}. Should be 1.0'.format(scale))

    def _adjoint_of_adjoint(self):
        """Verify (A^*)^* = A"""
        try:
            self.operator.adjoint.adjoint
        except AttributeError:
            print('A^* has no adjoint')
            return

        if self.operator.adjoint.adjoint is self.operator:
            print('(A^*)^* == A')
            return

        print('\nVerifying the identity Ax = (A^T)^T x')

        with FailCounter('error = ||Ax - (A^T)^T x|| / '
                         '||A|| ||x||') as counter:
            for [n_x, x] in vector_examples(self.operator.domain):
                A_result = self.operator(x)
                ATT_result = self.operator.adjoint.adjoint(x)

                denom = self.operator_norm * x.norm()
                if denom == 0:
                    error = 0
                else:
                    error = (A_result - ATT_result).norm() / denom
                if error > 0.00001:
                    counter.fail('x={:25s} : error={:6.5f}'
                                 ''.format(n_x, error))

    def adjoint(self):
        """Verify that the adjoint works appropriately.

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

        print('\n== Verifying adjoint of operator ==\n')

        domain_range_ok = True
        if self.operator.domain != self.operator.adjoint.range:
            print('*** ERROR: A.domain != A.adjoint.range ***')
            domain_range_ok = False

        if self.operator.range != self.operator.adjoint.domain:
            print('*** ERROR: A.domain != A.adjoint.range ***')
            domain_range_ok = False

        if domain_range_ok:
            print('Domain and range of adjoint is OK.')
        else:
            print('Domain and range of adjoint not OK exiting.')
            return

        self._adjoint_definition()
        self._adjoint_of_adjoint()

    def _derivative_convergence(self, step):
        print('\nTesting derivative is linear approximation')

        with FailCounter("error = ||A(x+c*dx)-A(x)-c*A'(x)(dx)|| / "
                         "|c| ||dx|| ||A||") as counter:
            for [n_x, x], [n_dx, dx] in samples(self.operator.domain,
                                                self.operator.domain):
                deriv = self.operator.derivative(x)
                opx = self.operator(x)

                exact_step = self.operator(x + dx * step) - opx
                expected_step = deriv(dx * step)
                denom = step * dx.norm() * self.operator_norm
                error = (0 if denom == 0
                         else (exact_step - expected_step).norm() / denom)

                if error > 0.00001:
                    counter.fail('x={:15s} dx={:15s} c={}: error={:6.5f}'
                                 ''.format(n_x, n_dx, step, error))

    def derivative(self, step=0.0001):
        """Verify that the derivative works appropriately.

        References
        ----------
        Wikipedia article on `Derivative
        <https://en.wikipedia.org/wiki/Derivative>`_.
        Wikipedia article on `Fréchet derivative
        <https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative>`_.
        """

        print('\n==Verifying derivative of operator ==')
        try:
            deriv = self.operator.derivative(self.operator.domain.zero())

            if not deriv.is_linear:
                print('Derivative is not a linear operator')
                return
        except NotImplementedError:
            print('Operator has no derivative')
            return

        if self.operator_norm is None:
            print('Cannot do tests before norm is calculated, run test.norm() '
                  'or give norm as a parameter')
            return

        if deriv == self:
            print('A is linear and A.derivative == A')
            return

        self._derivative_convergence(step)

    def _scale_invariance(self):
        print("\nCalculating invariance under scaling")

        # Test scaling
        with FailCounter('error = ||A(c*x)-c*A(x)|| / '
                         '|c| ||A|| ||x||') as counter:
            for [n_x, x], scale in samples(self.operator.domain,
                                           self.operator.domain.field):
                opx = self.operator(x)
                scaled_opx = self.operator(scale * x)

                denom = self.operator_norm * scale * x.norm()
                error = (0 if denom == 0
                         else (scaled_opx - opx * scale).norm() / denom)

                if error > 0.00001:
                    counter.fail('x={:25s} scale={:7.2f} error={:6.5f}'
                                 ''.format(n_x, scale, error))

    def _addition_invariance(self):
        print("\nCalculating invariance under addition")

        # Test addition
        with FailCounter('error = ||A(x+y) - A(x) - A(y)|| / '
                         '||A||(||x|| + ||y||)') as counter:
            for [n_x, x], [n_y, y] in samples(self.operator.domain,
                                              self.operator.domain):
                opx = self.operator(x)
                opy = self.operator(y)
                opxy = self.operator(x + y)

                denom = self.operator_norm * (x.norm() + y.norm())
                error = (0 if denom == 0
                         else (opxy - opx - opy).norm() / denom)

                if error > 0.00001:
                    counter.fail('x={:25s} y={:25s} error={:6.5f}'
                                 ''.format(n_x, n_y, error))

    def linear(self):
        """Verify that the operator is actualy linear."""
        if not self.operator.is_linear:
            print('Operator is not linear')
            return

        if self.operator_norm is None:
            print('Cannot do tests before norm is calculated, run test.norm() '
                  'or give norm as a parameter')
            return

        print('\n== Verifying linearity of operator ==\n')

        # Test zero gives zero
        result = self.operator(self.operator.domain.zero())
        print("||A(0)||={:6.5f}. Should be 0.0000".format(result.norm()))

        self._scale_invariance()
        self._addition_invariance()

    def run_tests(self):
        """Run all tests on this operator."""
        print('\n== RUNNING ALL TESTS ==\n')
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
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
