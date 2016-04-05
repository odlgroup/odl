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
from odl.util.testutils import FailCounter


__all__ = ('OperatorTest',)


class OperatorTest(object):

    """Automated tests for `Operator` implementations.

    This class allows users to automatically test various
    features of an Operator such as linearity and the
    adjoint definition.
    """

    def __init__(self, operator, operator_norm=None, verbose=True):
        """Create a new instance

        Parameters
        ----------
        operator : `Operator`
            The operator to run tests on
        operator_norm : `float`
            The norm of the operator, used for error estimates
            can be estimated otherwise.
        """
        self.operator = operator
        self.operator_norm = operator_norm
        self.verbose = True

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
        for _, x in samples(self.operator.domain):
            result = self.operator(x)
            x_norm = x.norm()
            estimate = 0 if x_norm == 0 else result.norm() / x_norm

            operator_norm = max(operator_norm, estimate)

        print('Norm is at least: {}'.format(operator_norm))
        self.operator_norm = operator_norm
        return operator_norm

    def self_adjoint(self):
        """Verify (Ax, y) = (x, Ay)"""
        name = 'Verifying the identity (Ax, y) = (x, Ay)'

        left_inner_vals = []
        right_inner_vals = []

        with FailCounter(name, 'error = ||(Ax, y) - (x, Ay)|| / '
                         '||A|| ||x|| ||y||') as counter:
            for [name_x, x], [name_y, y] in samples(self.operator.domain,
                                                    self.operator.range):
                x_norm = x.norm()
                y_norm = y.norm()

                l_inner = self.operator(x).inner(y)
                r_inner = x.inner(self.operator(y))

                denom = self.operator_norm * x_norm * y_norm
                error = 0 if denom == 0 else abs(l_inner - r_inner) / denom

                if error > 0.00001:
                    counter.fail('x={:25s} y={:25s} : error={:6.5f}'
                                 ''.format(name_x, name_y, error))

                left_inner_vals.append(l_inner)
                right_inner_vals.append(r_inner)

        scale = np.polyfit(left_inner_vals, right_inner_vals, 1)[0]
        print('\nThe adjoint seems to be scaled according to:')
        print('(x, Ay) / (Ax, y) = {}. Should be 1.0'.format(scale))

    def _adjoint_definition(self):
        """Verify (Ax, y) = (x, A^T y)"""
        name = 'Verifying the identity (Ax, y) = (x, A^T y)'

        left_inner_vals = []
        right_inner_vals = []

        with FailCounter(name, 'error = ||(Ax, y) - (x, A^T y)|| / '
                         '||A|| ||x|| ||y||') as counter:
            for [name_x, x], [name_y, y] in samples(self.operator.domain,
                                                    self.operator.range):
                x_norm = x.norm()
                y_norm = y.norm()

                l_inner = self.operator(x).inner(y)
                r_inner = x.inner(self.operator.adjoint(y))

                denom = self.operator_norm * x_norm * y_norm
                error = 0 if denom == 0 else abs(l_inner - r_inner) / denom

                if error > 0.00001:
                    counter.fail('x={:25s} y={:25s} : error={:6.5f}'
                                 ''.format(name_x, name_y, error))

                left_inner_vals.append(l_inner)
                right_inner_vals.append(r_inner)

        scale = np.polyfit(left_inner_vals, right_inner_vals, 1)[0]
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

        name = '\nVerifying the identity Ax = (A^T)^T x'

        with FailCounter(name, 'error = ||Ax - (A^T)^T x|| /'
                         '||A|| ||x||') as counter:
            for [name_x, x] in self.operator.domain.examples:
                opx = self.operator(x)
                op_adj_adj_x = self.operator.adjoint.adjoint(x)

                denom = self.operator_norm * x.norm()
                if denom == 0:
                    error = 0
                else:
                    error = (opx - op_adj_adj_x).norm() / denom
                if error > 0.00001:
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

        print('\n== Verifying adjoint of operator ==\n')

        domain_range_ok = True
        if self.operator.domain != self.operator.adjoint.range:
            print('*** ERROR: A.domain != A.adjoint.range ***')
            domain_range_ok = False

        if self.operator.range != self.operator.adjoint.domain:
            print('*** ERROR: A.range != A.adjoint.domain ***')
            domain_range_ok = False

        if domain_range_ok:
            print('Domain and range of adjoint is OK.')
        else:
            print('Domain and range of adjoint not OK exiting.')
            return

        self._adjoint_definition()
        self._adjoint_of_adjoint()

    def _derivative_convergence(self):
        name = 'Testing derivative is linear approximation'

        with FailCounter(name, "error = "
                         "inf_c ||A(x+cp)-A(x)-A'(x)(cp)|| / c") as counter:
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

                    if err < 1e-4:
                        derivative_ok = True
                        break
                    else:
                        minerror = min(minerror, err)

                    c /= 2.0

                if not derivative_ok:
                    counter.fail('x={:15s} dx={:15s}, error={}'
                                 ''.format(name_x, name_dx, minerror))

    def derivative(self):
        """Verify that `Operator.derivative` works appropriately.

        References
        ----------
        Wikipedia article on `Derivative
        <https://en.wikipedia.org/wiki/Derivative>`_.
        Wikipedia article on `Frechet derivative
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

        if self.operator.is_linear and deriv is self.operator:
            print('A is linear and A.derivative == A')
            return

        self._derivative_convergence()

    def _scale_invariance(self):
        name = "Calculating invariance under scaling"

        # Test scaling
        with FailCounter(name, 'error = ||A(c*x)-c*A(x)|| / '
                         '|c| ||A|| ||x||') as counter:
            for [name_x, x], [_, scale] in samples(self.operator.domain,
                                                   self.operator.domain.field):
                opx = self.operator(x)
                scaled_opx = self.operator(scale * x)

                denom = self.operator_norm * scale * x.norm()
                error = (0 if denom == 0
                         else (scaled_opx - opx * scale).norm() / denom)

                if error > 0.00001:
                    counter.fail('x={:25s} scale={:7.2f} error={:6.5f}'
                                 ''.format(name_x, scale, error))

    def _addition_invariance(self):
        name = "Calculating invariance under addition"

        # Test addition
        with FailCounter(name, 'error = ||A(x+y) - A(x) - A(y)|| / '
                         '||A||(||x|| + ||y||)') as counter:
            for [name_x, x], [name_y, y] in samples(self.operator.domain,
                                                    self.operator.domain):
                opx = self.operator(x)
                opy = self.operator(y)
                opxy = self.operator(x + y)

                denom = self.operator_norm * (x.norm() + y.norm())
                error = (0 if denom == 0
                         else (opxy - opx - opy).norm() / denom)

                if error > 0.00001:
                    counter.fail('x={:25s} y={:25s} error={:6.5f}'
                                 ''.format(name_x, name_y, error))

    def linear(self):
        """Verify that the operator is actually linear."""
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
    X = odl.uniform_discr([0, 0], [1, 1], [3, 3])
    # Linear operator
    I = odl.IdentityOperator(X)
    OperatorTest(I).run_tests()

    # Nonlinear operator op(x) = x**4
    op = odl.PowerOperator(X, 4)
    OperatorTest(op).run_tests()
