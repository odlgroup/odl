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

import numpy as np

from odl import LinearOperator
from odl.diagnostics.examples import scalar_examples, vector_examples
from odl.util.testutils import FailCounter


__all__ = ('OperatorTest',)


class OperatorTest(object):
    def __init__(self, operator, operator_norm=None):
        self.operator = operator
        self.operator_norm = operator_norm

    def norm(self):
        print('\n== Calculating operator norm ==\n')

        operator_norm = 0.0
        for [name, vec] in vector_examples(self.operator.domain):
            result = self.operator(vec)
            vecnorm = vec.norm()
            estimate = 0 if vecnorm == 0 else result.norm() / vecnorm

            operator_norm = max(operator_norm, estimate)

        print('Norm is at least: {}'.format(operator_norm))
        self.operator_norm = operator_norm
        return operator_norm

    def _adjoint_definition(self):
        print('\nVerifying the identity (Ax, y) = (x, A^T y)')

        x = []
        y = []

        with FailCounter('error = ||(Ax, y) - (x, A^T y)|| / '
                         '||A|| ||x|| ||y||') as counter:
            for [name_dom, vec_dom] in vector_examples(self.operator.domain):
                vec_dom_norm = vec_dom.norm()
                for [name_ran, vec_ran] in vector_examples(self.operator.range):
                    vec_ran_norm = vec_ran.norm()

                    Axy = self.operator(vec_dom).inner(vec_ran)
                    xAty = vec_dom.inner(self.operator.adjoint(vec_ran))

                    denom = self.operator_norm * vec_dom_norm * vec_ran_norm
                    error = 0 if denom == 0 else abs(Axy-xAty)/denom

                    if error > 0.00001:
                        print('x={:25s} y={:25s} : error={:6.5f}'
                              ''.format(name_dom, name_ran, error))
                        counter.fail()

                    x.append(Axy)
                    y.append(xAty)

        scale = np.polyfit(x, y, 1)[0]
        print('\nThe adjoint seems to be scaled according to:')
        print('(x, A^T y) / (Ax, y) = {}. Should be 1.0'.format(scale))

    def _adjoint_of_adjoint(self):
        # Verify (A^*)^* = A
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
            for [name, vec] in vector_examples(self.operator.domain):
                A_result = self.operator(vec)
                ATT_result = self.operator.adjoint.adjoint(vec)

                denom = self.operator_norm * vec.norm()
                error = 0 if denom == 0 else (A_result-ATT_result).norm()/denom
                if error > 0.00001:
                    print('x={:25s} : error={:6.5f}'.format(name, error))
                    counter.fail()

    def adjoint(self):
        """Verify that the adjoint works appropriately."""
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
            for [name_x, x] in vector_examples(self.operator.domain):
                deriv = self.operator.derivative(x)
                opx = self.operator(x)
                for [name_dx, dx] in vector_examples(self.operator.domain):
                    exact_step = self.operator(x+dx*step)-opx
                    expected_step = deriv(dx*step)
                    denom = step * dx.norm() * self.operator_norm
                    error = (0 if denom == 0
                             else (exact_step-expected_step).norm() / denom)

                    if error > 0.00001:
                        print('x={:15s} dx={:15s} c={}: error={:6.5f}'
                              ''.format(name_x, name_dx, step, error))
                        counter.fail()

    def derivative(self, step=0.0001):
        """Verify that the derivative works appropriately."""

        print('\n==Verifying derivative of operator ==')
        try:
            deriv = self.operator.derivative(self.operator.domain.zero())

            if not isinstance(deriv, LinearOperator):
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
            for [name_x, x] in vector_examples(self.operator.domain):
                opx = self.operator(x)
                for scale in scalar_examples(self.operator.domain):
                    scaled_opx = self.operator(scale*x)

                    denom = self.operator_norm * scale * x.norm()
                    error = (0 if denom == 0
                             else (scaled_opx - opx * scale).norm() / denom)

                    if error > 0.00001:
                        print('x={:25s} scale={:7.2f} error={:6.5f}'
                              ''.format(name_x, scale, error))
                        counter.fail()

    def _addition_invariance(self):
        print("\nCalculating invariance under addition")

        # Test addition
        with FailCounter('error = ||A(x+y) - A(x) - A(y)|| / '
                         '||A||(||x|| + ||y||)') as counter:
            for [name_x, x] in vector_examples(self.operator.domain):
                opx = self.operator(x)
                for [name_y, y] in vector_examples(self.operator.domain):
                    opy = self.operator(y)
                    opxy = self.operator(x+y)

                    denom = self.operator_norm * (x.norm() + y.norm())
                    error = (0 if denom == 0
                             else (opxy - opx - opy).norm() / denom)

                    if error > 0.00001:
                        print('x={:25s} y={:25s} error={:6.5f}'
                              ''.format(name_x, name_y, error))
                        counter.fail()

    def linear(self):
        """ Verifies that the operator is actually linear
        """
        if not isinstance(self.operator, LinearOperator):
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
        """Runs all tests on this operator
        """
        print('\n== RUNNING ALL TESTS ==\n')
        print('Operator = {}'.format(self.operator))

        self.norm()

        if isinstance(self.operator, LinearOperator):
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
