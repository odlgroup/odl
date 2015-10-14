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

from odl.diagnostics.examples import scalar_examples, vector_examples


__all__ = ('SpaceTest',)


def _apprimately_equal(x, y):
    if x.space != y.space:
        return False

    if x is y:
        return True

    try:
        return x.dist(y) < 0.0001
    except NotImplementedError:
        try:
            return x == y
        except NotImplementedError:
            return False


class SpaceTest(object):
    def __init__(self, space):
        self.space = space

    def _associativity_of_addition(self):
        print('\ntesting Associativity of addition, '
              'x + (y + z) = (x + y) + z\n')

        for [n_x, x] in vector_examples(self.space):
            for [n_y, y] in vector_examples(self.space):
                for [n_z, z] in vector_examples(self.space):
                    ok = _apprimately_equal(x + (y + z), (x + y) + z)
                    if not ok:
                        print('failed with x={:25s} y={:25s} z={:25s}'
                              ''.format(n_x, n_y, n_z))

    def _commutativity_of_addition(self):
        print('\ntesting Commutativity of addition, x + y = y + x\n')

        for [n_x, x] in vector_examples(self.space):
            for [n_y, y] in vector_examples(self.space):
                ok = _apprimately_equal(x + y, y + x)
                if not ok:
                    print('failed with x={:25s} y={:25s}'.format(n_x, n_y))

    def _identity_of_addition(self):
        print('\ntesting Identity element of addition, x + 0 = x\n')

        try:
            zero = self.space.zero()
        except:
            print('*** SPACE HAS NO ZERO VECTOR ***')

        for [n_x, x] in vector_examples(self.space):
            ok = _apprimately_equal(x + zero, x)
            if not ok:
                print('failed with x={:25s}'.format(n_x))

    def _inverse_element_of_addition(self):
        print('\ntesting Inverse element of addition, x + (-x) = 0\n')
        zero = self.space.zero()

        for [n_x, x] in vector_examples(self.space):
            ok = _apprimately_equal(x + (-x), zero)
            if not ok:
                print('failed with x={:25s}'.format(n_x))

    def _commutativity_of_scalar_mult(self):
        print('\ntesting Commutativity of scalar multiplication, '
              'a * (b * x) = (a * b) * x\n')

        for [n_x, x] in vector_examples(self.space):
            for a in scalar_examples(self.space):
                for b in scalar_examples(self.space):
                    ok = _apprimately_equal(a * (b * x), (a * b) * x)
                    if not ok:
                        print('failed with x={:25s}, a={}, b={}'
                              ''.format(n_x, a, b))

    def _identity_of_mult(self):
        print('\ntesting Identity element of multiplication, 1 * x = x\n')

        for [n_x, x] in vector_examples(self.space):
            ok = _apprimately_equal(1 * x, x)
            if not ok:
                print('failed with x={:25s}'.format(n_x))

    def _distributivity_of_mult_vector(self):
        print('\ntesting Distributivity of multiplication wrt vector add, '
              'a * (x + y) = a * x + a * y\n')

        for a in scalar_examples(self.space):
            for [n_x, x] in vector_examples(self.space):
                for [n_y, y] in vector_examples(self.space):
                    ok = _apprimately_equal(a * (x + y), a * x + a * y)
                    if not ok:
                        print('failed with x={:25s}, y={:25s}, a={}'
                              ''.format(n_x, n_y, a))

    def _distributivity_of_mult_scalar(self):
        print('\ntesting Distributivity of multiplication wrt scalar add, '
              '(a + b) * x = a * x + b * x\n')

        for a in scalar_examples(self.space):
            for b in scalar_examples(self.space):
                for [n_x, x] in vector_examples(self.space):
                    ok = _apprimately_equal((a + b) * x, a * x + b * x)
                    if not ok:
                        print('failed with x={:25s}, a={}, b={}'
                              ''.format(n_x, a, b))

    def _subtraction(self):
        print('\ntesting Subtraction, x - y = x + (-1 * y)\n')

        for [n_x, x] in vector_examples(self.space):
            for [n_y, y] in vector_examples(self.space):
                ok = (_apprimately_equal(x - y, x + (-1 * y)) and
                      _apprimately_equal(x - y, x + (-y)))
                if not ok:
                    print('failed with x={:25s}, y={:25s}'.format(n_x, n_y))

    def _division(self):
        print('\ntesting Division, x / a = x * (1/a) \n')

        for [n_x, x] in vector_examples(self.space):
            for a in scalar_examples(self.space):
                if a != 0:
                    ok = _apprimately_equal(x / a, x * (1.0/a))
                    if not ok:
                        print('failed with x={:25s}, a={}'.format(n_x, a))

    def linearity(self):
        print('\n== Verifying linear space properties ==\n')

        self._associativity_of_addition()
        self._commutativity_of_addition()
        self._identity_of_addition()
        self._inverse_element_of_addition()
        self._commutativity_of_scalar_mult()
        self._identity_of_mult()
        self._distributivity_of_mult_vector()
        self._distributivity_of_mult_scalar()
        self._subtraction()
        self._division()

    def _norm_positive(self):
        print('\ntesting positivity, ||x|| >= 0\n')
        print('error = -||x||')

        # TODO: assert ||x|| = 0 iff x = 0

        num_failed = 0
        for [name, vec] in vector_examples(self.space):
            norm = vec.norm()

            if norm < 0 or (norm == 0 and name != 'Zero'):
                print('x={:25s} : ||x||={}'
                      ''.format(name, norm))
                num_failed += 1

        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:
            print('*** FAILED {} TEST CASES ***'.format(num_failed))

    def _norm_subadditive(self):
        print('\ntesting subadditivity, ||x+y|| <= ||x|| + ||y||\n')
        print('error = ||x+y|| - ||x|| + ||y||')

        num_failed = 0
        for [name_x, vec_x] in vector_examples(self.space):
            norm_x = vec_x.norm()
            for [name_y, vec_y] in vector_examples(self.space):
                norm_xy = (vec_x + vec_y).norm()
                norm_y = vec_y.norm()

            error = norm_xy - norm_x - norm_y

            if error > 0:
                print('x={:25s} y={:25s}: error={}'
                      ''.format(name_x, name_y, error))
                num_failed += 1

        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:
            print('*** FAILED {} TEST CASES ***'.format(num_failed))

    def _norm_homogeneity(self):
        print('\ntesting homogeneity, ||a*x|| = |a| ||x||\n')
        print('error = abs(||a*x|| - |a| ||x||)')

        num_failed = 0
        for [name, vec] in vector_examples(self.space):
            for scalar in scalar_examples(self.space):
                error = abs((scalar * vec).norm() - abs(scalar) * vec.norm())
                if error > 0.00001:
                    print('x={:25s} a={}: ||x||={}'
                          ''.format(name, scalar, error))
                    num_failed += 1

        if num_failed == 0:
            print('error = 0.0 for all test cases')
        else:
            print('*** FAILED {} TEST CASES ***'.format(num_failed))

    def norm(self):
        print('\n== Verifying norm ==\n')

        try:
            self.space.zero().norm()
        except NotImplementedError:
            print('Space is not normed')
            return

        print('testing ||0|| = {}. Expected 0.0'
              ''.format(self.space.zero().norm()))
        self._norm_positive()
        self._norm_subadditive()
        self._norm_homogeneity()

    def run_tests(self):
        """Runs all tests on this operator
        """
        print('\n== RUNNING ALL TESTS ==\n')
        print('Space = {}'.format(self.space))

        self.linearity()
        self.norm()

    def __str__(self):
        return 'SpaceTest({})'.format(self.space)

    def __repr__(self):
        return 'SpaceTest({!r})'.format(self.space)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
