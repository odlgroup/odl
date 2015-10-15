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

from odl.diagnostics.examples import samples
from odl.util.testutils import FailCounter

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
        print('\nAssociativity of addition, '
              'x + (y + z) = (x + y) + z')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                ok = _apprimately_equal(x + (y + z), (x + y) + z)
                if not ok:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                    ''.format(n_x, n_y, n_z))

    def _commutativity_of_addition(self):
        print('\nCommutativity of addition, x + y = y + x')
        
        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                ok = _apprimately_equal(x + y, y + x)
                if not ok:
                    counter.fail('failed with x={:25s} y={:25s}'
                                    ''.format(n_x, n_y))

    def _identity_of_addition(self):
        print('\nIdentity element of addition, x + 0 = x')

        try:
            zero = self.space.zero()
        except:
            print('*** SPACE HAS NO ZERO VECTOR ***')
            
        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                ok = _apprimately_equal(x + zero, x)
                if not ok:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _inverse_element_of_addition(self):
        print('\nInverse element of addition, x + (-x) = 0')
        zero = self.space.zero()
        
        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                ok = _apprimately_equal(x + (-x), zero)
                if not ok:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _commutativity_of_scalar_mult(self):
        print('\nCommutativity of scalar multiplication, '
              'a * (b * x) = (a * b) * x')
        
        with FailCounter() as counter:
            for [n_x, x], a, b in samples(self.space,
                                          self.space.field,
                                          self.space.field):
                ok = _apprimately_equal(a * (b * x), (a * b) * x)
                if not ok:
                    counter.fail('failed with x={:25s}, a={}, b={}'
                                    ''.format(n_x, a, b))

    def _identity_of_mult(self):
        print('\nIdentity element of multiplication, 1 * x = x')
        
        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                ok = _apprimately_equal(1 * x, x)
                if not ok:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _distributivity_of_mult_vector(self):
        print('\nDistributivity of multiplication wrt vector add, '
              'a * (x + y) = a * x + a * y')
        
        with FailCounter() as counter:
            for [n_x, x], [n_y, y], a in samples(self.space,
                                                 self.space,
                                                 self.space.field):
                ok = _apprimately_equal(a * (x + y), a * x + a * y)
                if not ok:
                    counter.fail('failed with x={:25s}, y={:25s}, a={}'
                                    ''.format(n_x, n_y, a))

    def _distributivity_of_mult_scalar(self):
        print('\nDistributivity of multiplication wrt scalar add, '
              '(a + b) * x = a * x + b * x')
        
        with FailCounter() as counter:
            for [n_x, x], a, b in samples(self.space,
                                          self.space.field,
                                          self.space.field):
                ok = _apprimately_equal((a + b) * x, a * x + b * x)
                if not ok:
                    counter.fail('failed with x={:25s}, a={}, b={}'
                                    ''.format(n_x, a, b))

    def _subtraction(self):
        print('\nSubtraction, x - y = x + (-1 * y)')
        
        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                ok = (_apprimately_equal(x - y, x + (-1 * y)) and
                        _apprimately_equal(x - y, x + (-y)))
                if not ok:
                    counter.fail('failed with x={:25s}, y={:25s}'
                                    ''.format(n-x, n_y))

    def _division(self):
        print('\nDivision, x / a = x * (1/a)')
        
        with FailCounter() as counter:
            for [n_x, x], a in samples(self.space,
                                       self.space.field):
                if a != 0:
                    ok = _apprimately_equal(x / a, x * (1.0/a))
                    if not ok:
                        counter.fail('failed with x={:25s}, a={}'
                                        ''.format(n_x, a))

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

    def _dist_non_negative(self):
        pass

    def _dist_symmetric(self):
        pass

    def _dist_subadditive(self):
        pass

    def _norm_positive(self):
        print('\ntesting positivity, ||x|| >= 0\n')
        
        with FailCounter('error = -||x||') as counter:
            for [name, vec] in samples(self.space):
                norm = vec.norm()

                if norm < 0 or (norm == 0 and name != 'Zero'):
                    counter.fail('x={:25s} : ||x||={}'
                                 ''.format(name, norm))

    def _norm_subadditive(self):
        print('\ntesting subadditivity, ||x+y|| <= ||x|| + ||y||\n')
        
        with FailCounter('error = ||x+y|| - ||x|| + ||y||') as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                norm_x = x.norm()
                norm_y = y.norm()
                norm_xy = (x + y).norm()

                error = norm_xy - norm_x - norm_y

                if error > 0:
                    counter.fail('x={:25s} y={:25s}: error={}'
                                 ''.format(name_x, name_y, error))

    def _norm_homogeneity(self):
        print('\ntesting homogeneity, ||a*x|| = |a| ||x||\n')
        print('error = abs(||a*x|| - |a| ||x||)')
        
        with FailCounter('error = abs(||a*x|| - |a| ||x||)') as counter:
            for [name, vec], scalar in samples(self.space,
                                               self.space.field):
                error = abs((scalar * vec).norm() - abs(scalar) * vec.norm())
                if error > 0.00001:
                    counter.fail('x={:25s} a={}: ||x||={}'
                                    ''.format(name, scalar, error))

    def norm(self):
        """Run all norm-related tests on this space."""
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
        """Run all tests on this space."""
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
