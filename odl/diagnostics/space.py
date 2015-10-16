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

from odl.set.sets import Set
from odl.diagnostics.examples import samples
from odl.util.testutils import FailCounter
from copy import copy, deepcopy

__all__ = ('SpaceTest',)


def _apprimately_equal(x, y, eps):
    if x.space != y.space:
        return False

    if x is y:
        return True

    try:
        return x.dist(y) < eps
    except NotImplementedError:
        try:
            return x == y
        except NotImplementedError:
            return False


class SpaceTest(object):
    def __init__(self, space, eps=0.00001):
        self.space = space
        self.eps = eps

    def element(self):
        print('\n== Verifying element method ==\n')

        try:
            el = self.space.element()
        except NotImplementedError:
            print('*** element failed ***')
            return

        if el not in self.space:
            print('*** space.element() not in space ***')
        else:
            print('space.element() OK')

    def field(self):
        print('\n== Verifying field property ==\n')

        try:
            field = self.space.field
        except NotImplementedError:
            print('*** field failed ***')
            return

        if not isinstance(field, Set):
            print('*** space.element() not in space ***')
            return

        # Zero
        try:
            zero = field.element(0)
        except NotImplementedError:
            print('*** field.element(0) failed ***')
            return

        if not zero == 0:
            print('*** field.element(0) != 0 ***')

        if not zero == 0.0:
            print('*** field.element(0) != 0.0 ***')

        # one
        try:
            one = field.element(1)
        except NotImplementedError:
            print('*** field.element(1) failed ***')
            return

        if not one == 1:
            print('*** field.element(1) != 1 ***')

        if not one == 1.0:
            print('*** field.element(1) != 1.0 ***')

        # minus one
        try:
            minus_one = field.element(-1)
        except NotImplementedError:
            print('*** field.element(-1) failed ***')
            return

        if not minus_one == -1:
            print('*** field.element(-1) != -1 ***')

        if not minus_one == -1.0:
            print('*** field.element(-1) != -1.0 ***')

    def _associativity_of_addition(self):
        print('\nAssociativity of addition, '
              'x + (y + z) = (x + y) + z')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                ok = _apprimately_equal(x + (y + z), (x + y) + z, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def _commutativity_of_addition(self):
        print('\nCommutativity of addition, x + y = y + x')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                ok = _apprimately_equal(x + y, y + x, self.eps)
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
                ok = _apprimately_equal(x + zero, x, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _inverse_element_of_addition(self):
        print('\nInverse element of addition, x + (-x) = 0')
        zero = self.space.zero()

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                ok = _apprimately_equal(x + (-x), zero, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _commutativity_of_scalar_mult(self):
        print('\nCommutativity of scalar multiplication, '
              'a * (b * x) = (a * b) * x')

        with FailCounter() as counter:
            for [n_x, x], a, b in samples(self.space,
                                          self.space.field,
                                          self.space.field):
                ok = _apprimately_equal(a * (b * x), (a * b) * x, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s}, a={}, b={}'
                                 ''.format(n_x, a, b))

    def _identity_of_mult(self):
        print('\nIdentity element of multiplication, 1 * x = x')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                ok = _apprimately_equal(1 * x, x, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _distributivity_of_mult_vector(self):
        print('\nDistributivity of multiplication wrt vector add, '
              'a * (x + y) = a * x + a * y')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], a in samples(self.space,
                                                 self.space,
                                                 self.space.field):
                ok = _apprimately_equal(a * (x + y), a * x + a * y, self.eps)
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
                ok = _apprimately_equal((a + b) * x, a * x + b * x, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s}, a={}, b={}'
                                 ''.format(n_x, a, b))

    def _subtraction(self):
        print('\nSubtraction, x - y = x + (-1 * y)')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                ok = (_apprimately_equal(x - y, x + (-1 * y), self.eps) and
                      _apprimately_equal(x - y, x + (-y), self.eps))
                if not ok:
                    counter.fail('failed with x={:25s}, y={:25s}'
                                 ''.format(n_x, n_y))

    def _division(self):
        print('\nDivision, x / a = x * (1/a)')

        with FailCounter() as counter:
            for [n_x, x], a in samples(self.space,
                                       self.space.field):
                if a != 0:
                    ok = _apprimately_equal(x / a, x * (1.0/a), self.eps)
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

    def _inner_linear_scalar(self):
        print('\nLinearity scalar, (a*x, y) = a*(x, y)')

        with FailCounter('error = |(a*x, y) - a*(x, y)|') as counter:
            for [n_x, x], [n_y, y], a in samples(self.space,
                                                 self.space,
                                                 self.space.field):
                error = abs((a*x).inner(y) - a * x.inner(y))
                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}, a={}: error={}'
                                 ''.format(n_x, n_y, a, error))

    def _inner_conjugate_symmetry(self):
        print('\nConjugate symmetry, (x, y) = (y, x).conj()')

        with FailCounter('error = |(x, y) - (y, x).conj()|') as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                error = abs((x).inner(y) - y.inner(x).conjugate())
                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def _inner_linear_sum(self):
        print('\nLinearity sum, (x+y, z) = (x, z) + (y, z)')

        with FailCounter('error = |(x+y, z) - ((x, z)+(y, z))|') as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                error = abs((x+y).inner(z) - (x.inner(z) + y.inner(z)))
                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}, z={:25s}: error={}'
                                 ''.format(n_x, n_y, n_z, error))

    def _inner_positive(self):
        print('\nPositivity, (x, x) >= 0')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                inner = x.inner(x)

                if abs(inner.imag) > self.eps:
                    counter.fail('(x, x).imag != 0, x={:25s}'
                                 ''.format(n_x))

                if n_x == 'Zero' and inner.real != 0:
                    counter.fail('(0, 0) != 0.0, x={:25s}: (0, 0)={}'
                                 ''.format(n_x, inner))

                elif n_x != 'Zero' and inner.real <= 0:
                    counter.fail('(x, x) <= 0,   x={:25s}: (x, x)={}'
                                 ''.format(n_x, inner))

    def inner(self):
        print('\n== Verifying inner product ==\n')

        try:
            zero = self.space.zero()
            zero.inner(zero)
        except NotImplementedError:
            print('Space has no inner product')
            return

        self._inner_conjugate_symmetry()
        self._inner_linear_scalar()
        self._inner_linear_sum()
        self._inner_positive()

    def _norm_positive(self):
        print('\nPositivity, ||x|| >= 0')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                norm = x.norm()

                if n_x == 'Zero' and norm != 0:
                    counter.fail('||0|| != 0.0, x={:25s}: ||x||={}'
                                 ''.format(n_x, norm))

                elif n_x != 'Zero' and norm <= 0:
                    counter.fail('||x|| <= 0,   x={:25s}: ||x||={}'
                                 ''.format(n_x, norm))

    def _norm_subadditive(self):
        print('\nSub-additivity, ||x+y|| <= ||x|| + ||y||')

        with FailCounter('error = ||x+y|| - ||x|| + ||y||') as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                norm_x = x.norm()
                norm_y = y.norm()
                norm_xy = (x + y).norm()

                error = norm_xy - norm_x - norm_y

                if error > 0:
                    counter.fail('x={:25s} y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def _norm_homogeneity(self):
        print('\nHomogeneity, ||a*x|| = |a| ||x||')

        with FailCounter('error = abs(||a*x|| - |a| ||x||)') as counter:
            for [name, vec], scalar in samples(self.space,
                                               self.space.field):
                error = abs((scalar * vec).norm() - abs(scalar) * vec.norm())
                if error > self.eps:
                    counter.fail('x={:25s} a={}: ||x||={}'
                                 ''.format(name, scalar, error))

    def _norm_inner_compatible(self):
        print('\nInner compatibility, ||x||^2 = (x, x)')

        try:
            zero = self.space.zero()
            zero.inner(zero)
        except NotImplementedError:
            print('Space is not a inner product space')
            return

        with FailCounter('error = | ||x||^2 = (x, x) |') as counter:
            for [n_x, x] in samples(self.space):
                error = abs(x.norm()**2 - x.inner(x))

                if error > self.eps:
                    counter.fail('x={:25s}: error={}'
                                 ''.format(n_x, error))

    def norm(self):
        """Run all norm-related tests on this space."""
        print('\n== Verifying norm ==\n')

        try:
            self.space.zero().norm()
        except NotImplementedError:
            print('Space is not normed')
            return

        self._norm_positive()
        self._norm_subadditive()
        self._norm_homogeneity()
        self._norm_inner_compatible()

    def _dist_positivity(self):
        print('\nPositivity, d(x, y) >= 0')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                dist = x.dist(y)

                if n_x == n_y and dist != 0:
                    counter.fail('d(x, x) != 0.0, x={:25s}: dist={}'
                                 ''.format(n_x, dist))
                elif n_x != n_y and dist <= 0:
                    counter.fail('d(x, y) <= 0,   x={:25s} y={:25s}: dist={}'
                                 ''.format(n_x, n_y, dist))

    def _dist_symmetric(self):
        print('\nSymmetry, d(x, y) = d(y, x)')

        with FailCounter('error = |d(x, y) - d(y, x)|') as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                dist_1 = x.dist(y)
                dist_2 = y.dist(x)
                error = abs(dist_1 - dist_2)

                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def _dist_subadditive(self):
        print('\nSub-additivity, d(x, y) = d(y, x)')

        with FailCounter('error = d(x,z) - (d(x, y) + d(y, z))') as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                dxz = x.dist(z)
                dxy = x.dist(y)
                dyz = y.dist(z)
                error = dxz - (dxy + dyz)

                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}, z={:25s}: error={}'
                                 ''.format(n_x, n_y, n_z, error))

    def _dist_norm_compatible(self):
        print('\nNorm compatibility, d(x, y) = ||x-y||')

        try:
            self.space.zero().norm()
        except NotImplementedError:
            print('Space is not normed')
            return

        with FailCounter('error = |d(x, y) - ||x-y|| |') as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                error = abs(x.dist(y) - (x-y).norm())

                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def dist(self):
        print('\n== Verifying dist ==\n')

        try:
            zero = self.space.zero()
            self.space.dist(zero, zero)
        except NotImplementedError:
            print('Space is not metric')
            return

        self._dist_positivity()
        self._dist_symmetric()
        self._dist_subadditive()
        self._dist_norm_compatible()

    def _multiply_zero(self):
        print('\nMultiplication by zero, x * 0 = 0')

        zero = self.space.zero()

        with FailCounter('error = ||x*0||') as counter:
            for [n_x, x] in samples(self.space):
                error = (zero * x).norm()

                if error > self.eps:
                    counter.fail('x={:25s},: error={}'
                                 ''.format(n_x, error))

    def _multiply_commutative(self):
        print('\nMultiplication commutative, x * y = y * x')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                ok = _apprimately_equal(x * y, y * x, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s} y={:25s}'
                                 ''.format(n_x, n_y))

    def _multiply_associative(self):
        print('\nMultiplication associative, x * (y * z) = (x * y) * z')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                ok = _apprimately_equal(x * (y * z), (x * y) * z, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def _multiply_distributive_scalar(self):
        print('\nMultiplication associative wrt scal, a * (y * z) = '
              '(a * y) * z = y * (a * z)')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], a in samples(self.space,
                                                 self.space,
                                                 self.space.field):
                ok = _apprimately_equal(a * (x * y), (a * x) * y, self.eps)
                ok = _apprimately_equal(a * (x * y), x * (a * y), self.eps)
                if not ok:
                    counter.fail('failed with x={:25s} y={:25s} a={}'
                                 ''.format(n_x, n_y, a))

    def _multiply_distributive_vec(self):
        print('\nMultiplication associative wrt vec, x * (y * z) = '
              '(x * y) * z')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                ok = _apprimately_equal(x * (y * z), (x * y) * z, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def multiply(self):
        print('\n== Verifying multiplication ==\n')

        try:
            zero = self.space.zero()
            self.space.multiply(zero, zero)
        except NotImplementedError:
            print('Space is not a algebra')
            return

        self._multiply_zero()
        self._multiply_commutative()
        self._multiply_associative()
        self._multiply_distributive_scalar()
        self._multiply_distributive_vec()

    def equals(self):
        print('\n== Verifying __eq__ ==\n')

        if not self.space == self.space:
            print('** space == space failed ***')

        if self.space != self.space:
            print('** not space != space failed***')

        if self.space != copy(self.space):
            print('** space == copy(space) failed***')

        if self.space != deepcopy(self.space):
            print('** space == deepcopy(space) failed***')

        with FailCounter('Space equal to non-space') as counter:
            for obj in [[1, 2], list(), tuple(), dict(), 5.0]:
                if self.space == obj:
                    counter.fail('space == obj,  with obj={}'
                                 ''.format(obj))

                if not self.space != obj:
                    counter.fail('not space != obj,  with obj={}'
                                 ''.format(obj))

    def contains(self):
        print('\n== Verifying __contains__ ==\n')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                if x not in self.space:
                    counter.fail('x not in space,  with x={}'
                                 ''.format(n_x))

                if x not in self.space:
                    counter.fail('not x in space,  with x={}'
                                 ''.format(n_x))

            for obj in [[1, 2], list(), tuple(), dict(), 5.0]:
                if obj in self.space:
                    counter.fail('obj in space,  with obj={}'
                                 ''.format(obj))

                if not obj not in self.space:
                    counter.fail('not obj not in space,  with obj={}'
                                 ''.format(obj))

    def _vector_assign(self):
        print('\nVector.assign()')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                x.assign(y)
                ok = _apprimately_equal(x, y, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s} y={:25s}'
                                 ''.format(n_x, n_y))

    def _vector_copy(self):
        print('\nVector.copy()')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                # equal after copy
                y = x.copy()
                ok = _apprimately_equal(x, y, self.eps)
                if not ok:
                    counter.fail('failed with x={:s5s}'
                                 ''.format(n_x))

                # modify y, x stays the same
                y *= 2.0
                ok = n_x == 'Zero' or not _apprimately_equal(x, y, self.eps)
                if not ok:
                    counter.fail('modified y, x changed with x={:25s}'
                                 ''.format(n_x))

    def _vector_set_zero(self):
        print('\nVector.set_zero()')

        zero = self.space.zero()
        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                x.set_zero()
                ok = _apprimately_equal(x, zero, self.eps)
                if not ok:
                    counter.fail('failed with x={:25s}'
                                 ''.format(n_x))

    def _vector_equals(self):
        print('\nVector.__eq__()')

        try:
            zero = self.space.zero()
            zero == zero
        except NotImplementedError:
            print('Vector has no __eq__')
            return

        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                if n_x == n_y:
                    if not x == y:
                        counter.fail('failed x == x with x={:25s}'
                                     ''.format(n_x))

                    if x != y:
                        counter.fail('failed not x != x with x={:25s}'
                                     ''.format(n_x))
                else:
                    if x == y:
                        counter.fail('failed not x == y with x={:25s}, '
                                     'x={:25s}'.format(n_x, n_y))

                    if not x != y:
                        counter.fail('failed x != y with x={:25s}, x={:25s}'
                                     ''.format(n_x, n_y))

    def _vector_space(self):
        print('\nVector.space')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                if not x.space == self.space:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def vector(self):
        print('\n== Verifying Vector ==\n')

        self._vector_assign()
        self._vector_copy()
        self._vector_set_zero()
        self._vector_equals()
        self._vector_space()

    def run_tests(self):
        """Run all tests on this space."""
        print('\n== RUNNING ALL TESTS ==\n')
        print('Space = {}'.format(self.space))

        self.field()
        self.element()
        self.linearity()
        self.element()
        self.inner()
        self.norm()
        self.dist()
        self.multiply()
        self.equals()
        self.contains()
        self.vector()

    def __str__(self):
        return 'SpaceTest({})'.format(self.space)

    def __repr__(self):
        return 'SpaceTest({!r})'.format(self.space)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
