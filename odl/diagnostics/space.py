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

"""Standardized tests for `LinearSpace`'s."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

from copy import copy, deepcopy

from odl.set.sets import Field
from odl.diagnostics.examples import samples
from odl.util.testutils import FailCounter


__all__ = ('SpaceTest',)


def _approx_equal(x, y, eps):
    """Test if vectors ``x`` and ``y`` are approximately equal.

    ``eps`` is a given absolute tolerance.
    """
    if x.space != y.space:
        return False

    if x is y:
        return True

    try:
        return x.dist(y) <= eps
    except NotImplementedError:
        try:
            return x == y
        except NotImplementedError:
            return False


class SpaceTest(object):

    """Automated tests for `LinearSpace` instances.

    This class allows users to automatically test various
    features of an ``LinearSpace`` such as linearity and the
    various operators.
    """

    def __init__(self, space, eps=0.00001):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            The space that should be tested
        eps : `float`, optional
            Precision of the tests.
        """
        self.space = space
        self.eps = eps

    def element(self):
        """Verify `LinearSpace.element`"""
        print('\n== Verifying element method ==\n')

        try:
            elem = self.space.element()
        except NotImplementedError:
            print('*** element failed ***')
            return

        if elem not in self.space:
            print('*** space.element() not in space ***')
        else:
            print('space.element() OK')

    def field(self):
        """Verify `LinearSpace.field`"""
        print('\n== Verifying field property ==\n')

        try:
            field = self.space.field
        except NotImplementedError:
            print('*** field failed ***')
            return

        if not isinstance(field, Field):
            print('*** space.field not a `Field` ***')
            return

        # Zero
        try:
            zero = field.element(0)
        except NotImplementedError:
            print('*** field.element(0) failed ***')
            return

        if zero != 0:
            print('*** field.element(0) != 0 ***')

        if zero != 0.0:
            print('*** field.element(0) != 0.0 ***')

        # one
        try:
            one = field.element(1)
        except NotImplementedError:
            print('*** field.element(1) failed ***')
            return

        if one != 1:
            print('*** field.element(1) != 1 ***')

        if one != 1.0:
            print('*** field.element(1) != 1.0 ***')

        # minus one
        try:
            minus_one = field.element(-1)
        except NotImplementedError:
            print('*** field.element(-1) failed ***')
            return

        if minus_one != -1:
            print('*** field.element(-1) != -1 ***')

        if minus_one != -1.0:
            print('*** field.element(-1) != -1.0 ***')

    def _associativity_of_addition(self):
        """Check addition associativity."""
        print('\nAssociativity of addition, '
              'x + (y + z) = (x + y) + z')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                correct = _approx_equal(x + (y + z), (x + y) + z, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def _commutativity_of_addition(self):
        """Check addition commutativity."""
        print('\nCommutativity of addition, x + y = y + x')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                correct = _approx_equal(x + y, y + x, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s}'
                                 ''.format(n_x, n_y))

    def _identity_of_addition(self):
        """Check additional neutral element ('zero')."""
        print('\nIdentity element of addition, x + 0 = x')

        try:
            zero = self.space.zero()
        except (AttributeError, NotImplementedError):
            print('*** SPACE HAS NO ZERO VECTOR ***')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                correct = _approx_equal(x + zero, x, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _inverse_element_of_addition(self):
        """Check additional inverse."""
        print('\nInverse element of addition, x + (-x) = 0')
        zero = self.space.zero()

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                correct = _approx_equal(x + (-x), zero, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _commutativity_of_scalar_mult(self):
        """Check scalar multiplication commutativity."""
        print('\nCommutativity of scalar multiplication, '
              'a * (b * x) = (a * b) * x')

        with FailCounter() as counter:
            for [n_x, x], a, b in samples(self.space,
                                          self.space.field,
                                          self.space.field):
                correct = _approx_equal(a * (b * x), (a * b) * x, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s}, a={}, b={}'
                                 ''.format(n_x, a, b))

    def _identity_of_mult(self):
        """Check multiplicative neutral element ('one')."""
        print('\nIdentity element of multiplication, 1 * x = x')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                correct = _approx_equal(1 * x, x, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _distributivity_of_mult_vector(self):
        """Check vector multiplication distributivity."""
        print('\nDistributivity of multiplication wrt vector add, '
              'a * (x + y) = a * x + a * y')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], a in samples(self.space,
                                                 self.space,
                                                 self.space.field):
                correct = _approx_equal(a * (x + y), a * x + a * y, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s}, y={:25s}, a={}'
                                 ''.format(n_x, n_y, a))

    def _distributivity_of_mult_scalar(self):
        """Check scalar multiplication distributivity."""
        print('\nDistributivity of multiplication wrt scalar add, '
              '(a + b) * x = a * x + b * x')

        with FailCounter() as counter:
            for [n_x, x], a, b in samples(self.space,
                                          self.space.field,
                                          self.space.field):
                correct = _approx_equal((a + b) * x, a * x + b * x, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s}, a={}, b={}'
                                 ''.format(n_x, a, b))

    def _subtraction(self):
        """Check subtraction as addition of additive inverse."""
        print('\nSubtraction, x - y = x + (-1 * y)')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                correct = (_approx_equal(x - y, x + (-1 * y), self.eps) and
                           _approx_equal(x - y, x + (-y), self.eps))
                if not correct:
                    counter.fail('failed with x={:25s}, y={:25s}'
                                 ''.format(n_x, n_y))

    def _division(self):
        """Check scalar division as multiplication with mult. inverse."""
        print('\nDivision, x / a = x * (1/a)')

        with FailCounter() as counter:
            for [n_x, x], a in samples(self.space,
                                       self.space.field):
                if a != 0:
                    correct = _approx_equal(x / a, x * (1.0 / a), self.eps)
                    if not correct:
                        counter.fail('failed with x={:25s}, a={}'
                                     ''.format(n_x, a))

    def _lincomb_aliased(self):
        """Check several scenarios of aliased linear combination."""
        print('\nAliased input in lincomb')

        with FailCounter() as counter:
            for [n_x, x_in], [n_y, y] in samples(self.space, self.space):
                x = x_in.copy()
                x.lincomb(1, x, 1, y)
                correct = _approx_equal(x, x_in + y, self.eps)
                if not correct:
                    counter.fail('failed with x.lincomb(1, x, 1, y),'
                                 'x={:25s} y={:25s} '
                                 ''.format(n_x, n_y))

                x = x_in.copy()
                x.lincomb(1, x, 1, x)
                correct = _approx_equal(x, x_in + x_in, self.eps)
                if not correct:
                    counter.fail('failed with x.lincomb(1, x, 1, x),'
                                 'x={:25s} '
                                 ''.format(n_x))

    def _lincomb(self):
        """Check linear combination."""
        print('\nTesting lincomb')

        self._lincomb_aliased()

    def linearity(self):
        """Verify the linear space properties by examples.

        These properties include things such as associativity

        ``x + y = y + x``

        and identity of the `LinearSpace.zero` element

        ``x + 0 = x``

        References
        ----------
        Wikipedia article on `Vector space`_.

        .. _Vector space: https://en.wikipedia.org/wiki/Vector_space
        """
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
        self._lincomb()

    def _inner_linear_scalar(self):
        """Check homogeneity of the inner product in the first argument."""
        print('\nLinearity scalar, (a*x, y) = a*(x, y)')

        with FailCounter('error = |(a*x, y) - a*(x, y)|') as counter:
            for [n_x, x], [n_y, y], a in samples(self.space,
                                                 self.space,
                                                 self.space.field):
                error = abs((a * x).inner(y) - a * x.inner(y))
                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}, a={}: error={}'
                                 ''.format(n_x, n_y, a, error))

    def _inner_conjugate_symmetry(self):
        """Check conjugate symmetry of the inner product."""
        print('\nConjugate symmetry, (x, y) = (y, x).conj()')

        with FailCounter('error = |(x, y) - (y, x).conj()|') as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                error = abs((x).inner(y) - y.inner(x).conjugate())
                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def _inner_linear_sum(self):
        """Check additivity of the inner product in the first argument."""
        print('\nLinearity sum, (x+y, z) = (x, z) + (y, z)')

        with FailCounter('error = |(x+y, z) - ((x, z)+(y, z))|') as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                error = abs((x + y).inner(z) - (x.inner(z) + y.inner(z)))
                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}, z={:25s}: error={}'
                                 ''.format(n_x, n_y, n_z, error))

    def _inner_positive(self):
        """Check positive definiteness of the inner product."""
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
        """Verify `LinearSpace.inner`.

        The inner product satisfies properties such as

        conjugate symmetry
        ``(x, y) = (y, x)^*`` (^* complex conjugate)

        linearity
        ``(a * x, y) = a * (x, y)``
        ``(x + y, z) = (x, z) + (y, z)``

        positivity
        ``(x, x) >= 0``

        References
        ----------
        Wikipedia article on `inner product`_.

        .. _inner product: https://en.wikipedia.org/wiki/Inner_product_space
        """
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
        """Check nonnegativity of the norm."""
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
        """Check subadditivity of the norm."""
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
        """Check positive homogeneity of the norm."""
        print('\nHomogeneity, ||a*x|| = |a| ||x||')

        with FailCounter('error = abs(||a*x|| - |a| ||x||)') as counter:
            for [name, vec], scalar in samples(self.space,
                                               self.space.field):
                error = abs((scalar * vec).norm() - abs(scalar) * vec.norm())
                if error > self.eps:
                    counter.fail('x={:25s} a={}: ||x||={}'
                                 ''.format(name, scalar, error))

    def _norm_inner_compatible(self):
        """Check compatibility of norm and inner product."""
        print('\nInner compatibility, ||x||^2 = (x, x)')

        try:
            zero = self.space.zero()
            zero.inner(zero)
        except NotImplementedError:
            print('Space is not a inner product space')
            return

        with FailCounter('error = | ||x||^2 = (x, x) |') as counter:
            for [n_x, x] in samples(self.space):
                error = abs(x.norm() ** 2 - x.inner(x))

                if error > self.eps:
                    counter.fail('x={:25s}: error={}'
                                 ''.format(n_x, error))

    def norm(self):
        """Verify `LinearSpace.norm`.

        The norm satisfies properties

        linearity
        ``||a * x|| = |a| * ||x||``

        triangle inequality
        ``||x + y|| = ||x|| + ||y||``

        separation
        ``||x|| = 0`` iff ``x = 0``

        positivity
        ``||x|| >= 0``

        References
        ----------
        Wikipedia article on norm_.

        .. _norm: https://en.wikipedia.org/wiki/Norm_(mathematics)
        """
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
        """Check nonnegativity of the distance."""
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
        """Check symmetry of the distance."""
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
        """Check subadditivity of the distance."""
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
        """Check compatibility of distance and norm."""
        print('\nNorm compatibility, d(x, y) = ||x-y||')

        try:
            self.space.zero().norm()
        except NotImplementedError:
            print('Space is not normed')
            return

        with FailCounter('error = |d(x, y) - ||x-y|| |') as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                error = abs(x.dist(y) - (x - y).norm())

                if error > self.eps:
                    counter.fail('x={:25s}, y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def dist(self):
        """Verify `LinearSpace.dist`.

        The dist satisfies properties

        positivity
        ``d(x, y) >= 0``

        coincidence
        ``d(x, y) = 0`` iff ``x = y``

        symmetry
        ``d(x, y) = d(y, x)``

        triangle inequality
        ``d(x, z) = d(x, y) + d(y, z)``

        References
        ----------
        Wikipedia article on metric_

        .. _metric: https://en.wikipedia.org/wiki/Metric_(mathematics)
        """
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
        """Check vector multiplication with zero is zero."""
        print('\nMultiplication by zero, x * 0 = 0')

        zero = self.space.zero()

        with FailCounter('error = ||x*0||') as counter:
            for [n_x, x] in samples(self.space):
                error = (zero * x).norm()

                if error > self.eps:
                    counter.fail('x={:25s},: error={}'
                                 ''.format(n_x, error))

    def _multiply_commutative(self):
        """Check commutativity of vector multiplication."""
        print('\nMultiplication commutative, x * y = y * x')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], _ in samples(self.space,
                                                 self.space,
                                                 self.space):
                correct = _approx_equal(x * y, y * x, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s}'
                                 ''.format(n_x, n_y))

    def _multiply_associative(self):
        """Check associativity of vector multiplication."""
        print('\nMultiplication associative, x * (y * z) = (x * y) * z')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                correct = _approx_equal(x * (y * z), (x * y) * z, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def _multiply_distributive_scalar(self):
        """Check distributivity of scalar multiplication."""
        print('\nMultiplication distributive wrt scal, a * (x + y) = '
              'a * x + a * y')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], a in samples(self.space,
                                                 self.space,
                                                 self.space.field):
                correct = _approx_equal(a * (x + y), a * x + a * y, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s} a={}'
                                 ''.format(n_x, n_y, a))

    def _multiply_distributive_vec(self):
        """Check distributivity of vector multiplication."""
        print('\nMultiplication distributive wrt vec, x * (y + z) = '
              'x * y + x * z')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                correct = _approx_equal(x * (y + z), x * y + x * z, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def multiply(self):
        """Verify `LinearSpace.multiply`.

        Multiplication satisfies

        Zero element
        ``0 * x = 0``

        Commutativity
        ``x * y = y * x``

        Associativity
        ``x * (y * z) = (x * y) * z``

        Distributivity
        ``a * (x + y) = a * x + a * y``
        ``x * (y + z) = x * y + x * z``

        """

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
        """Verify `LinearSpace.__eq__`."""

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
        """Verify `LinearSpace.__contains__`."""

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

    def vector_assign(self):
        """Verify `LinearSpaceVector.assign`."""

        print('\nVector.assign()')

        with FailCounter() as counter:
            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                x.assign(y)
                correct = _approx_equal(x, y, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s}'
                                 ''.format(n_x, n_y))

    def vector_copy(self):
        """Verify `LinearSpaceVector.copy`."""

        print('\nVector.copy()')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                # equal after copy
                y = x.copy()
                correct = _approx_equal(x, y, self.eps)
                if not correct:
                    counter.fail('failed with x={:s5s}'
                                 ''.format(n_x))

                # modify y, x stays the same
                y *= 2.0
                correct = n_x == 'Zero' or not _approx_equal(x, y, self.eps)
                if not correct:
                    counter.fail('modified y, x changed with x={:25s}'
                                 ''.format(n_x))

    def vector_set_zero(self):
        """Verify `LinearSpaceVector.set_zero`."""

        print('\nVector.set_zero()')

        zero = self.space.zero()
        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                x.set_zero()
                correct = _approx_equal(x, zero, self.eps)
                if not correct:
                    counter.fail('failed with x={:25s}'
                                 ''.format(n_x))

    def vector_equals(self):
        """Verify `LinearSpaceVector.__eq__`."""

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

    def vector_space(self):
        """Verify `LinearSpaceVector.space`."""

        print('\nVector.space')

        with FailCounter() as counter:
            for [n_x, x] in samples(self.space):
                if x.space != self.space:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def vector(self):
        """Verify `LinearSpaceVector`."""

        print('\n== Verifying Vector ==\n')

        self.vector_assign()
        self.vector_copy()
        self.vector_set_zero()
        self.vector_equals()
        self.vector_space()

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
        """Return ``str(self)``."""
        return 'SpaceTest({})'.format(self.space)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'SpaceTest({!r})'.format(self.space)


if __name__ == '__main__':
    from odl.space.ntuples import Rn
    SpaceTest(Rn(10)).run_tests()
