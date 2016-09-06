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

from odl.set import Field
from odl.diagnostics.examples import samples
from odl.util.testutils import FailCounter


__all__ = ('SpaceTest',)


def _approx_equal(x, y, eps):
    """Test if elements ``x`` and ``y`` are approximately equal.

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

    This class allows users to automatically test various features of a
    `LinearSpace` such as linearity and vector space operations.
    """

    def __init__(self, space, verbose=True, tol=1e-5):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space that should be tested.
        verbose : bool, optional
            If ``True``, print additional info text.
        tol : float, optional
            Tolerance parameter used as a base for the actual tolerance
            in the tests. Depending on the expected accuracy, the actual
            tolerance used in a test can be a factor times this number.
        """
        self.space = space
        self.verbose = bool(verbose)
        self.tol = float(tol)

    def log(self, message):
        """Print message if ``self.verbose == True``."""
        if self.verbose:
            print(message)

    def element_method(self):
        """Verify `LinearSpace.element`."""
        with FailCounter(test_name='Verifying element method',
                         logger=self.log) as counter:
            try:
                elem = self.space.element()
            except NotImplementedError:
                counter.fail('*** element failed ***')
                return

            if elem not in self.space:
                counter.fail('*** space.element() not in space ***')

    def field(self):
        """Verify `LinearSpace.field`."""
        with FailCounter(test_name='Verifying field property',
                         logger=self.log) as counter:
            try:
                field = self.space.field
            except NotImplementedError:
                counter.fail('*** field failed ***')
                return

            if not isinstance(field, Field):
                counter.fail('*** space.field not a `Field` ***')
                return

            try:
                zero = field.element(0)
            except NotImplementedError:
                counter.fail('*** field.element(0) failed ***')
                zero = None

            if zero is not None and zero != 0:
                counter.fail('*** field.element(0) != 0 ***')

            if zero is not None and zero != 0.0:
                counter.fail('*** field.element(0) != 0.0 ***')

            try:
                one = field.element(1)
            except NotImplementedError:
                counter.fail('*** field.element(1) failed ***')
                one = None

            if one is not None and one != 1:
                counter.fail('*** field.element(1) != 1 ***')

            if one is not None and one != 1.0:
                counter.fail('*** field.element(1) != 1.0 ***')

            try:
                minus_one = field.element(-1)
            except NotImplementedError:
                counter.fail('field.element(-1) failed')
                minus_one = None

            if minus_one is not None and minus_one != -1:
                counter.fail('field.element(-1) != -1')

            if minus_one is not None and minus_one != -1.0:
                counter.fail('field.element(-1) != -1.0')

    def _associativity_of_addition(self):
        """Verify addition associativity."""
        with FailCounter(
                test_name='Verifying associativity of addition',
                err_msg='error = dist(x + (y + z), (x + y) + z)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                correct = _approx_equal(x + (y + z), (x + y) + z, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def _commutativity_of_addition(self):
        """Verify addition commutativity."""
        with FailCounter(
                test_name='Verifying commutativity of addition',
                err_msg='error = dist(x + y, y + x)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y] in samples(self.space, self.space):
                correct = _approx_equal(x + y, y + x, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s}'
                                 ''.format(n_x, n_y))

    def _identity_of_addition(self):
        """Verify additive neutral element ('zero')."""
        try:
            zero = self.space.zero()
        except (AttributeError, NotImplementedError):
            print('*** SPACE HAS NO ZERO VECTOR ***')
            return

        with FailCounter(
                test_name='Verifying identity element of addition',
                err_msg='error = dist(x + 0, x)',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                correct = _approx_equal(x + zero, x, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _inverse_element_of_addition(self):
        """Verify additive inverse."""
        try:
            zero = self.space.zero()
        except (AttributeError, NotImplementedError):
            print('*** SPACE HAS NO ZERO VECTOR ***')
            return

        with FailCounter(
                test_name='Verifying inverse element of addition',
                err_msg='error = dist(x + (-x), 0)',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                correct = _approx_equal(x + (-x), zero, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _commutativity_of_scalar_mult(self):
        """Verify scalar multiplication commutativity."""
        with FailCounter(
                test_name='Verifying commutativity of scalar multiplication',
                err_msg='error = dist(a * (b * x), (a * b) * x)',
                logger=self.log) as counter:

            for [n_x, x], [_, a], [_, b] in samples(self.space,
                                                    self.space.field,
                                                    self.space.field):
                correct = _approx_equal(a * (b * x), (a * b) * x, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s}, a={}, b={}'
                                 ''.format(n_x, a, b))

    def _identity_of_mult(self):
        """Verify multiplicative neutral element ('one')."""
        with FailCounter(
                test_name='Verifying identity element of multiplication',
                err_msg='error = dist(1 * x, x)',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                correct = _approx_equal(1 * x, x, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def _distributivity_of_mult_vector(self):
        """Verify scalar multiplication distributivity wrt vector addition."""
        with FailCounter(
                test_name='Verifying distributivity of scalar multiplication '
                          'under vector addition',
                err_msg='error = dist(a * (x + y), a * x + a * y)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], [_, a] in samples(self.space,
                                                      self.space,
                                                      self.space.field):
                correct = _approx_equal(a * (x + y), a * x + a * y, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s}, y={:25s}, a={}'
                                 ''.format(n_x, n_y, a))

    def _distributivity_of_mult_scalar(self):
        """Verify scalar multiplication distributivity wrt scalar addition."""
        with FailCounter(
                test_name='Verifying distributivity of scalar multiplication '
                          'under scalar addition',
                err_msg='error = dist((a + b) * x, a * x + b * x)',
                logger=self.log) as counter:

            for [n_x, x], [_, a], [_, b] in samples(self.space,
                                                    self.space.field,
                                                    self.space.field):
                correct = _approx_equal((a + b) * x, a * x + b * x, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s}, a={}, b={}'
                                 ''.format(n_x, a, b))

    def _subtraction(self):
        """Verify element subtraction as addition of additive inverse."""
        with FailCounter(
                test_name='Verifying element subtraction',
                err_msg='error = dist(x - y, x + (-1 * y))',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y] in samples(self.space, self.space):
                correct = (_approx_equal(x - y, x + (-1 * y), self.tol) and
                           _approx_equal(x - y, x + (-y), self.tol))
                if not correct:
                    counter.fail('failed with x={:25s}, y={:25s}'
                                 ''.format(n_x, n_y))

    def _division(self):
        """Verify scalar division as multiplication with mult. inverse."""
        with FailCounter(
                test_name='Verifying scalar division',
                err_msg='error = dist(x / a, x * (1/a))',
                logger=self.log) as counter:

            for [n_x, x], [_, a] in samples(self.space, self.space.field):
                if a != 0:
                    correct = _approx_equal(x / a, x * (1.0 / a), self.tol)
                    if not correct:
                        counter.fail('failed with x={:25s}, a={}'
                                     ''.format(n_x, a))

    def _lincomb_aliased(self):
        """Verify several scenarios of aliased linear combination."""
        with FailCounter(
                test_name='Verifying linear combination with aliased input',
                err_msg='error = dist(aliased, non-aliased)',
                logger=self.log) as counter:

            for [n_x, x_in], [n_y, y] in samples(self.space, self.space):
                x = x_in.copy()
                x.lincomb(1, x, 1, y)
                correct = _approx_equal(x, x_in + y, self.tol)
                if not correct:
                    counter.fail('failed with x.lincomb(1, x, 1, y),'
                                 'x={:25s} y={:25s} '
                                 ''.format(n_x, n_y))

                x = x_in.copy()
                x.lincomb(1, x, 1, x)
                correct = _approx_equal(x, x_in + x_in, self.tol)
                if not correct:
                    counter.fail('failed with x.lincomb(1, x, 1, x),'
                                 'x={:25s} '
                                 ''.format(n_x))

    def _lincomb(self):
        """Verify linear combination."""
        self.log('\nTesting lincomb')
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
        self.log('\n== Verifying linear space properties ==\n')
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
        """Verify homogeneity of the inner product in the first argument."""
        with FailCounter(
                test_name='Verifying homogeneity of the inner product in the '
                          'first argument',
                err_msg='error = |<a*x, y> - a*<x, y>|',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], [_, a] in samples(self.space,
                                                      self.space,
                                                      self.space.field):
                error = abs((a * x).inner(y) - a * x.inner(y))
                if error > self.tol:
                    counter.fail('x={:25s}, y={:25s}, a={}: error={}'
                                 ''.format(n_x, n_y, a, error))

    def _inner_conjugate_symmetry(self):
        """Verify conjugate symmetry of the inner product."""
        with FailCounter(
                test_name='Verifying conjugate symmetry of the inner product',
                err_msg='error = |<x, y> - <y, x>.conj()|',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y] in samples(self.space, self.space):
                error = abs((x).inner(y) - y.inner(x).conjugate())
                if error > self.tol:
                    counter.fail('x={:25s}, y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def _inner_linear_sum(self):
        """Verify distributivity of the inner product in the first argument."""
        with FailCounter(
                test_name='Verifying distributivity of the inner product'
                          'in the first argument',
                err_msg='error = |<x+y, z> - (<x, z> + <y, z>)|',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                error = abs((x + y).inner(z) - (x.inner(z) + y.inner(z)))
                if error > self.tol:
                    counter.fail('x={:25s}, y={:25s}, z={:25s}: error={}'
                                 ''.format(n_x, n_y, n_z, error))

    def _inner_positive(self):
        """Verify positive definiteness of the inner product."""
        with FailCounter(
                test_name='Verifying positive definiteness of the inner '
                          'product',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                inner = x.inner(x)

                if abs(inner.imag) > self.tol:
                    counter.fail('<x, x>.imag != 0, x={:25s}, <x, x>.imag = {}'
                                 ''.format(n_x, inner.imag))

                if n_x == 'Zero' and inner.real != 0:
                    counter.fail('<0, 0> != 0.0, x={:25s}: <0, 0>={}'
                                 ''.format(n_x, inner))

                elif n_x != 'Zero' and inner.real <= 0:
                    counter.fail('<x, x> <= 0,   x={:25s}: <x, x>={}'
                                 ''.format(n_x, inner))

    def inner(self):
        """Verify `LinearSpace.inner`.

        The inner product is checked for the following properties:

        - conjugate symmetry:

            ``<x, y> = <y, x>^*`` (^* complex conjugate)

        - linearity:

            ``<a * x, y> = a * <x, y>``

            ``<x + y, z> = <x, z> + <y, z>``

        - positive definiteness:

            ``<x, x> >= 0``

        References
        ----------
        Wikipedia article on `inner product`_.

        .. _inner product: https://en.wikipedia.org/wiki/Inner_product_space
        """
        self.log('\n== Verifying inner product ==\n')

        try:
            zero = self.space.zero()
            zero.inner(zero)
        except NotImplementedError:
            self.log('Space has no inner product')
            return

        self._inner_conjugate_symmetry()
        self._inner_linear_scalar()
        self._inner_linear_sum()
        self._inner_positive()

    def _norm_positive(self):
        """Verify positive definiteness of the norm."""
        with FailCounter(
                test_name='Verifying positive definiteness of the norm',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                norm = x.norm()

                if n_x == 'Zero' and norm != 0:
                    counter.fail('||0|| != 0.0, x={:25s}: ||x||={}'
                                 ''.format(n_x, norm))

                elif n_x != 'Zero' and norm <= 0:
                    counter.fail('||x|| <= 0,   x={:25s}: ||x||={}'
                                 ''.format(n_x, norm))

    def _norm_subadditive(self):
        """Verify subadditivity of the norm."""
        with FailCounter(
                test_name='Verifying sub-additivity of the norm',
                err_msg='error = max(||x+y|| - (||x|| + ||y||), 0)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y] in samples(self.space, self.space):
                norm_x = x.norm()
                norm_y = y.norm()
                norm_xy = (x + y).norm()

                error = norm_xy - norm_x - norm_y

                if error > 0:
                    counter.fail('x={:25s} y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def _norm_homogeneity(self):
        """Verify positive homogeneity of the norm."""
        with FailCounter(
                test_name='Verifying positive homogeneity of the norm',
                err_msg='error = | ||a*x|| - |a|*||x|| |',
                logger=self.log) as counter:

            for [n_x, x], [_, a] in samples(self.space, self.space.field):
                error = abs((a * x).norm() - abs(a) * x.norm())
                if error > self.tol:
                    counter.fail('x={:25s} a={}: error={}'
                                 ''.format(n_x, a, error))

    def _norm_inner_compatible(self):
        """Verify compatibility of norm and inner product."""
        try:
            zero = self.space.zero()
            zero.inner(zero)
        except NotImplementedError:
            self.log('Space has no inner product')
            return

        with FailCounter(
                test_name='Verifying compatibility of norm and inner product',
                err_msg='error = | ||x||^2 - <x, x> |',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                error = abs(x.norm() ** 2 - x.inner(x))

                if error > self.tol:
                    counter.fail('x={:25s}: error={}'
                                 ''.format(n_x, error))

    def norm(self):
        """Verify `LinearSpace.norm`.

        The norm is checked for the following properties:

        - linearity:

            ``||a * x|| = |a| * ||x||``

        - subadditivity:

            ``||x + y|| <= ||x|| + ||y||``

        - positive homogeneity:

            ``||a * x|| = |a| * ||x||``

        - positive definiteness:

            ``||x|| >= 0``

            ``||x|| = 0`` iff ``x = 0``

        - compatibility with the inner product (if available):

            ``||x||^2 = <x, x>``

        References
        ----------
        Wikipedia article on norm_.

        .. _norm: https://en.wikipedia.org/wiki/Norm_(mathematics)
        """
        self.log('\n== Verifying norm ==\n')

        try:
            self.space.zero().norm()
        except NotImplementedError:
            self.log('Space has no norm')
            return

        self._norm_positive()
        self._norm_subadditive()
        self._norm_homogeneity()
        self._norm_inner_compatible()

    def _dist_positivity(self):
        """Verify nonnegativity of the distance."""

        with FailCounter(
                test_name='Verifying nonnegativity of the distance',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y] in samples(self.space, self.space):
                dist = x.dist(y)

                if n_x == n_y and dist != 0:
                    counter.fail('d(x, x) != 0.0, x={:25s}: dist={}'
                                 ''.format(n_x, dist))
                elif n_x != n_y and dist <= 0:
                    counter.fail('d(x, y) <= 0,   x={:25s} y={:25s}: dist={}'
                                 ''.format(n_x, n_y, dist))

    def _dist_symmetric(self):
        """Verify symmetry of the distance."""
        with FailCounter(
                test_name='Verifying symmetry of the distance',
                err_msg='error = |d(x, y) - d(y, x)|',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y] in samples(self.space, self.space):
                dist_1 = x.dist(y)
                dist_2 = y.dist(x)
                error = abs(dist_1 - dist_2)

                if error > self.tol:
                    counter.fail('x={:25s}, y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def _dist_subtransitive(self):
        """Verify sub-transitivity of the distance."""
        with FailCounter(
                test_name='Verifying sub-additivity of the distance',
                err_msg='error = max(d(x,z) - (d(x, y) + d(y, z)), 0)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                dxz = x.dist(z)
                dxy = x.dist(y)
                dyz = y.dist(z)
                error = dxz - (dxy + dyz)

                if error > self.tol:
                    counter.fail('x={:25s}, y={:25s}, z={:25s}: error={}'
                                 ''.format(n_x, n_y, n_z, error))

    def _dist_norm_compatible(self):
        """Verify compatibility of distance and norm."""
        try:
            self.space.zero().norm()
        except NotImplementedError:
            self.log('Space has no norm')
            return

        with FailCounter(
                test_name='Verifying compatibility of distance and norm',
                err_msg='error = |d(x, y) - ||x-y|| |',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                error = abs(x.dist(y) - (x - y).norm())

                if error > self.tol:
                    counter.fail('x={:25s}, y={:25s}: error={}'
                                 ''.format(n_x, n_y, error))

    def dist(self):
        """Verify `LinearSpace.dist`.

        The distance metric is checked for the following properties:

        - positive definiteness:

            ``d(x, y) >= 0``

            ``d(x, y) = 0`` iff ``x = y``

        - symmetry:

            ``d(x, y) = d(y, x)``

        - sub-transitivity:

            ``d(x, z) <= d(x, y) + d(y, z)``

        - compatibility with the norm (if available):

            ``d(x, y) = ||x - y||``

        References
        ----------
        Wikipedia article on metric_

        .. _metric: https://en.wikipedia.org/wiki/Metric_(mathematics)
        """
        self.log('\n== Verifying dist ==\n')

        try:
            zero = self.space.zero()
            self.space.dist(zero, zero)
        except NotImplementedError:
            self.log('Space has no distance metric')
            return

        self._dist_positivity()
        self._dist_symmetric()
        self._dist_subtransitive()
        self._dist_norm_compatible()

    def _multiply_zero(self):
        """Verify that vector multiplication with zero is zero."""
        try:
            zero = self.space.zero()
        except NotImplementedError:
            print('*** SPACE HAS NO ZERO VECTOR ***')
            return

        with FailCounter(
                test_name='Verifying vector multiplication with zero',
                err_msg='error = ||x * 0||',
                logger=self.log) as counter:
            for [n_x, x] in samples(self.space):
                error = (zero * x).norm()

                if error > self.tol:
                    counter.fail('x={:25s},: error={}'
                                 ''.format(n_x, error))

    def _multiply_commutative(self):
        """Verify commutativity of vector multiplication."""
        with FailCounter(
                test_name='Verifying commutativity of vector multiplication',
                err_msg='error = dist(x * y, y * x)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], _ in samples(self.space,
                                                 self.space,
                                                 self.space):
                correct = _approx_equal(x * y, y * x, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s}'
                                 ''.format(n_x, n_y))

    def _multiply_associative(self):
        """Verify associativity of vector multiplication."""
        with FailCounter(
                test_name='Verifying associativity of vector multiplication',
                err_msg='error = dist(x * (y * z), (x * y) * z)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                correct = _approx_equal(x * (y * z), (x * y) * z, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def _multiply_distributive_scalar(self):
        """Verify distributivity of scalar multiplication."""
        with FailCounter(
                test_name='Verifying distributivity of vector multiplication '
                          'under scalar multiplication',
                err_msg='error = dist(a * (x + y), a * x + a * y)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], [_, a] in samples(self.space,
                                                      self.space,
                                                      self.space.field):
                correct = _approx_equal(a * (x + y), a * x + a * y, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s} a={}'
                                 ''.format(n_x, n_y, a))

    def _multiply_distributive_vector(self):
        """Verify distributivity of vector multiplication."""
        with FailCounter(
                test_name='Verifying distributivity of vector multiplication '
                          'under vector multiplication',
                err_msg='error = dist(x * (y + z), x * y + x * z)',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y], [n_z, z] in samples(self.space,
                                                        self.space,
                                                        self.space):
                correct = _approx_equal(x * (y + z), x * y + x * z, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s} z={:25s}'
                                 ''.format(n_x, n_y, n_z))

    def multiply(self):
        """Verify `LinearSpace.multiply`.

        The vector multiplication is checked for the following properties:

        - Zero element:

            ``0 * x = 0``

        - Commutativity:

            ``x * y = y * x``

        - Associativity:

            ``x * (y * z) = (x * y) * z``

        - Distributivity:

            ``a * (x + y) = a * x + a * y``

            ``x * (y + z) = x * y + x * z``
        """
        self.log('\n== Verifying multiplication ==\n')

        try:
            zero = self.space.zero()
        except NotImplementedError:
            print('*** SPACE HAS NO ZERO VECTOR ***')
            return

        try:
            self.space.multiply(zero, zero)
        except NotImplementedError:
            self.log('Space has no vector multiplication.')
            return

        self._multiply_zero()
        self._multiply_commutative()
        self._multiply_associative()
        self._multiply_distributive_scalar()
        self._multiply_distributive_vector()

    def equals(self):
        """Verify `LinearSpace.__eq__`."""

        self.log('\n== Verifying __eq__ ==\n')

        if not self.space == self.space:
            print('** space == space failed ***')

        if self.space != self.space:
            print('** not space != space failed***')

        if self.space != copy(self.space):
            print('** space == copy(space) failed***')

        if self.space != deepcopy(self.space):
            print('** space == deepcopy(space) failed***')

        with FailCounter(
                test_name='Verify behavior of `space == obj` when `obj` '
                          'is not a space',
                logger=self.log) as counter:

            for obj in [[1, 2], list(), tuple(), dict(), 5.0]:
                if self.space == obj:
                    counter.fail('space == obj,  with obj={}'
                                 ''.format(obj))

                if not self.space != obj:
                    counter.fail('not space != obj,  with obj={}'
                                 ''.format(obj))

    def contains(self):
        """Verify `LinearSpace.__contains__`."""
        with FailCounter(
                test_name='Verify behavior of `obj in space`',
                logger=self.log) as counter:

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

    def element_assign(self):
        """Verify `LinearSpaceElement.assign`."""
        with FailCounter(
                test_name='Verify behavior of `LinearSpaceElement.assign`',
                logger=self.log) as counter:

            for [n_x, x], [n_y, y] in samples(self.space,
                                              self.space):
                x.assign(y)
                correct = _approx_equal(x, y, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s} y={:25s}'
                                 ''.format(n_x, n_y))

    def element_copy(self):
        """Verify `LinearSpaceElement.copy`."""
        with FailCounter(
                test_name='Verify behavior of `LinearSpaceElement.copy`',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                # equal after copy
                y = x.copy()
                correct = _approx_equal(x, y, self.tol)
                if not correct:
                    counter.fail('failed with x={:s5s}'
                                 ''.format(n_x))

                # modify y, x stays the same
                y *= 2.0
                correct = n_x == 'Zero' or not _approx_equal(x, y, self.tol)
                if not correct:
                    counter.fail('modified y, x changed with x={:25s}'
                                 ''.format(n_x))

    def element_set_zero(self):
        """Verify `LinearSpaceElement.set_zero`."""
        try:
            zero = self.space.zero()
        except NotImplementedError:
            print('*** SPACE HAS NO ZERO VECTOR ***')
            return

        with FailCounter(
                test_name='Verify behavior of `LinearSpaceElement.set_zero`',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                x.set_zero()
                correct = _approx_equal(x, zero, self.tol)
                if not correct:
                    counter.fail('failed with x={:25s}'
                                 ''.format(n_x))

    def element_equals(self):
        """Verify `LinearSpaceElement.__eq__`."""
        try:
            zero = self.space.zero()
        except NotImplementedError:
            print('*** SPACE HAS NO ZERO VECTOR ***')
            return

        try:
            zero == zero
        except NotImplementedError:
            self.log('Vector has no __eq__')
            return

        with FailCounter(
                test_name='Verify behavior of `element1 == element2`',
                logger=self.log) as counter:

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

    def element_space(self):
        """Verify `LinearSpaceElement.space`."""
        with FailCounter(
                test_name='Verify `LinearSpaceElement.space`',
                logger=self.log) as counter:

            for [n_x, x] in samples(self.space):
                if x.space != self.space:
                    counter.fail('failed with x={:25s}'.format(n_x))

    def element(self):
        """Verify `LinearSpaceElement`."""

        self.log('\n== Verifying element attributes ==\n')
        self.element_assign()
        self.element_copy()
        self.element_set_zero()
        self.element_equals()
        self.element_space()

    def run_tests(self):
        """Run all tests on this space."""
        self.log('\n== RUNNING ALL TESTS ==\n')
        self.log('Space = {}'.format(self.space))
        self.field()
        self.element_method()
        self.linearity()
        self.inner()
        self.norm()
        self.dist()
        self.multiply()
        self.equals()
        self.contains()
        self.element()

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({})'.format(self.__class__.__name__, self.space)

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '{!r}'.format(self.space)
        if not self.verbose:
            inner_str += ', verbose=False'
        if self.tol != 1e-5:
            inner_str += ', tol={}'.format(self.tol)
        return '{}({})'.format(self.__class__.__name__, inner_str)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl import rn, uniform_discr
    SpaceTest(rn(10), verbose=False).run_tests()
    SpaceTest(uniform_discr([0, 0], [1, 1], [5, 5])).run_tests()
