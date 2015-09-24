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

"""Cartesian products of `LinearSpace`'s.

TODO: document public interface
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from builtins import str, zip, super
from future import standard_library
standard_library.install_aliases()

from numbers import Integral

# External
import numpy as np

# ODL imports
from odl.sets.space import LinearSpace


__all__ = ('ProductSpace',)


def _strip_space(x):
    """Strip the SPACE.element( ... ) part from a repr."""
    r = repr(x)
    space_repr = '{!r}.element('.format(x.space)
    if r.startswith(space_repr) and r.endswith(')'):
        r = r[len(space_repr):-1]
    return r


def _indent(x):
    """Indent a string by 4 characters."""
    lines = x.split('\n')
    for i in range(len(lines)):
        lines[i] = '    ' + lines[i]
    return '\n'.join(lines)


def _prod_inner_sum_not_defined(x):
    raise NotImplementedError('inner product not defined with custom product '
                              'norm.')


class ProductSpace(LinearSpace):

    """The Cartesian product of N linear spaces.

    The product X1 x ... x XN is itself a linear space, where the
    linear combination is defined component-wise.

    TODO: document public interface
    """

    def __init__(self, *spaces, **kwargs):
        """Initialize a new ProductSpace.

        The product X1 x ... x XN is itself a linear space, where the
        linear combination is defined component-wise.

        Parameters
        ----------
        args : {'LinearSpace' and 'int' OR 'LinearSpace' instances
            Either a space and an integer,
            in this case the power of the space is taken (R^n)
            Otherwise, a set of spaces,
            in this case the product is taken (RxRxRxC)
        kwargs : {'ord', 'weights', 'prod_norm'}
            'ord' : float, optional
                Order of the product distance/norm, i.e.
                dist(x, y) = np.linalg.norm(x-y, ord=ord)
                norm(x) = np.linalg.norm(x, ord=ord)
                Default: 2.0
            'weights' : array-like, optional, only usable with 'ord'
                Array of weights, same size as number of space
                components. All weights must be positive. It is
                multiplied with the tuple of distances before
                applying the Rn norm or 'prod_norm'.
                Default: (1.0,...,1.0)
            'prod_norm' : callable, optional
                Function that should be applied to the array of
                distances/norms. Specifying a product norm causes
                the space to NOT be a Hilbert space.
                Default: np.linalg.norm(x, ord=ord)

        The following float values for `prod_norm` can be specified.
        Note that any value of ord < 1 only gives a pseudonorm.

        =========  ==========================
        prod_norm    Distance Definition
        =========  ==========================
        'inf'       max(w * z)
        '-inf'      min(w * z)
        0           sum(w * z != 0)
        other       sum(w * z**ord)**(1/ord)
        =========  ==========================

        Here, z = (x[0].dist(y[0]),..., x[n-1].dist(y[n-1])) and
        w = weights.

        Returns
        -------
        prodspace : ProductSpace instance

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> r2x3 = ProductSpace(Rn(2), Rn(3))
        """
        if (len(spaces) == 2 and
                isinstance(spaces[0], LinearSpace) and
                isinstance(spaces[1], Integral)):
            # Powerspace initialization
            spaces = [spaces[0]] * spaces[1]

        wrong_spaces = [spc for spc in spaces
                        if not isinstance(spc, LinearSpace)]
        if wrong_spaces:
            raise TypeError('{!r} not LinearSpace instance(s).'
                            ''.format(wrong_spaces))

        if not all(spc.field == spaces[0].field for spc in spaces):
            raise TypeError('All spaces must have the same field')

        prod_norm = kwargs.get('prod_norm', None)

        if prod_norm is not None:
            if not callable(prod_norm):
                raise TypeError('product norm is not callable.')

            self._prod_norm = prod_norm
            self._prod_inner_sum = _prod_inner_sum_not_defined
        else:
            order = float(kwargs.get('ord', 2.0))

            weights = kwargs.get('weights', None)
            if weights is not None:
                weights = np.atleast_1d(weights)
                if not np.all(weights > 0):
                    raise ValueError('weights must all be positive')
                if not len(weights) == len(spaces):
                    raise ValueError('spaces and weights have different '
                                     'lengths ({} != {}).'
                                     ''.format(len(spaces), len(weights)))

                def w_norm(x):
                    return np.linalg.norm(x*weights, ord=order)

                self._prod_norm = w_norm

                if order == 2.0:
                    def w_inner_sum(x):
                        return np.linalg.dot(x, weights)

                    self._prod_inner_sum = w_inner_sum
                else:
                    self._prod_inner_sum = _prod_inner_sum_not_defined
            else:
                def norm(x):
                    return np.linalg.norm(x, ord=order)

                self._prod_norm = norm

                if order == 2.0:
                    self._prod_inner_sum = np.sum
                else:
                    self._prod_inner_sum = _prod_inner_sum_not_defined

        self._spaces = tuple(spaces)
        self._size = len(spaces)
        self._field = spaces[0].field
        super().__init__()

    @property
    def size(self):
        """The number of factors."""
        return self._size

    @property
    def field(self):
        """The common underlying field of all factors."""
        return self._field

    @property
    def spaces(self):
        """A tuple containing all spaces."""
        return self._spaces

    def element(self, inp=None):
        """Create an element in the product space.

        Parameters
        ----------
        The method has three call patterns, the first is:

        args : None
            Create a new vector from scratch.

        The second is to wrap existing vectors:

        args : tuple of `LinearSpace.Vector`s
            A tuple of vectors in the underlying spaces.
            This will simply wrap the Vectors (not copy).

        The third pattern is to create a new Vector from scratch, in
        this case

        args : tuple of array-like objects

        Returns
        -------
        ProductSpace.Vector instance

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> vec_2, vec_3 = r2.element(), r3.element()
        >>> r2x3 = ProductSpace(r2, r3)
        >>> vec_2x3 = r2x3.element()
        >>> vec_2.space == vec_2x3[0].space
        True
        >>> vec_3.space == vec_2x3[1].space
        True

        Creates an element in the product space

        >>> from odl.space.cartesian import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> prod = ProductSpace(r2, r3)
        >>> x2 = r2.element([1, 2])
        >>> x3 = r3.element([1, 2, 3])
        >>> x = prod.element([x2, x3])
        >>> print(x)
        {[1.0, 2.0], [1.0, 2.0, 3.0]}
        """

        # If data is given as keyword arg, prefer it over arg list
        if inp is None:
            inp = [space.element() for space in self.spaces]

        if (all(isinstance(v, LinearSpace.Vector) for v in inp) and
                all(part.space == space
                    for part, space in zip(inp, self.spaces))):
            parts = list(inp)
        else:
            # Delegate constructors
            parts = [space.element(arg)
                     for arg, space in zip(inp, self.spaces)]

        return self.Vector(self, parts)

    def zero(self):
        """Create the zero vector of the product space.

        The i:th component of the product space zero vector is the
        zero vector of the i:th space in the product.

        Parameters
        ----------
        None

        Returns
        -------
        zero : ProductSpace.Vector
            The zero vector in the product space

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> zero_2, zero_3 = r2.zero(), r3.zero()
        >>> r2x3 = ProductSpace(r2, r3)
        >>> zero_2x3 = r2x3.zero()
        >>> zero_2 == zero_2x3[0]
        True
        >>> zero_3 == zero_2x3[1]
        True
        """
        return self.element([space.zero() for space in self.spaces])

    def _lincomb(self, z, a, x, b, y):
        # pylint: disable=protected-access
        for space, zp, xp, yp in zip(self.spaces, z.parts, x.parts, y.parts):
            space._lincomb(zp, a, xp, b, yp)

    def _dist(self, x, y):
        dists = np.fromiter(
            (spc._dist(xp, yp)
             for spc, xp, yp in zip(self.spaces, x.parts, y.parts)),
            dtype=np.float64, count=self.size)
        return self._prod_norm(dists)

    def _norm(self, x):
        norms = np.fromiter(
            (spc._norm(xp)
             for spc, xp in zip(self.spaces, x.parts)),
            dtype=np.float64, count=self.size)
        return self._prod_norm(norms)

    def _inner(self, x, y):
        inners = np.fromiter(
            (spc._inner(xp, yp)
             for spc, xp, yp in zip(self.spaces, x.parts, y.parts)),
            dtype=np.float64, count=self.size)
        return self._prod_norm(inners)

    def _multiply(self, z, x, y):
        for spc, zp, xp, yp in zip(self.spaces, z.parts, x.parts, y.parts):
            spc._multiply(zp, xp, yp)

    def __eq__(self, other):
        """`s.__eq__(other) <==> s == other`.

        Returns
        -------
        equals : bool
            `True` if `other` is a `ProductSpace` instance, has
            the same length and the same factors. `False` otherwise.

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> rn, rm = Rn(2), Rn(3)
        >>> r2x3, rnxm = ProductSpace(r2, r3), ProductSpace(rn, rm)
        >>> r2x3 == rnxm
        True
        >>> r3x2 = ProductSpace(r3, r2)
        >>> r2x3 == r3x2
        False
        >>> r5 = ProductSpace(*[Rn(1)]*5)
        >>> r2x3 == r5
        False
        >>> r5 = Rn(5)
        >>> r2x3 == r5
        False
        """
        if other is self:
            return True
        else:
            return (type(self) == type(other) and
                    len(self) == len(other) and
                    all(x == y for x, y in zip(self.spaces,
                                               other.spaces)))

    def __len__(self):
        """The number of factors."""
        return self._size

    def __getitem__(self, index_or_slice):
        """Implementation of spc[index] and spc[slice]."""
        return self.spaces[index_or_slice]

    def __str__(self):
        if all(self.spaces[0] == space for space in self.spaces):
            return '{' + str(self.spaces[0]) + '}^' + str(self.size)
        else:
            return ' x '.join(str(space) for space in self.spaces)

    def __repr__(self):
        if all(self.spaces[0] == space for space in self.spaces):
            return 'ProductSpace({!r}, {})'.format(self.spaces[0],
                                                   self.size)
        else:
            inner_str = ', '.join(repr(space) for space in self.spaces)
            return 'ProductSpace({})'.format(inner_str)

    class Vector(LinearSpace.Vector):
        def __init__(self, space, parts):
            super().__init__(space)
            self.parts = parts

        @property
        def size(self):
            return self.space.size

        def __len__(self):
            return len(self.space)

        def __getitem__(self, index):
            return self.parts[index]

        def __setitem__(self, index, value):
            self.parts[index] = value

        def __str__(self):
            inner_str = ', '.join(str(part) for part in self.parts)
            return '{{{}}}'.format(inner_str)

        def __repr__(self):
            """ Get a representation of this vector

            Returns
            -------
            repr : string


            Examples
            --------
            >>> from odl.space.cartesian import Rn
            >>> r2, r3 = Rn(2), Rn(3)
            >>> r2x3 = ProductSpace(r2, r3)
            >>> x = r2x3.element([[1, 2], [3, 4, 5]])
            >>> eval(repr(x)) == x
            True

            The result is readable

            >>> x
            ProductSpace(Rn(2), Rn(3)).element([
                [1.0, 2.0],
                [3.0, 4.0, 5.0]
            ])

            Nestled spaces work as well

            >>> X = ProductSpace(r2x3, r2x3)
            >>> x = X.element([[[1, 2], [3, 4, 5]],[[1, 2], [3, 4, 5]]])
            >>> eval(repr(x)) == x
            True
            >>> x
            ProductSpace(ProductSpace(Rn(2), Rn(3)), 2).element([
                [
                    [1.0, 2.0],
                    [3.0, 4.0, 5.0]
                ],
                [
                    [1.0, 2.0],
                    [3.0, 4.0, 5.0]
                ]
            ])
            """
            inner_str = '[\n'
            if len(self) < 5:
                inner_str += ',\n'.join('{}'.format(
                    _indent(_strip_space(part))) for part in self.parts)
            else:
                inner_str += ',\n'.join('{}'.format(
                    _indent(_strip_space(part))) for part in self.parts[:3])
                inner_str += ',\n    ...\n'
                inner_str += ',\n'.join('{}'.format(
                    _indent(_strip_space(part))) for part in self.parts[-1:])

            inner_str += '\n]'

            return '{!r}.element({})'.format(self.space, inner_str)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
