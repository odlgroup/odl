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

"""Cartesian products of `LinearSpace` instances."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from builtins import range, str, super, zip
from future import standard_library
standard_library.install_aliases()

from numbers import Integral

# External
import numpy as np

# ODL imports
from odl.set.space import LinearSpace, LinearSpaceVector


__all__ = ('ProductSpace', 'ProductSpaceVector')


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

    """Cartesian product of linear spaces."""

    def __init__(self, *spaces, **kwargs):
        """Initialize a new instance.

        The Cartesian product
        :math:`\mathcal{X}_1 \\times \dots \\times \mathcal{X}_n` for
        linear spaces :math:`\mathcal{X}_i` is itself a linear space,
        where the linear combination is defined component-wise.

        Parameters
        ----------
        spaces : `LinearSpace` or `int`
            Can be specified either as a space and an integer, in which
            case the power space ``space**n`` is created, or
            an arbitrary number of spaces.
        kwargs : {'ord', 'weights', 'prod_norm'}
            'ord' : `float`, optional
                Order of the product distance/norm, i.e.

                ``dist(x, y) = np.linalg.norm(x-y, ord=ord)``

                ``norm(x) = np.linalg.norm(x, ord=ord)``

                Default: 2.0

                The following `float` values for ``ord`` can be specified.
                Note that any value of ``ord < 1`` only gives a pseudo-norm.

                +-------------+------------------------------+
                | 'prod_norm' | Distance Definition          |
                +=============+==============================+
                | 'inf'       | ``max(w * z)``               |
                +-------------+------------------------------+
                | '-inf'      | ``min(w * z)``               |
                +-------------+------------------------------+
                | other       | ``sum(w * z**ord)**(1/ord)`` |
                +-------------+------------------------------+

                Here,

                ``z = (x[0].dist(y[0]),..., x[n-1].dist(y[n-1]))``

                and ``w = weights``.

                Note that ``0 <= ord < 1`` are not allowed since these
                pseudo-norms are very unstable numerically.
            'weights' : array-like, optional, only usable with 'ord'
                Array of weights, same size as number of space
                components. All weights must be positive. It is
                multiplied with the tuple of distances before
                applying the Rn norm or ``prod_norm``.

                Default: ``(1.0,...,1.0)``

            'prod_norm' : `callable`, optional
                Function that should be applied to the array of
                distances/norms. Specifying a product norm causes
                the space to NOT be a Hilbert space.

                Default: ``np.linalg.norm(x, ord=ord)``.

        Returns
        -------
        prodspace : `ProductSpace`

        Examples
        --------
        >>> from odl import Rn
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
            if 0 <= order < 1:
                raise ValueError('Cannot use {:.2}-norm due to numerical '
                                 'instability.'.format(order))

            # TODO: handle weights more elegantly
            weights = kwargs.get('weights', None)
            if weights is not None:
                self._weights = np.atleast_1d(weights)
                if not np.all(self._weights > 0):
                    raise ValueError('weights must all be positive')
                if not len(self._weights) == len(spaces):
                    raise ValueError('spaces and weights have different '
                                     'lengths ({} != {}).'
                                     ''.format(len(spaces), len(weights)))

                def w_norm(x):
                    return np.linalg.norm(x * self._weights, ord=order)

                self._prod_norm = w_norm

                if order == 2.0:
                    def w_inner_sum(x):
                        return np.linalg.dot(x, self._weights)

                    self._prod_inner_sum = w_inner_sum
                else:
                    self._prod_inner_sum = _prod_inner_sum_not_defined
            else:
                self._weights = None

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
        inp : `object`, optional
            If ``inp`` is `None`, a new element is created from
            scratch by allocation in the spaces. If ``inp`` is
            already an element in this space, it is re-wrapped.
            Otherwise, a new element is created from the
            components by calling the ``element()`` methods
            in the component spaces.

        Returns
        -------
        element : `ProductSpaceVector`
            The new element

        Examples
        --------
        >>> from odl import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> vec_2, vec_3 = r2.element(), r3.element()
        >>> r2x3 = ProductSpace(r2, r3)
        >>> vec_2x3 = r2x3.element()
        >>> vec_2.space == vec_2x3[0].space
        True
        >>> vec_3.space == vec_2x3[1].space
        True

        Creates an element in the product space

        >>> from odl import Rn
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

        # TODO: how does this differ from "if inp in self"?
        if (all(isinstance(v, LinearSpaceVector) for v in inp) and
                all(part.space == space
                    for part, space in zip(inp, self.spaces))):
            parts = list(inp)
        else:
            # Delegate constructors
            parts = [space.element(arg)
                     for arg, space in zip(inp, self.spaces)]

        return self.element_type(self, parts)

    @property
    def weights(self):
        """Weighting vector or scalar of this product space."""
        return self._weights

    def zero(self):
        """Create the zero vector of the product space.

        The i:th component of the product space zero vector is the
        zero vector of the i:th space in the product.

        Parameters
        ----------
        None

        Returns
        -------
        zero : ProductSpaceVector
            The zero vector in the product space

        Examples
        --------
        >>> from odl import Rn
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

    def one(self):
        """Create the one vector of the product space.

        The i:th component of the product space one vector is the
        one vector of the i:th space in the product.

        Parameters
        ----------
        None

        Returns
        -------
        one : ProductSpaceVector
            The one vector in the product space

        Examples
        --------
        >>> from odl import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> one_2, one_3 = r2.one(), r3.one()
        >>> r2x3 = ProductSpace(r2, r3)
        >>> one_2x3 = r2x3.one()
        >>> one_2 == one_2x3[0]
        True
        >>> one_3 == one_2x3[1]
        True
        """
        return self.element([space.one() for space in self.spaces])

    def _lincomb(self, a, x, b, y, out):
        # pylint: disable=protected-access
        for space, xp, yp, outp in zip(self.spaces, x.parts, y.parts,
                                       out.parts):
            space._lincomb(a, xp, b, yp, outp)

    def _dist(self, x1, x2):
        dists = np.fromiter(
            (spc._dist(x1p, x2p)
             for spc, x1p, x2p in zip(self.spaces, x1.parts, x2.parts)),
            dtype=np.float64, count=self.size)
        return self._prod_norm(dists)

    def _norm(self, x):
        norms = np.fromiter(
            (spc._norm(xp)
             for spc, xp in zip(self.spaces, x.parts)),
            dtype=np.float64, count=self.size)
        return self._prod_norm(norms)

    def _inner(self, x1, x2):
        inners = np.fromiter(
            (spc._inner(x1p, x2p)
             for spc, x1p, x2p in zip(self.spaces, x1.parts, x2.parts)),
            dtype=np.float64, count=self.size)
        return self._prod_inner_sum(inners)

    def _multiply(self, x1, x2, out):
        for spc, xp, yp, outp in zip(self.spaces, x1.parts, x2.parts,
                                     out.parts):
            spc._multiply(xp, yp, outp)

    def __eq__(self, other):
        """``ps.__eq__(other) <==> ps == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `ProductSpace` instance, has
            the same length and the same factors. `False` otherwise.

        Examples
        --------
        >>> from odl import Rn
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
            return (isinstance(other, ProductSpace) and
                    len(self) == len(other) and
                    all(x == y for x, y in zip(self.spaces,
                                               other.spaces)))

    def __len__(self):
        """``ps.__len__() <==> len(ps)``."""
        return self._size

    def __getitem__(self, indices):
        """``ps.__getitem__(indices) <==> ps[indices]``."""

        if isinstance(indices, Integral):
            return self.spaces[indices]
        elif isinstance(indices, slice):
            return ProductSpace(*self.spaces[indices])
        else:
            return ProductSpace(*[self.spaces[i] for i in indices])

    def __str__(self):
        """``ps.__str__() <==> str(ps)``."""
        if all(self.spaces[0] == space for space in self.spaces):
            return '{' + str(self.spaces[0]) + '}^' + str(self.size)
        else:
            return ' x '.join(str(space) for space in self.spaces)

    def __repr__(self):
        """``ps.__repr__() <==> repr(ps)``."""
        if all(self.spaces[0] == space for space in self.spaces):
            return 'ProductSpace({!r}, {})'.format(self.spaces[0],
                                                   self.size)
        else:
            inner_str = ', '.join(repr(space) for space in self.spaces)
            return 'ProductSpace({})'.format(inner_str)

    @property
    def element_type(self):
        """ `ProductSpaceVector` """
        return ProductSpaceVector

class ProductSpaceVector(LinearSpaceVector):
    """ Elements in a `ProductSpace` """

    def __init__(self, space, parts):
        """"Initialize a new instance."""
        super().__init__(space)
        self._parts = list(parts)

    @property
    def parts(self):
        """The parts of this vector."""
        return self._parts

    @property
    def size(self):
        """The number of factors of this vector's space."""
        return self.space.size

    def __eq__(self, other):
        """``ps.__eq__(other) <==> ps == other``.

        Overrides the default `LinearSpace` method since it is
        implemented with the distance function, which is prone to
        numerical errors. This function checks equality per
        component.
        """
        if other not in self.space:
            return False
        elif other is self:
            return True
        else:
            return all(sp == op for sp, op in zip(self.parts, other.parts))

    def __len__(self):
        """``v.__len__() <==> len(v)``."""
        return len(self.space)

    def __getitem__(self, indices):
        """``ps.__getitem__(indices) <==> ps[indices]``."""
        if isinstance(indices, Integral):
            return self.parts[indices]
        elif isinstance(indices, slice):
            return self.space[indices].element(self.parts[indices])
        else:
            out_parts = [self.parts[i] for i in indices]
            return self.space[indices].element(out_parts)

    def __setitem__(self, indices, values):
        """``ps.__setitem__(indcs, vals) <==> ps[indcs] = vals``."""
        try:
            self.parts[indices] = values
        except TypeError:
            for i, index in enumerate(indices):
                self.parts[index] = values[i]

    def __str__(self):
        """``ps.__str__() <==> str(ps)``."""
        inner_str = ', '.join(str(part) for part in self.parts)
        return '{{{}}}'.format(inner_str)

    def __repr__(self):
        """``s.__repr__() <==> repr(s)``.

        Examples
        --------
        >>> from odl import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> r2x3 = ProductSpace(r2, r3)
        >>> x = r2x3.element([[1, 2], [3, 4, 5]])
        >>> eval(repr(x)) == x
        True

        The result is readable:

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
