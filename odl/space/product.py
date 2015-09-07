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
from __future__ import unicode_literals
from builtins import str, zip, super
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np

# ODL imports
from odl.space.space import LinearSpace
from odl.utility.utility import errfmt


# TODO: adapt product spaces to support function spaces

def _product_space_str(spaces):
    if all(spaces[0] == space for space in spaces):
        return '{' + str(spaces[0]) + '}^' + str(len(spaces))
    else:
        return ' x '.join(str(space) for space in spaces)


def _product_space_repr(spaces):
    if all(spaces[0] == space for space in spaces):
        return 'powerspace(' + str(spaces[0]) + ', ' + str(len(spaces)) + ')'
    else:
        return ('productspace(' +
                ', '.join(repr(space) for space in spaces) + ')')


def _prod_inner_sum_not_defined(x):
    raise NotImplementedError("Inner product not defined with custom product norm")


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
        spaces : LinearSpace instances
            The factors of the product space
        kwargs : {'ord', 'weights', 'prod_norm'}
            'ord' : float, optional
                Order of the product distance/norm, i.e.
                dist(x, y) = np.linalg.norm(x-y, ord=ord)
                norm(x) = np.linalg.norm(x, ord=ord)
                Default: 2.0
            'weights' : array-like, optional, only usable with 'ord' option.
                Array of weights, same size as number of space
                components. All weights must be positive. It is
                multiplied with the tuple of distances before
                applying the Rn norm or 'prod_norm'.
                Default: (1.0,...,1.0)
            'prod_norm' : callable, optional
                Function that should be applied to the array of distances/norms
                Specifying a product norm causes the space to NOT be a hilbert space.
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
        >>> from odl.space.cartesian import Rn, Rn
        >>> r2x3 = ProductSpace(Rn(2), Rn(3))
        >>> r2x3.__class__.__name__
        'ProductSpace'
        >>> r2x3 = ProductSpace(Rn(2), Rn(3))
        >>> r2x3.__class__.__name__
        'ProductSpace'
        """
        if not all(isinstance(spc, LinearSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, LinearSpace)]
            raise TypeError('{} not LinearSpace instance(s)'.format(wrong_spc))

        if not all(spc.field == spaces[0].field for spc in spaces):
            raise TypeError('All spaces must have the same field')

        prod_norm = kwargs.get('prod_norm', None)

        if prod_norm is not None:
            if not callable(prod_norm):
                raise TypeError(errfmt("'prod_norm' must be callable"))

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
                    raise ValueError(errfmt('''
                    'spaces' and 'weights' have different lengths ({} != {})
                    '''.format(len(spaces), len(weights))))

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

        self._spaces = spaces
        self._nfactors = len(spaces)
        self._field = spaces[0].field
        super().__init__()

    @property
    def field(self):
        """The common underlying field of all factors."""
        return self._field

    @property
    def spaces(self):
        """A tuple containing all spaces."""
        return self._spaces

    def element(self, *args, **kwargs):
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
        >>> x = prod.element(x2, x3)
        >>> print(x)
        {[1.0, 2.0], [1.0, 2.0, 3.0]}
        """
        # TODO: update this function!

        # If data is given as keyword arg, prefer it over arg list
        data = kwargs.pop('data', None)
        if data is None:
            if not args:  # No argument at all -> arbitrary vector
                data = elements = [space.element(**kwargs)
                                   for space in self.spaces]
            else:
                data = args

        if not all(isinstance(v, LinearSpace.Vector) for v in data):
            # Delegate constructors
            elements = [space.element(arg, **kwargs)
                        for arg, space in zip(data, self.spaces)]
        else:  # Construct from existing tuple
            if any(part.space != space
                   for part, space in zip(data, self.spaces)):
                raise TypeError(errfmt('''
                The spaces of all parts must correspond to this
                space's parts'''))
            elements = data

        # Use __class__ to allow subclassing
        return self.__class__.Vector(self, *elements)

    def zero(self):
        """Create the zero vector of the product space.

        The i:th component of the product space zero vector is the
        zero vector of the i:th space in the product.

        Parameters
        ----------
        None

        Returns
        -------
        zero : ProducSpace.Vector
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
        return self.element(data=[space.zero() for space in self.spaces])

    def _lincomb(self, z, a, x, b, y):
        # pylint: disable=protected-access
        for space, zp, xp, yp in zip(self.spaces, z.parts, x.parts, y.parts):
            space._lincomb(zp, a, xp, b, yp)

    def _dist(self, x, y):
        dists = np.fromiter(
            (spc._dist(xp, yp)
             for spc, xp, yp in zip(self.spaces, x.parts, y.parts)),
            dtype=np.float64, count=self._nfactors)
        return self._prod_norm(dists)

    def _norm(self, x):
        norms = np.fromiter(
            (spc._norm(xp)
             for spc, xp in zip(self.spaces, x.parts)),
            dtype=np.float64, count=self._nfactors)
        return self._prod_norm(norms)

    def _inner(self, x, y):
        inners = np.fromiter(
            (spc._inner(xp, yp)
             for spc, xp, yp in zip(self.spaces, x.parts, y.parts)),
            dtype=np.float64, count=self._nfactors)
        return self._prod_norm(inners)

    def equals(self, other):
        """Check if the `other` is the same product space.

        Parameters
        ----------
        other : object
            The object to be compared

        Returns
        -------
        equal : boolean
            `True` if `other` is a ProductSpace instance, has
            the same length and the same factors. `False` otherwise.

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> rn, rm = Rn(2), Rn(3)
        >>> r2x3, rnxm = ProductSpace(r2, r3), ProductSpace(rn, rm)
        >>> r2x3.equals(rnxm)
        True
        >>> r3x2 = ProductSpace(r3, r2)
        >>> r2x3.equals(r3x2)
        False
        >>> r5 = ProductSpace(*[Rn(1)]*5)
        >>> r2x3.equals(r5)
        False
        >>> r5 = Rn(5)
        >>> r2x3.equals(r5)
        False
        """
        return (type(self) == type(other) and
                len(self) == len(other) and
                all(x.equals(y) for x, y in zip(self.spaces, other.spaces)))

    def __len__(self):
        """The number of factors."""
        return self._nfactors

    def __getitem__(self, index_or_slice):
        """Implementation of spc[index] and spc[slice]."""
        return self.spaces[index_or_slice]

    def __str__(self):
        return _product_space_str(self.spaces)

    def __repr__(self):
        return _product_space_repr(self.spaces)

    class Vector(LinearSpace.Vector):
        def __init__(self, space, *args):
            super().__init__(space)
            self.parts = args

        def __len__(self):
            return len(self.space)

        def __getitem__(self, index):
            return self.parts[index]

        def __str__(self):
            return ('{' + ', '.join(str(part) for part in self.parts) +
                    '}')

        def __repr__(self):
            return (repr(self.space) + '.element(' +
                    ', '.join(repr(part) for part in self.parts) + ')')


def productspace(*spaces, **kwargs):
    """ Creates a product space X1 x ... x XN

    Selects the 'most powerful' space possible, i.e. if all spaces
    are HilbertSpace instances, a HilbertProductSpace instance is
    returned.

    Parameters
    ----------
    spaces : <Which>Space instances
        <Which> is either Hilbert, Normed, Metric or Linear
    kwargs : {'ord', 'weights', 'prod_norm'}
              'ord' : float, optional
                  Order of the product distance/norm, i.e.
                  dist(x, y) = np.linalg.norm(x-y, ord=ord)
                  norm(x) = np.linalg.norm(x, ord=ord)
                  If used, forces the space to not be a Hilbert space.
                  Default: 2.0
              'weights' : array-like, optional, only usable with the
                          'ord' option.
                  Array of weights, same size as number of space
                  components. All weights must be positive. It is
                  multiplied with the tuple of distances before
                  applying the Rn norm or 'prod_norm'.
                  Default: (1.0,...,1.0)
              'prod_norm' : callable, optional
                  Function that should be applied to the array of
                  distances/norms.
                  If used, forces the space to not be a Hilbert space.
                  Defaults if applicable:
                      dist = np.linalg.norm(x-y, ord=ord)
                      norm = np.linalg.norm(x, ord=ord)
                      inner = np.vdot(x,y)


    Returns
    -------
    prodspace : <Which>ProductSpace instance
        <Which> is either Hilbert, Normed, Metric or Linear

    Remark
    ------
    productspace(Rn(1), Rn(1)) is mathematically equivalent to Rn(2),
    however the latter is usually more efficient numerically.

    See also
    --------
    ProductSpace, MetricProductSpace, NormedProductSpace,
    HilbertProductSpace
    """

    return ProductSpace(*spaces, **kwargs)


def powerspace(base, power, **kwargs):
    """ Creates a power space X^N = X x ... x X

    Selects the 'most powerful' space possible, i.e. if all spaces
    are HilbertSpace instances, a HilbertProductSpace instance is
    returned.

    Parameters
    ----------
    base : <Which>Space instance
        <Which> is either Hilbert, Normed, Metric or Linear
    power : int
        The number of factors in the product
    kwargs : {'ord', 'weights', 'prod_norm'}
              'ord' : float, optional
                  Order of the product distance/norm, i.e.
                  dist(x, y) = np.linalg.norm(x-y, ord=ord)
                  norm(x) = np.linalg.norm(x, ord=ord)
                  If used, forces the space to not be a hilbert space.
                  Default: 2.0
              'weights' : array-like, optional, only usable with 'ord' option.
                  Array of weights, same size as number of space
                  components. All weights must be positive. It is
                  multiplied with the tuple of distances before
                  applying the Rn norm or 'prod_norm'.
                  Default: (1.0,...,1.0)
              'prod_norm' : callable, optional
                  Function that should be applied to the array of
                  distances/norms
                  If used, forces the space to not be a hilbert space.
                  Defaults if applicable:
                      dist = np.linalg.norm(x-y, ord=ord)
                      norm = np.linalg.norm(x, ord=ord)

    Returns
    -------
    prodspace : <Which>ProductSpace instance
        <Which> is either Hilbert, Normed, Metric or Linear

    Remark
    ------
    powerspace(Rn(1), 2) is mathematically equivalent to Rn(2),
    however the latter is usually more efficient numerically.

    See also
    --------
    ProductSpace
    """

    return productspace(*([base] * power), **kwargs)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
