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

"""Cartesian products of `LinearSpace`s.

TODO: document public interface
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import str, zip, super
from future import standard_library

# External
import numpy as np

# ODL imports
from odl.space.space import HilbertSpace, NormedSpace, MetricSpace, LinearSpace
from odl.utility.utility import errfmt

standard_library.install_aliases()


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


class LinearProductSpace(LinearSpace):

    """The Cartesian product of N linear spaces.

    The product X1 x ... x XN is itself a linear space, where the
    linear combination is defined component-wise.

    TODO: document public interface
    """

    def __init__(self, *spaces, **_):
        """Initialize a new LinearProductSpace.

        The product X1 x ... x XN is itself a linear space, where the
        linear combination is defined component-wise.

        Parameters
        ----------
        spaces : LinearSpace instances
            The factors of the product space

        Returns
        -------
        prodspace : LinearProductSpace instance

        Examples
        --------
        >>> from odl.space.cartesian import Rn, En
        >>> r2x3 = LinearProductSpace(Rn(2), Rn(3))
        >>> r2x3.__class__.__name__
        'LinearProductSpace'
        >>> r2x3 = LinearProductSpace(En(2), Rn(3))
        >>> r2x3.__class__.__name__
        'LinearProductSpace'
        """
        if not all(isinstance(spc, LinearSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, LinearSpace)]
            raise TypeError('{} not LinearSpace instance(s)'.format(wrong_spc))

        if not all(spc.field == spaces[0].field for spc in spaces):
            raise TypeError('All spaces must have the same field')

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
        LinearProductSpace.Vector instance

        Examples
        -------
        >>> from odl.space.cartesian import En
        >>> r2, r3 = En(2), En(3)
        >>> vec_2, vec_3 = r2.element(), r3.element()
        >>> r2x3 = LinearProductSpace(r2, r3)
        >>> vec_2x3 = r2x3.element()
        >>> vec_2.space == vec_2x3[0].space
        True
        >>> vec_3.space == vec_2x3[1].space
        True

        Creates an element in the product space
        >>> from odl.space.cartesian import Rn
        >>> r2, r3 = Rn(2), Rn(3)
        >>> prod = LinearProductSpace(r2, r3)
        >>> x2 = r2.element([1, 2])
        >>> x3 = r3.element([1, 2, 3])
        >>> x = prod.element(x2, x3)
        >>> print(x)
        {[1.0, 2.0], [1.0, 2.0, 3.0]}
        """
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

        Example
        -------
        >>> from odl.space.cartesian import En
        >>> r2, r3 = En(2), En(3)
        >>> zero_2, zero_3 = r2.zero(), r3.zero()
        >>> r2x3 = LinearProductSpace(r2, r3)
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

    def equals(self, other):
        """Check if the `other` is the same product space.

        Parameters
        ----------
        other : object
            The object to be compared

        Returns
        -------
        equal : boolean
            `True` if `other` is a LinearProductSpace instance, has
            the same length and the same factors. `False` otherwise.

        Example
        -------
        >>> from odl.space.cartesian import En
        >>> r2, r3 = En(2), En(3)
        >>> rn, rm = En(2), En(3)
        >>> r2x3, rnxm = LinearProductSpace(r2, r3), LinearProductSpace(rn, rm)
        >>> r2x3.equals(rnxm)
        True
        >>> r3x2 = LinearProductSpace(r3, r2)
        >>> r2x3.equals(r3x2)
        False
        >>> r5 = LinearProductSpace(*[En(1)]*5)
        >>> r2x3.equals(r5)
        False
        >>> r5 = En(5)
        >>> r2x3.equals(r5)
        False
        """
        return (isinstance(other, LinearProductSpace) and
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


class MetricProductSpace(LinearProductSpace, MetricSpace):

    """The Cartesian product of N metric linear spaces.

    The product X1 x ... x XN is itself a metric space, where the
    linear combination is defined component-wise. The product space
    distance is the R^N norm of the vector of distances of the
    components, i.e. for x = (x1, ..., xN) and y = (y1, ..., yN),
    it is dist(x, y) = Rn(N).norm(z), where
    z = (dist(x1, y1), ..., dist(xN, yN)).

    TODO: document public interface

    """

    # TODO: harmonize notation with Rn

    def __init__(self, *spaces, **kwargs):
        """Initialize a new MetricProductSpace.

        Parameters
        ----------
        spaces : MetricSpace instances
            The factors of the product space
        kwargs : {'ord', 'weights', 'prod_norm'}
            'ord' : float, optional
                Order of the product distance, i.e.
                dist(x, y) = np.linalg.norm(x-y, ord=ord)
                Default: 2.0
            'weights' : array-like, optional, only usable with 'ord' option.
                Array of weights, same size as number of space
                components. All weights must be positive. It is
                multiplied with the tuple of distances before
                applying the Rn norm or 'prod_norm'.
                Default: (1.0,...,1.0)
            'prod_norm' : callable, optional
                Function that should be applied to the array of distances
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
        prodspace : MetricProductSpace instance

        Examples
        --------
        >>> from odl.space.cartesian import En
        >>> r2, r3 = En(2), En(3)
        >>> r2x3 = MetricProductSpace(r2, r3, ord='inf')
        >>> x_2 = r2.element([0, 0])
        >>> y_2 = r2.element([3, 4])
        >>> x_3 = r3.element([0, 0, 0])
        >>> y_3 = r3.element([1, 2, 2])
        >>> x = r2x3.element(x_2, x_3)
        >>> y = r2x3.element(y_2, y_3)
        >>> r2x3.dist(x, y)
        5.0
        >>> r2x3.dist(x, y) == max((r2.dist(x_2, y_2), r3.dist(x_3, y_3)))
        True
        >>> r2x3.dist(x, y) == r2x3.dist(y, x)
        True

        """
        if not all(isinstance(spc, MetricSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, MetricSpace)]
            raise TypeError('{} not MetricSpace instance(s)'.format(wrong_spc))

        prod_norm = kwargs.get('prod_norm', None)

        if prod_norm is not None:
            if callable(prod_norm):
                self._prod_norm = prod_norm
            else:
                raise TypeError(errfmt("'prod_norm' must be callable"))
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
            else:
                def norm(x):
                    return np.linalg.norm(x, ord=order)

                self._prod_norm = norm

        super().__init__(*spaces, **kwargs)

    def _dist(self, x, y):
        dists = np.fromiter(
            (spc._dist(xp, yp)
             for spc, xp, yp in zip(self.spaces, x.parts, y.parts)),
            dtype=np.float64, count=self._nfactors)
        return self._prod_norm(dists)

    class Vector(LinearProductSpace.Vector, MetricSpace.Vector):
        pass


class NormedProductSpace(MetricProductSpace, NormedSpace):
    """The Cartesian product of N normed linear spaces.

    The product X1 x ... x XN is itself a normed space, where the
    linear combination is defined component-wise. The product space
    norm is the R^N norm of the vector of norms of the components.

    If weights are provided, a weighted R^N norm is applied to the
    vector of component norms. All weight entries must be
    positive since norm() does not define a norm otherwise.

    Parameters
    ----------
    spaces : NormedSpace instances
             The factors of the product space
    kwargs : {'ord', 'weights', 'prod_norm'}
              'ord' : float, optional
                  Order of the product distance, i.e.
                  dist(x, y) = np.linalg.norm(x-y, ord=ord)
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
                  distances.
                  Default: np.linalg.norm(x, ord=ord)

    The following float values for `prod_norm` can be specified.
    Note that any value of ord < 1 only gives a pseudonorm.

    =========  ==========================
    prod_norm    Product Norm Definition
    =========  ==========================
    'inf'       max(w * z)
    '-inf'      min(w * z)
    0           sum(w * z != 0)
    other       sum(w * z**ord)**(1/ord)
    =========  ==========================

    Here, z = (x[0].norm(),..., x[n-1].norm()) and w = weights.

    Returns
    -------
    prodspace : NormedProductSpace instance

    Examples
    --------
    >>> from odl.space.cartesian import En
    >>> r2, r3 = En(2), En(3)
    >>> r2x3 = NormedProductSpace(r2, r3, ord='inf')
    >>> x_2 = r2.element([3, 4])
    >>> x_3 = r3.element([2, 2, 1])
    >>> x = r2x3.element(x_2, x_3)
    >>> r2x3.norm(x)
    5.0
    >>> r2x3.norm(x) == max((r2.norm(x_2), r3.norm(x_3)))
    True
    >>> w2x3 = NormedProductSpace(r2, r3, ord=1, weights=[0.2, 1])
    >>> w2x3.norm(x)
    4.0
    >>> w2x3.norm(x) == 0.2*r2.norm(x_2) + 1.0*r3.norm(x_3)
    True
    """

    def __init__(self, *spaces, **kwargs):
        if not all(isinstance(spc, NormedSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, NormedSpace)]
            raise TypeError('{} not NormedSpace instance(s)'.format(wrong_spc))

        super().__init__(*spaces, **kwargs)

    def _norm(self, x):
        norms = np.fromiter(
            (spc._norm(xp)
             for spc, xp in zip(self.spaces, x.parts)),
            dtype=np.float64, count=self._nfactors)
        return self._prod_norm(norms)

    class Vector(MetricProductSpace.Vector, NormedSpace.Vector):
        pass


class HilbertProductSpace(NormedProductSpace, HilbertSpace):
    """The Cartesian product of N Hilbert spaces.

    The product X1 x ... x XN is itself a (pre-)Hilbert space, where
    the linear combination is defined component-wise. The product space
    inner product is sum of the inner products of the components, i.e.
    for x = (x1,..., xN) and y = (y1,..., yN), it is
    inner(x, y) = sum(z), where z = (inner(x1, y1),..., inner(xN, yN)).

    If weights w = (w1,..., wN) are provided,
    inner(x, y) = sum(w * z) instead. All weight entries must be
    positive since inner() does not define an inner product otherwise.

    Parameters
    ----------
    spaces : HilbertSpace instances
             The factors of the product space
    kwargs : {'weights'}
              'weights' : array-like, optional, only usable with the
                          'ord' option.
                  Array of weights, same size as number of space
                  components. All weights must be positive.
                  Default: (1.0,...,1.0)

    Returns
    -------
    prodspace : HilbertProductSpace instance

    Examples
    --------
    TODO
    """

    def __init__(self, *spaces, **kwargs):
        if not all(isinstance(spc, HilbertSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, HilbertSpace)]
            raise TypeError(errfmt('''
            {} not HilbertSpace instance(s)'''.format(wrong_spc)))

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
                return np.linalg.norm(x*weights)

            def w_inner_sum(x):
                return np.dot(x, weights)

            self._prod_norm = w_norm
            self._prod_inner_sum = w_inner_sum
        else:
            self._prod_norm = np.linalg.norm
            self._prod_inner_sum = np.sum

        if 'ord' in kwargs or 'prod_norm' in kwargs:
            raise ValueError(errfmt('''
            Cannot provide 'prod_norm' or 'ord' for hilbert space'''))

        super().__init__(*spaces, **kwargs)

    def _inner(self, x, y):
        inners = np.fromiter(
            (spc._inner(xp, yp)
             for spc, xp, yp in zip(self.spaces, x.parts, y.parts)),
            dtype=np.float64, count=self._nfactors)
        return self._prod_norm(inners)

    class Vector(NormedProductSpace.Vector, HilbertSpace.Vector):
        pass


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
    LinearProductSpace, MetricProductSpace, NormedProductSpace,
    HilbertProductSpace
    """

    if ('ord' not in kwargs and 'prod_norm' not in kwargs and
            all(isinstance(spc, HilbertSpace) for spc in spaces)):
        return HilbertProductSpace(*spaces, **kwargs)
    elif all(isinstance(spc, NormedSpace) for spc in spaces):
        return NormedProductSpace(*spaces, **kwargs)
    elif all(isinstance(spc, MetricSpace) for spc in spaces):
        return MetricProductSpace(*spaces, **kwargs)
    else:
        return LinearProductSpace(*spaces, **kwargs)


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
    LinearProductSpace, MetricProductSpace, NormedProductSpace,
    HilbertProductSpace
    """

    return productspace(*([base] * power), **kwargs)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
