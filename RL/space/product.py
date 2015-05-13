# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
try:
    from builtins import str, zip, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, zip, super
from future import standard_library

# External
import numpy as np

# RL imports
from RL.space.space import HilbertSpace, NormedSpace, MetricSpace, LinearSpace
from RL.utility.utility import errfmt

standard_library.install_aliases()


class LinearProductSpace(LinearSpace):
    """The Cartesian product of N linear spaces.

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
    >>> from RL.space.euclidean import RN, EuclideanSpace
    >>> r2x3 = LinearProductSpace(RN(2), RN(3))
    >>> r2x3.__class__.__name__
    'LinearProductSpace'
    >>> r2x3 = LinearProductSpace(EuclideanSpace(2), RN(3))
    >>> r2x3.__class__.__name__
    'LinearProductSpace'
    """

    def __init__(self, *spaces, **kwargs):
        if not all(isinstance(spc, LinearSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, LinearSpace)]
            raise TypeError('{} not LinearSpace instance(s)'.format(wrong_spc))

        if not all(spc.field == spaces[0].field for spc in spaces):
            raise TypeError('All spaces must have the same field')

        self._spaces = spaces
        self._nfactors = len(self.spaces)
        self._field = spaces[0].field
        super().__init__()

    @property
    def field(self):
        return self._field

    @property
    def spaces(self):
        return self._spaces

    def zero(self):
        """ Create the zero vector of the product space

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
        >>> from RL.space.euclidean import EuclideanSpace
        >>> r2, r3 = EuclideanSpace(2), EuclideanSpace(3)
        >>> zero_2, zero_3 = r2.zero(), r3.zero()
        >>> r2x3 = LinearProductSpace(r2, r3)
        >>> zero_2x3 = r2x3.zero()
        >>> r2.norm(zero_2 - zero_2x3[0]) == 0
        True
        >>> r3.norm(zero_3 - zero_2x3[1]) == 0
        True
        """

        return self.makeVector(*[space.zero() for space in self.spaces])

    def empty(self):
        """ Create some vector in the product space

        The main purpose of this function is to allocate memory

        Parameters
        ----------
        None

        Returns
        -------
        vec : ProducSpace.Vector
            Some vector in the product space

        Example
        -------
        >>> from RL.space.euclidean import EuclideanSpace
        >>> r2, r3 = EuclideanSpace(2), EuclideanSpace(3)
        >>> vec_2, vec_3 = r2.empty(), r3.empty()
        >>> r2x3 = LinearProductSpace(r2, r3)
        >>> vec_2x3 = r2x3.empty()
        >>> vec_2.space == vec_2x3[0].space
        True
        >>> vec_3.space == vec_2x3[1].space
        True
        """

        return self.makeVector(*[space.empty() for space in self.spaces])

    def linCombImpl(self, z, a, x, b, y):
        for space, zp, xp, yp in zip(self.spaces, z.parts, x.parts, y.parts):
            space.linCombImpl(zp, a, xp, b, yp)

    def equals(self, other):
        """ Test if the product space is equal to another

        Parameters
        ----------
        other : object
            The object to be compared

        Returns
        -------
        equal : boolean
            `True` if `other` is a LinearProductSpace instance, has same length
            and the same factors. `False` otherwise.

        Example
        -------
        >>> from RL.space.euclidean import EuclideanSpace
        >>> r2, r3 = EuclideanSpace(2), EuclideanSpace(3)
        >>> rn, rm = EuclideanSpace(2), EuclideanSpace(3)
        >>> r2x3, rnxm = LinearProductSpace(r2, r3), LinearProductSpace(rn, rm)
        >>> r2x3.equals(rnxm)
        True
        >>> r3x2 = LinearProductSpace(r3, r2)
        >>> r2x3.equals(r3x2)
        False
        >>> r5 = LinearProductSpace(*[EuclideanSpace(1)]*5)
        >>> r2x3.equals(r5)
        False
        >>> r5 = EuclideanSpace(5)
        >>> r2x3.equals(r5)
        False
        """

        return (isinstance(other, LinearProductSpace) and
                len(self) == len(other) and
                all(x.equals(y) for x, y in zip(self.spaces, other.spaces)))

    def makeVector(self, *args):
        """ Creates an element in the product space

        Parameters
        ----------
        The method has two call patterns, the first is:

        args : tuple of LinearSpace.Vector's
            A tuple of vectors in the underlying space.
            This will simply wrap the Vectors (not copy).

        The second pattern is to create a new Vector from scratch, in
        this case

        args : tuple of array-like objects

        Returns
        -------
        LinearProductSpace.Vector instance


        Example
        -------
        >>> r2, r3 = RN(2), RN(3)
        >>> prod = LinearProductSpace(r2, r3)
        >>> x2 = r2.makeVector([1, 2])
        >>> x3 = r3.makeVector([1, 2, 3])
        >>> x = prod.makeVector(x2, x3)
        >>> print(x)
        {[ 1.  2.], [ 1.  2.  3.]}
        """

        if not isinstance(args[0], LinearSpace.Vector):
            # Delegate constructors
            return self.makeVector(*(space.makeVector(arg)
                                     for arg, space in zip(args, self.spaces)))
        else:  # Construct from existing tuple
            if any(part.space != space
                   for part, space in zip(args, self.spaces)):
                raise TypeError(errfmt('''
                The spaces of all parts must correspond to this
                space's parts'''))

            # Use __class__ to allow subclassing
            return self.__class__.Vector(self, *args)

    def __len__(self):
        return self._nfactors

    def __getitem__(self, index):
        return self.spaces[index]

    def __str__(self):
        return ' x '.join(str(space) for space in self.spaces)

    def __repr__(self):
        return ('LinearProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')

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
            return (repr(self.space) + '.makeVector(' +
                    ', '.join(repr(part) for part in self.parts) + ')')


class MetricProductSpace(LinearProductSpace, MetricSpace):
    """The Cartesian product of N metric linear spaces.

    The product X1 x ... x XN is itself a metric space, where the
    linear combination is defined component-wise. The product space
    distance is the R^N norm of the vector of distances of the
    components, i.e. for x = (x1, ..., xN) and y = (y1, ..., yN),
    it is dist(x, y) = RN(N).norm(z), where
    z = (dist(x1, y1), ..., dist(xN, yN)).

    Parameters
    ----------
    spaces : MetricSpace instances
        The factors of the product space
    kwargs : {'ord', 'weights'}
              'ord' : float, optional
                  The order of the norm. Default: 2
              'weights' : array-like, optional
                  Array of weights, same size as number of space
                  components.  All weights must be positive.
                  Default: (1,...,1)

    The following values for `ord` can be specified.
    Note that any value of ord < 1 only gives a pseudonorm

    =====  ==========================
    ord    Distance Definition [z = (dist(x[0], y[0]),...,\
     dist(x[n-1], y[n-1]))]
    =====  ==========================
    inf    max(z)
    -inf   min(z)
    0      sum(z != 0)
    other  sum(z**ord)**(1/ord) -- unweighted\n
           sum(w * z**ord)**(1/ord) -- weighted
    =====  ==========================

    Returns
    -------
    prodspace : MetricProductSpace instance

    Examples
    --------
    >>> from RL.space.euclidean import EuclideanSpace
    >>> r2, r3 = EuclideanSpace(2), EuclideanSpace(3)
    >>> r2x3 = MetricProductSpace(r2, r3, ord=float('inf'))
    >>> x_2 = r2.makeVector([2.0, 3.5])
    >>> y_2 = r2.makeVector([2.0, 3.5])
    >>> x_3 = r3.makeVector([-0.3, 2.0, 3.5])
    >>> y_3 = r3.makeVector([-1.1, 7.2, 3.5])
    >>> x = r2x3.makeVector(x_2, x_3)
    >>> y = r2x3.makeVector(y_2, y_3)
    >>> r2x3.dist(x, y)
    <Value>
    >>> r2x3.dist(x, y) == max((r2.dist(x_2, x_3), r3.dist(x_3, y_3)))
    True
    >>> r2x3.dist(x, y) == x.dist(y) == y.dist(x)
    True
    >>> # TODO: weights
    """

    def __init__(self, *spaces, **kwargs):
        if not all(isinstance(spc, MetricSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, MetricSpace)]
            raise TypeError('{} not MetricSpace instance(s)'.format(wrong_spc))

        weights = kwargs.get('weights', None)
        if weights is not None:
            weights = np.atleast_1d(weights)
            if not np.all(weights > 0):
                raise ValueError('weights must all be positive')
            if not len(weights) == len(spaces):
                raise ValueError(errfmt('''
                'spaces' and 'weights' have different lengths ({} != {})
                '''.format(len(spaces), len(weights))))
        else:
            weights = np.ones(len(spaces))

        self.ord = float(kwargs.get('ord', 2))
        self.weights = weights
        super().__init__(*spaces, **kwargs)

    def distImpl(self, x, y):
        if self.ord == float('inf'):
            return max(
                spc.distImpl(xp, yp) * w
                for spc, w, xp, yp in zip(self.spaces, self.weights,
                                          x.parts, y.parts))
        elif self.ord == -float('inf'):
            return min(
                spc.distImpl(xp, yp) * w
                for spc, w, xp, yp in zip(self.spaces, self.weights,
                                          x.parts, y.parts))
        elif self.ord == 0:
            return sum(
                spc.distImpl(xp, yp) != 0
                for spc, xp, yp in zip(self.spaces, x.parts, y.parts))

        else:
            return sum(
                spc.distImpl(xp, yp)**self.ord * w
                for spc, w, xp, yp in zip(self.spaces, self.weights,
                                          x.parts, y.parts)
                )**(1/self.ord)

    def __repr__(self):
        return ('MetricProductSpace(' + ', '.join(
            str(space) for space in self.spaces) + ')')

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
    kwargs : {'ord', 'weights'}
              'ord' : float, optional
                  The order of the norm. Default: 2
              'weights' : array-like, optional
                  Array of weights, same size as number of space
                  components. All weights must be positive.
                  Default: (1,...,1)

    The following values for 'ord' can be specified.
    Note that any value of ord < 1 only gives a pseudonorm

    =====  ==========================
    ord    Norm Definition [z = (norm(x[0]),..., norm(x[n-1]))]
    =====  ==========================
    inf    max(z)
    -inf   min(z)
    0      sum(z != 0)
    other  sum(z**ord)**(1/ord) -- unweighted\n
           sum(w * z**ord)**(1/ord) -- weighted
    =====  ==========================

    Returns
    -------
    prodspace : NormedProductSpace instance

    Examples
    --------
    >>> from RL.space.euclidean import EuclideanSpace
    >>> r2, r3 = EuclideanSpace(2), EuclideanSpace(3)
    >>> r2x3 = NormedProductSpace(r2, r3, ord=float('inf'))
    >>> x_2 = r2.makeVector([2.0, 3.5])
    >>> x_3 = r3.makeVector([-0.3, 2.0, 3.5])
    >>> x = r2x3.makeVector(x_2, x_3)
    >>> r2x3.norm(x)
    4.042276586281547
    >>> r2x3.norm(x) == max((r2.norm(x_2), r3.norm(x_3)))
    True
    >>> r2x3.norm(x) == x.norm()
    True
    >>> w2x3 = NormedProductSpace(r2, r3, ord=1, weights=[0.2, 0.8])
    >>> w2x3.norm(x)
    <value>
    """

    def __init__(self, *spaces, **kwargs):
        if not all(isinstance(spc, NormedSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, NormedSpace)]
            raise TypeError('{} not NormedSpace instance(s)'.format(wrong_spc))

        weights = kwargs.get('weights', None)
        if weights is not None:
            weights = np.atleast_1d(weights)
            if not np.all(weights > 0):
                raise ValueError('weights must all be positive')
            if not len(weights) == len(spaces):
                raise ValueError(errfmt('''
                'spaces' and 'weights' have different lengths ({} != {})
                '''.format(len(spaces), len(weights))))
        else:
            weights = np.ones(len(spaces))

        self.ord = float(kwargs.get('ord', 2))
        self.weights = weights
        super().__init__(*spaces, **kwargs)

    def normImpl(self, x):
        if self.ord == float('inf'):
            return max(spc.normImpl(xp) * w
                       for spc, w, xp in zip(self.spaces, self.weights,
                                             x.parts))
        elif self.ord == -float('inf'):
            return min(spc.normImpl(xp) * w
                       for spc, w, xp in zip(self.spaces, self.weights,
                                             x.parts))
        elif self.ord == 0:
            return sum(space.normImpl(xp) != 0
                       for space, xp in zip(self.spaces, x.parts))
        if self.weights is not None:
            return sum(
                spc.normImpl(xp)**self.ord * w
                for spc, w, xp in zip(self.spaces, self.weights,
                                      x.parts)
                )**(1/self.ord)

    def __repr__(self):
        return ('NormedProductSpace(' + ', '.join(
            str(space) for space in self.spaces) + ')')

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
    spaces : NormedSpace instances\n
    kwargs : {'weights'}
              'weights' : array-like, optional
                  Array of weights, same size as number of space
                  components. All weights must be positive.
                  Default: (1,...,1)

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
        else:
            weights = np.ones(len(spaces))

        self.weights = weights
        super().__init__(*spaces, **kwargs)

    def innerImpl(self, x, y):
        return sum(
            spc.innerImpl(xp, yp) * w
            for spc, w, xp, yp in zip(self.spaces, self.weights,
                                      x.parts, y.parts))

    def __repr__(self):
        return ('HilbertProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')

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
    kwargs : {'ord', 'weights'}
        'ord' : float, optional
            The order of the R^N norm. Default: 2
            see MetricProductSpace or NormedProductSpace
        'weights' : array-like, optional
            Array of weights, same size as number of space
            components. All weights must be positive.
            Default: (1,...,1)
            See MetricProductSpace, NormedProductSpace or
            HilbertProductSpace

    Returns
    -------
    prodspace : <Which>ProductSpace instance
        <Which> is either Hilbert, Normed, Metric or Linear

    Remark
    ------
    productspace(RN(1), RN(1)) is mathematically equivalent to RN(2),
    however the latter is much more efficient numerically.

    See also
    --------
    LinearProductSpace, MetricProductSpace, NormedProductSpace,
    HilbertProductSpace
    """

    if all(isinstance(spc, HilbertSpace) for spc in spaces):
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
    kwargs : {'ord', 'weights'}
        'ord' : float, optional
            The order of the R^N norm. Default: 2
            see MetricProductSpace or NormedProductSpace
        'weights' : array-like, optional
            Array of weights, same size as number of space
            components. All weights must be positive.
            Default: (1,...,1)
            See MetricProductSpace, NormedProductSpace or
            HilbertProductSpace

    Returns
    -------
    prodspace : <Which>ProductSpace instance
        <Which> is either Hilbert, Normed, Metric or Linear

    Remark
    ------
    powerspace(RN(1), 2) is mathematically equivalent to RN(2),
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
