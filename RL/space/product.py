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

# RL imports
from RL.space.space import (HilbertSpace, NormedSpace, LinearSpace,
                            MetricSpace)
from RL.utility.utility import errfmt

standard_library.install_aliases()


class ProductSpace(LinearSpace):
    """ The Cartesian product of N linear spaces.

    The product X1 x ... x XN is itself a linear space, where the
    linear combination is defined component-wise. Automatically
    selects the most specific subclass possible.

    Parameters
    ----------
    spaces : LinearSpace instances
    kwargs : {'ord', 'weights'}
        Passed to the init function of the subclass

    Returns
    -------
    prodspace : ProductSpace instance
        If `all(isinstance(spc, LinearSpace) for spc in spaces) ->
        `type(prodspace) == LinearProductSpace`\n
        If `all(isinstance(spc, MetricSpace) for spc in spaces) ->
        `type(prodspace) == MetricProductSpace`\n
        If `all(isinstance(spc, NormedSpace) for spc in spaces) ->
        `type(prodspace) == NormedProductSpace`\n
        If `all(isinstance(spc, HilbertSpace) for spc in spaces) ->
        `type(prodspace) == HilbertProductSpace`

    Examples
    --------
    >>> from RL.space.euclidean import RN, EuclideanSpace
    >>> r2x3 = ProductSpace(RN(2), RN(3))
    >>> r2x3.__class__.__name__
    'LinearProductSpace'
    >>> r2x3 = ProductSpace(EuclideanSpace(2), RN(3))
    >>> r2x3.__class__.__name__
    'LinearProductSpace'
    >>> r2x3 = ProductSpace(EuclideanSpace(2), EuclideanSpace(3))
    >>> r2x3.__class__.__name__
    'HilbertProductSpace'
    """

    def __new__(cls, *spaces, **kwargs):
        if not all(spaces[0].field == spc.field for spc in spaces):
            raise TypeError('All spaces must have the same field')

        # Do not change subclass if it was explicitly called
        subs = (LinearProductSpace, NormedProductSpace, MetricProductSpace,
                HilbertProductSpace)
        if cls not in subs:
            if all(isinstance(spc, HilbertSpace) for spc in spaces):
                newcls = HilbertProductSpace
            elif all(isinstance(spc, NormedSpace) for spc in spaces):
                newcls = NormedProductSpace
            elif all(isinstance(spc, MetricSpace) for spc in spaces):
                newcls = MetricProductSpace
            elif all(isinstance(spc, LinearSpace) for spc in spaces):
                newcls = LinearProductSpace
        else:
            newcls = cls

        print('Calling ProductSpace.__new__() with kwargs=', kwargs)
        instance = super().__new__(newcls, **kwargs)

        # __init__ is called automatically only if `newcls` is a
        # subclass of `cls`. This may not always be the case, e.g.
        # if `cls == PowerSpace`
        if not issubclass(newcls, cls):
            instance.__init__(*spaces, **kwargs)

        return instance

    # Dummy methods to overload abstract base class methods
    def empty(self):
        raise NotImplementedError
    field = linCombImpl = empty


class LinearProductSpace(ProductSpace):
    """The Cartesian product of N linear spaces.

    The product X1 x ... x XN is itself a linear space, where the
    linear combination is defined component-wise.

    Parameters
    ----------
    spaces : LinearSpace instances

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
    >>> r2x3 = LinearProductSpace(EuclideanSpace(2), EuclideanSpace(3))
    >>> r2x3.__class__.__name__
    'LinearProductSpace'
    """

    def __init__(self, *spaces, **kwargs):
        if not all(isinstance(spc, LinearSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, LinearSpace)]
            raise TypeError('{} not LinearSpace instance(s)'.format(wrong_spc))

        print('Calling LinearProductSpace.__init__() with kwargs=', kwargs)
        self.spaces = spaces
        self._nfactors = len(self.spaces)
        self._field = spaces[0].field
        super().__init__(**kwargs)

    def zero(self):
        """ Create the zero vector of the product space

        The i:th component of the product space zero vector is the
        zero vector of the i:th space in the product.

        Parameters
        ----------
        None

        Returns
        -------
        zero : LinearProducSpace.Vector
            The zero vector in the product space

        Example
        -------
        >>> from RL.space.euclidean import EuclideanSpace
        >>> r2, r3 = EuclideanSpace(2), EuclideanSpace(3)
        >>> zero_2, zero_3 = r2.zero(), r3.zero()
        >>> r2x3 = ProductSpace(r2, r3)
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
        vec : LinearProducSpace.Vector
            Some vector in the product space

        Example
        -------
        >>> from RL.space.euclidean import EuclideanSpace
        >>> r2, r3 = EuclideanSpace(2), EuclideanSpace(3)
        >>> vec_2, vec_3 = r2.empty(), r3.empty()
        >>> r2x3 = ProductSpace(r2, r3)
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

    @property
    def field(self):
        return self._field

    def equals(self, other):
        """ Test if the product space is equal to another

        Parameters
        ----------
        other : object
            The object to be compared

        Returns
        -------
        equal : boolean
            `True` if `other` is a ProductSpace instance, has same length
            and the same factors. `False` otherwise.

        Example
        -------
        >>> from RL.space.euclidean import EuclideanSpace
        >>> r2, r3 = EuclideanSpace(2), EuclideanSpace(3)
        >>> rn, rm = EuclideanSpace(2), EuclideanSpace(3)
        >>> r2x3, rnxm = ProductSpace(r2, r3), ProductSpace(rn, rm)
        >>> r2x3.equals(rnxm)
        True
        >>> r3x2 = ProductSpace(r3, r2)
        >>> r2x3.equals(r3x2)
        False
        >>> r5 = Productspace(*[EuclideanSpace(1)]*5)
        >>> r2x3.equals(r5)
        False
        >>> r5 = EuclideanSpace(5)
        >>> r2x3.equals(r5)
        False
        """

        return (isinstance(other, ProductSpace) and
                len(self) == len(other) and
                all(x.equals(y) for x, y in zip(self.spaces, other.spaces)))

    def makeVector(self, *args):
        """ Creates an element in the product space

        Parameters
        ----------
        The method has two call patterns, the first is:

        *args : tuple of LinearSpace.Vector's
                A tuple of vectors in the underlying space.
                This will simply wrap the Vectors (not copy).

        The second pattern is to create a new Vector from scratch, in this case

        *args : Argument for i:th vector

        Returns
        -------
        ProductSpace.Vector instance


        Examples
        --------

        >>> r2, r3 = RN(2), RN(3)
        >>> rm = RN(3)
        >>> prod = ProductSpace(rn,rm)
        >>> x = rn.makeVector([1,2])
        >>> y = rm.makeVector([1,2,3])
        >>> z = prod.makeVector(x, y)
        >>> z
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
        """ Access the index:th subspace of this space

        TODO: Move this to the module docstring
        """

        return self.spaces[index]

    def __str__(self):
        return ' x '.join(str(space) for space in self.spaces)

    def __repr__(self):
        return ('ProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')

    class Vector(LinearSpace.Vector):
        def __init__(self, space, *args):
            super().__init__(space)
            self.parts = args

        def __len__(self):
            return len(self.space)

        def __getitem__(self, index):
            """ Access the index:th component of this vector

            TODO: Move this to the module docstring
            """

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
    kwargs : {'ord', 'weights'}
              'ord' : float, optional
                  The order of the norm. Default: 2
              'weights' : array-like, optional
                  Array of weights, same size as number of space
                  components

    The following values for `ord` can be specified.
    Note that any value of ord < 1 only gives a pseudonorm

    =====  ==========================
    ord    Definition
    =====  ==========================
    inf    max(norm(x[0]), ..., norm(x[n-1]))
    -inf   min(norm(x[0]), ..., norm(x[n-1]))
    0      (norm(x[0]) != 0 + ... + norm(x[n-1]) != 0)
    other  (norm(x[0])**ord + ... + norm(x[n-1])**ord)**(1/ord)
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
    """

    # TODO: weights!
    def __init__(self, *spaces, **kwargs):
        if not all(isinstance(spc, MetricSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, MetricSpace)]
            raise TypeError('{} not MetricSpace instance(s)'.format(wrong_spc))

        print('Calling MetricProductSpace.__init__() with kwargs=', kwargs)
        self.ord = kwargs.get('ord', 2)
        self.weights = kwargs.pop('weights', None)
        super().__init__(*spaces, **kwargs)

    def distImpl(self, x, y):
        if self.ord == float('inf'):
            return max(
                space.distImpl(xp, yp)
                for space, xp, yp in zip(self.spaces, x.parts, y.parts))
        elif self.ord == -float('inf'):
            return min(
                space.distImpl(xp, yp)
                for space, xp, yp in zip(self.spaces, x.parts, y.parts))
        elif self.ord == 0:
            return sum(
                space.distImpl(xp, yp) != 0
                for space, xp, yp in zip(self.spaces, x.parts, y.parts))
        else:
            return sum(
                space.distImpl(xp, yp)**self.ord
                for space, xp, yp in zip(self.spaces, x.parts, y.parts)
                )**(1/self.ord)

    def __repr__(self):
        return ('MetricProductSpace(' + ', '.join(str(space)
                for space in self.spaces) + ')')

    class Vector(LinearProductSpace.Vector, MetricSpace.Vector):
        pass


class NormedProductSpace(MetricProductSpace, NormedSpace):
    """The Cartesian product of N normed linear spaces.

    The product X1 x ... x XN is itself a normed space, where the
    linear combination is defined component-wise. The product space
    norm is the R^N norm of the vector of norms of the components, i.e.
    for x = (x1, ..., xN), it is norm(x) = RN(N).norm(z), where
    z = (norm(x1), ..., norm(xN)).

    Parameters
    ----------
    spaces : NormedSpace instances
    kwargs : {'ord', 'weights'}
              'ord' : float, optional
                  The order of the norm. Default: 2
              'weights' : array-like, optional
                  Array of weights, same size as number of space
                  components

    The following values for `ord` can be specified.
    Note that any value of ord < 1 only gives a pseudonorm

    =====  ==========================
    ord    Definition
    =====  ==========================
    inf    max(norm(x[0]), ..., norm(x[n-1]))
    -inf   min(norm(x[0]), ..., norm(x[n-1]))
    0      (norm(x[0]) != 0 + ... + norm(x[n-1]) != 0)
    other  (norm(x[0])**ord + ... + norm(x[n-1])**ord)**(1/ord)
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
    """

    # TODO: weights!
    def __init__(self, *spaces, **kwargs):
        if not all(isinstance(spc, NormedSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces
                         if not isinstance(spc, NormedSpace)]
            raise TypeError('{} not NormedSpace instance(s)'.format(wrong_spc))

        print('Calling NormedProductSpace.__init__() with kwargs=', kwargs)
        self.ord = kwargs.get('ord', 2)
        super().__init__(*spaces, **kwargs)

    def normImpl(self, x):
        if self.ord == float('inf'):
            return max(space.normImpl(xp)
                       for space, xp in zip(self.spaces, x.parts))
        elif self.ord == -float('inf'):
            return min(space.normImpl(xp)
                       for space, xp in zip(self.spaces, x.parts))
        elif self.ord == 0:
            return sum(space.normImpl(xp) != 0
                       for space, xp in zip(self.spaces, x.parts))
        else:
            return sum(
                space.normImpl(xp)**self.ord
                for space, xp in zip(self.spaces, x.parts))**(1/self.ord)

    def __repr__(self):
        return ('NormedProductSpace(' + ', '.join(str(space)
                for space in self.spaces) + ')')

    class Vector(MetricProductSpace.Vector, NormedSpace.Vector):
        pass


class HilbertProductSpace(NormedProductSpace, HilbertSpace):
    """The Cartesian product of N Hilbert spaces.

    The product X1 x ... x XN is itself a (pre-)Hilbert space, where
    the linear combination is defined component-wise. The product space
    inner product is sum of the inner products of the components, i.e.
    for x = (x1, ..., xN) and y = (y1, ..., yN), it is
    inner(x, y) = sum(z), where z = (inner(x1, y1), ..., inner(xN, yN)).

    Parameters
    ----------
    spaces : NormedSpace instances
    kwargs : {'ord', 'weights'}
              'ord' : float, optional
                  The order of the norm. Default: 2
              'weights' : array-like, optional
                  Array of weights, same size as number of space
                  components

    The following values for `ord` can be specified.
    Note that any value of ord < 1 only gives a pseudonorm

    =====  ==========================
    ord    Definition
    =====  ==========================
    inf    max(norm(x[0]), ..., norm(x[n-1]))
    -inf   min(norm(x[0]), ..., norm(x[n-1]))
    0      (norm(x[0]) != 0 + ... + norm(x[n-1]) != 0)
    other  (norm(x[0])**ord + ... + norm(x[n-1])**ord)**(1/ord)
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
    """
    """ A product space of HilbertSpaces (X1 x X2 x ... x Xn)

    Creates a Cartesian product of an arbitrary set of spaces.

    For example:

    `HilbertProductSpace(Reals(), Reals())` is mathematically equivalent to `EuclideanSpace(2)`

    Note that the later is obviously more efficient.

    Parameters
    ----------

    *spaces : HilbertSpace's
              A set of HilbertSpace's
    **kwargs : {'weights'}
                'weights' : Array-Like
                            List of weights, same size as spaces

    The inner product in the HilbertProductSpace is per default defined as

    inner(x, y) = x[0]*y[0] + ... + x[n-1]*y[n-1]

    The optional parameter `weights` changes this behaviour to

    inner(x, y) = weights[0]*x[0]*y[0] + ... + weights[n-1]*x[n-1]*y[n-1]
    """

    def __init__(self, *spaces, **kwargs):
        self.weights = kwargs.pop('weights', None)

        if not all(isinstance(spc, HilbertSpace) for spc in spaces):
            wrong_spc = [spc for spc in spaces if not isinstance(spc, HilbertSpace)]
            raise TypeError('{} not HilbertSpace instance(s)'.format(wrong_spc))

        super().__init__(*spaces, **kwargs)

    def innerImpl(self, x, y):
        if self.weights:
            return sum(space.innerImpl(xp, yp)
                       for space, weight, xp, yp in zip(self.spaces, self.weights, x.parts, y.parts))
        else:
            return sum(space.innerImpl(xp, yp)
                       for space, xp, yp in zip(self.spaces, x.parts, y.parts))

    def __repr__(self):
        return ('HilbertProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')

    class Vector(NormedProductSpace.Vector, HilbertSpace.Vector):
        pass


class PowerSpace(ProductSpace):
    """ Creates a Cartesian "power" of a space. For example,

    `PowerSpace(Reals(),3)` is mathematically the same as `RN(3)` or `R x R x R`
    Note that the later is more efficient.

    Selects the type of product space that has the most structure (Inner product, norm)
    given a set of spaces

    Parameters
    ----------

    underlying_space : LinearSpace
                       The underlying space that should be repeated
    nProduct         : Int
                       Number of products in the result

    Returns
    -------
    ProductSpace, NormedProductSpace or HilbertProductSpace depending on the lowest common denominator in spaces.

    """

    def __new__(cls, base, nfactors, **kwargs):
        if nfactors <= 0:
            raise ValueError(errfmt('''
            'nfactors' must be a positive integer, got {}
            '''.format(nfactors)))

        spaces = [base] * nfactors

        # __init__ is not automatically called since __new__ does not return
        # and instance of 'cls'
        instance = super().__new__(cls, *spaces, **kwargs)
        return instance

    # Dummy methods to overload abstract base class methods
    def empty(self):
        raise NotImplementedError
    field = linCombImpl = empty


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
