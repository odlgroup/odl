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

"""General vector spaces.

General vector spaces with varying amounts of structure such
as metric, norm, inner product.
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import object, str, super
from future import standard_library
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta, abstractmethod, abstractproperty
from math import sqrt

# RL imports
from RL.space.set import Set
from RL.utility.utility import errfmt

standard_library.install_aliases()

__all__ = ['LinearSpace', 'MetricSpace', 'NormedSpace', 'HilbertSpace',
           'Algebra']


class LinearSpace(Set):

    """Abstract linear space.

    Introduction
    ------------
    A linear space is a set with two operations:

    Addition, `+`
    Scalar multiplication `*`

    The set is closed under these operations (the result still belongs
    to the set).

    Linear Spaces in RL
    -------------------
    In RL the two operations are supplied using the fused "lincomb"
    method, inspired from RVL (Rice Vector Library).

    What follows is a short introduction of the methods that each space
    has to implement.

    Linear Combination
    ~~~~~~~~~~~~~~~~~~
    The method `lincomb` is defined as:

    ``
    lincomb(z, a, x, b, y)    < == >    z = a*x + b*y
    ``

    where `x`, `y` and `z` are vectors in the space, and `a` and `b`
    are scalars.

    RL allows `x`, `y` and `z` to be aliased, i.e. they may be the same
    vector.

    Using this many of the usual vector arithmetic operations can be
    performed, for example

    | Mathematical | Linear combination      |
    |--------------|-------------------------|
    | z = 0        | lincomb(z,  0, z, 0, z) |
    | z = x        | lincomb(z,  1, x, 0, x) |
    | z = -x       | lincomb(z, -1, x, 0, x) |
    | z = x + y    | lincomb(z,  1, x, 1, y) |
    | z = 3*z      | lincomb(z,  3, z, 0, z) |

    To aid in rapid prototyping, an implementer needs only implement
    lincomb, and RL then provides all of the standard mathematical
    operators

    `+`, `*`, `-`, `/`

    as well as their in-place counterparts

    `+=`, `*`, `-`, `/`

    Constructing Elements
    ~~~~~~~~~~~~~~~~~~~~~
    RL also requires the existence of an `element()` method. This method
    can be used to construct a vector in the space, which may be
    assigned to.

    Using this method RL provides some auxiliary methods, such as
    `zero()`, which returns a zero vector in the space.

    ``
    x = space.zero()

    Is the same as

    x = space.element()
    space.lincomb(x, 0, x, 0, x)
    ``

    Field of a space
    ~~~~~~~~~~~~~~~~
    Each space also needs to provide access to its underlying field.
    This field is an instance of `RL.space.set.AbstractSet` and allows
    space to test scalars for validity. For example, in a real space,
    trying to multiply a vector by a complex number will yield an
    error.

    Modifying other methods
    -----------------------
    LinearSpace provides several other methods which have default
    implementations provided using the above mentioned methods.

    A subclass may want to modify these for performance reasons.
    However, be advised that modification should be done in a
    consistent manner. For example, if a space modifies `+`, it should
    also modify `+=`, `-` and `-=`.
    """

    @abstractmethod
    def element(self, data=None):
        """Create an arbitrary element or an element from given data.

        If called without 'data' argument, an arbitrary element in the
        space is generated without guarantee of its state.

        Parameters
        ----------
        data : object, optional
            The data from which to create the element

        Returns
        -------
        v : Vector
            A vector in this space
        """

    @abstractmethod
    def _lincomb(self, z, a, x, b, y):
        """Calculate z = a*x + b*y.

        This method is intended to be private, public callers should
        resort to lincomb which is type-checked.
        """

    @abstractproperty
    def field(self):
        """The underlying field of the vector space."""
        pass

    # Also abstract equals(self,other) from set

    # Default implemented operators
    def zero(self):
        """A zero vector in this space.

        The zero vector is defined as the additive unit of a space.

        Parameters
        ----------
        None

        Returns
        -------
        v : Vector
            The zero vector of this space
        """
        # Default implementation using lincomb
        tmp = self.element()
        self._lincomb(tmp, 0, tmp, 0, tmp)
        return tmp

    def contains(self, x):
        """Check for membership in space.

        Parameters
        ----------
        x : object
            Any object

        Returns
        -------
        result : Boolean
                 True if x is a member of this space.

        Notes
        -----

        Subclasses
        ~~~~~~~~~~
        If X is a subclass of Y, then `Y.contains(X.vector(...))`
        returns True.
        """

        return isinstance(x, LinearSpace.Vector) and x.space.equals(self)

    # Overload for `vec in space` syntax
    def __contains__(self, other):
        """Implementation of 'x in ...' syntax."""
        return self.contains(other)

    # Error checking variant of methods
    def lincomb(self, z, a, x, b=None, y=None):
        """Linear combination of vectors.

        Calculates

        z = a*x
        or if b and y are given
        z = a*x + b*y

        with error checking of types.

        Parameters
        ----------
        z : Vector
            The Vector that the result should be written to.
        a : Scalar in the field of this space
            Scalar to multiply `x` with.
        x : Vector
            The first of the summands
        b : Scalar, optional
            Scalar to multiply `y` with.
        y : Vector, optional
            The second of the summands

        Returns
        -------
        None

        Notes
        -----
        Some notes and examples

        Alignment
        ~~~~~~~~~
        The vectors `z`, `x` and `y` may be aligned, thus a call

        space.lincomb(x, 2, x, 3.14, x)

        is (mathematically) equivalent to

        x = x * (1 + 2 + 3.14)
        """
        if not self.contains(z):
            raise TypeError(errfmt('''
            lincomb failed, z ({}) is not in space ({})
            '''.format(repr(z), repr(self))))

        if not self.field.contains(a):
            raise TypeError(errfmt('''
            lincomb failed, a ({}) is not in field ({})
            '''.format(repr(a), repr(self.field))))

        if not self.contains(x):
            raise TypeError(errfmt('''
            lincomb failed, x ({}) is not in space ({})
            '''.format(repr(x), repr(self))))

        if b is None:  # Single argument
            if y is not None:
                raise ValueError(errfmt('''
                lincomb failed, y ({}) provided but not b'''.format(repr(y))))

            # Call method
            return self._lincomb(z, a, x, 0, x)
        else:  # Two arguments
            if not self.field.contains(b):
                raise TypeError(errfmt('''
                lincomb failed, b ({}) is not in field ({})
                '''.format(repr(b), repr(self.field))))
            if not self.contains(y):
                raise TypeError(errfmt('''
                lincomb failed, y ({}) is not in space ({})
                '''.format(repr(y), repr(self))))

            # Call method
            return self._lincomb(z, a, x, b, y)

    class Vector(with_metaclass(ABCMeta, object)):

        """Abstract vector, an element in the linear space."""

        def __init__(self, space):
            """Default initializer of vectors.

            All deriving classes must call this method to set space.
            """
            if not isinstance(space, LinearSpace):
                raise TypeError(errfmt('''
                'space' ({}) is not a LinearSpace instance'''.format(space)))
            self._space = space

        @property
        def space(self):
            """The space this vector belongs to."""
            return self._space

        # Convenience functions
        def assign(self, other):
            """Assign the values of other to this vector."""
            self.space.lincomb(self, 1, other)

        def copy(self):
            """Create an identical (deep) copy of this vector."""
            result = self.space.element()
            result.assign(self)
            return result

        def lincomb(self, a, x, b=None, y=None):
            """Assign a linear combination to this vector.

            Implemented as space.lincomb(self, a, x, b, y).
            """
            self.space.lincomb(self, a, x, b, y)

        def set_zero(self):
            """Set this vector to the zero vector."""
            self.space.lincomb(self, 0, self, 0, self)

        # Convenience operators
        def __iadd__(self, other):
            """Implementation of 'self += other'."""
            self.space.lincomb(self, 1, self, 1, other)
            return self

        def __isub__(self, other):
            """Implementation of 'self -= other'."""
            self.space.lincomb(self, 1, self, -1, other)
            return self

        def __imul__(self, scalar):
            """Implementation of 'self *= scalar'."""
            self.space.lincomb(self, scalar, self)
            return self

        def __itruediv__(self, scalar):
            """Implementation of 'self /= scalar' (true division)."""
            return self.__imul__(1.0 / scalar)

        def __idiv__(self, scalar):
            """Implementation of 'self /= scalar'."""
            return self.__itruediv__(scalar)

        def __add__(self, other):
            """Implementation of 'self + other'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, 1, self, 1, other)
            return tmp

        def __sub__(self, other):
            """Implementation of 'self - other'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, 1, self, -1, other)
            return tmp

        def __mul__(self, scalar):
            """Implementation of 'self * other'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, scalar, self)
            return tmp

        def __rmul__(self, other):
            """Implementation of 'other * self'."""
            return self.__mul__(other)

        def __truediv__(self, scalar):
            """Implementation of 'self / scalar' (true division)."""
            return self.__mul__(1.0 / scalar)

        def __div__(self, scalar):
            """Implementation of 'self / scalar'."""
            return self.__truediv__(scalar)

        def __neg__(self):
            """Implementation of '-self'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, -1.0, self)
            return tmp

        def __pos__(self):
            """Implementation of '+self'."""
            return self.copy()

        def __str__(self):
            """Implementation of str()."""
            return str(self.space) + ".Vector"


class MetricSpace(LinearSpace):

    """Abstract metric space."""

    @abstractmethod
    def _dist(self, x, y):
        """ implementation of distance
        """

    # Default implemented methods
    def dist(self, x, y):
        """
        Calculate the distance between two vectors

        Parameters
        ----------
        x : MetricSpace.Vector
            Vector in this space.

        y : MetricSpace.Vector
            Vector in this space.

        Returns
        -------
        dist : float
               Distance between vectors
        """
        if not self.contains(x):
            raise TypeError('x ({}) is not in space ({})'.format(x, self))

        if not self.contains(y):
            raise TypeError('y ({}) is not in space ({})'.format(y, self))

        return float(self._dist(x, y))

    class Vector(LinearSpace.Vector):
        """ Abstract vector in a metric space
        """

        def __init__(self, space):
            if not isinstance(space, MetricSpace):
                raise TypeError(errfmt('''
                'space' ({}) is not a MetricSpace instance'''.format(space)))
            super().__init__(space)

        def dist(self, other):
            """
            Calculates the distance to another vector.

            Shortcut for self.space.dist(self, other)

            Parameters
            ----------
            None

            Returns
            -------
            dist : float
                   Distance to other.
            """

            return self.space.dist(self, other)

        def equals(self, other):
            """
            Test two vectors for equality.

            Two vectors are equal if their distance is 0

            Parameters
            ----------
            other : MetricSpace.Vector
                    Vector in this space.

            Returns
            -------
            equals : boolean
                     True if the vectors are equal, else false.

            Note
            ----
            Equality is very sensitive to numerical errors, thus any
            operations on a vector should be expected to break equality
            testing.

            Example
            -------

            >>> from RL.space.cartesian import NormedRn
            >>> X = NormedRn(1)
            >>> x = X.element([0.1])
            >>> x == x
            True
            >>> y = X.element([0.1])
            >>> x == y
            True
            >>> z = X.element([0.3])
            >>> x+x+x == z
            False
            """

            if (not isinstance(other, LinearSpace.Vector) or
                    other.space != self.space):
                # Cannot use (if other not in self.space) since this is not
                # reflexive.
                return False
            elif other is self:
                # Optimization for the most common case
                return True
            else:
                return self.dist(other) == 0

        def __eq__(self, other):
            return self.equals(other)

        def __ne__(self, other):
            return not self.equals(other)


class NormedSpace(MetricSpace):
    """ Abstract normed space
    """

    @abstractmethod
    def _norm(self, vector):
        """ implementation of norm
        """

    # Default implemented methods
    def norm(self, vector):
        """ Calculate the norm of a vector
        """
        if not self.contains(vector):
            raise TypeError('x ({}) is not in space ({})'.format(vector, self))

        return float(self._norm(vector))

    # Default implmentation
    def _dist(self, x, y):
        """ The distance in Normed spaces is implicitly defined by the norm
        """
        return self._norm(x-y)

    class Vector(MetricSpace.Vector):
        """ Abstract vector in a normed space
        """

        def __init__(self, space):
            if not isinstance(space, NormedSpace):
                raise TypeError(errfmt('''
                'space' ({}) is not a NormedSpace instance'''.format(space)))
            super().__init__(space)

        def norm(self):
            """
            Calculates the norm of this Vector

            Shortcut for self.space.norm(self)

            Parameters
            ----------
            None

            Returns
            -------
            norm : float
                   Norm of the vector.

            """

            return self.space.norm(self)


class HilbertSpace(NormedSpace):
    """ Abstract (pre)-Hilbert space or inner product space
    """

    @abstractmethod
    def _inner(self, x, y):
        """ Implementation of inner
        """

    # Default implemented methods
    def inner(self, x, y):
        """ Calculates the inner product of the vectors x and y
        """

        # Check spaces
        if not self.contains(x):
            raise TypeError('x ({}) is not in space ({})'.format(x, self))

        if not self.contains(y):
            raise TypeError('y ({}) is not in space ({})'.format(y, self))

        return float(self._inner(x, y))

    # Default implmentation
    def _norm(self, x):
        """ The norm in Hilbert spaces is implicitly defined by the inner
        product
        """

        return sqrt(self._inner(x, x))

    class Vector(NormedSpace.Vector):
        """ Abstract vector in a Hilbert-space
        """

        def __init__(self, space):
            if not isinstance(space, HilbertSpace):
                raise TypeError(errfmt('''
                'space' ({}) is not a HilbertSpace instance'''.format(space)))
            super().__init__(space)

        def inner(self, x):
            """ Calculate the inner product of this and another vector

            Shortcut for self.space.inner(self, x)

            Args:
                x:  Vector in same space as self
            """

            return self.space.inner(self, x)


class Algebra(LinearSpace):
    """ Algebras, or Banach Algebras are linear spaces with multiplication
    defined
    """

    @abstractmethod
    def _multiply(self, x, y):
        """ Implementation of multiply
        """

    def multiply(self, x, y):
        """ Calculates the pointwise product of x and y and assigns it to y
        y = x * y
        """
        # Check spaces
        if not self.contains(x):
            raise TypeError('x ({}) is in wrong space'.format(x))

        if not self.contains(y):
            raise TypeError('y ({}) is in wrong space'.format(y))

        self._multiply(x, y)

    class Vector(LinearSpace.Vector):

        def __init__(self, space):
            if not isinstance(space, Algebra):
                raise TypeError(errfmt('''
                'space' ({}) is not an Algebra instance'''.format(space)))
            super().__init__(space)

        def multiply(self, other):
            """ Shortcut for space.multiply(self, other)
            """
            self.space.multiply(other, self)

        def __imul__(self, other):
            """ Overloads the *= operator to mean pointwise multiplication if
            the other object is a vector
            """
            if isinstance(other, Algebra.Vector):
                self.multiply(other)
                return self
            else:
                return super().__imul__(other)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
