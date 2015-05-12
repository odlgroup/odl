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
    from builtins import object, str, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import object, str, super
from future.utils import with_metaclass
from future import standard_library

# External module imports
from abc import ABCMeta, abstractmethod, abstractproperty
from math import sqrt

# RL imports
from RL.space.set import Set
from RL.utility.utility import errfmt

standard_library.install_aliases()

__all__ = ['LinearSpace', 'NormedSpace', 'HilbertSpace', 'Algebra']


class LinearSpace(with_metaclass(ABCMeta, Set)):
    """ Abstract linear space
    """

    @abstractmethod
    def empty(self):
        """ Create an empty vector (of undefined state)

        An empty vector may be any vector in this space.
        No guarantee of the state of the vector is given.

        Parameters
        ----------
        None

        Returns
        -------
        v : Vector
            An arbitrary vector in this space
        """

    @abstractmethod
    def linCombImpl(self, z, a, x, b, y):
        """ Calculate z = a*x + b*y. This method is intended to be private,
        public callers should resort to linComb which is type checked.
        """

    @abstractproperty
    def field(self):
        """ Get the underlying field
        """

    # Also abstract equals(self,other) from set

    # Default implemented operators
    def zero(self):
        """ A zero vector in this space

        The zero vector is defined as the additive unit of a space.

        Parameters
        ----------
        None

        Returns
        -------
        v : Vector
            The zero vector of this space
        """
        # Default implementation using linComb
        tmp = self.empty()
        self.linCombImpl(tmp, 0, tmp, 0, tmp)
        return tmp

    def contains(self, x):
        """ check vector for membership in space
        """
        return isinstance(x, LinearSpace.Vector) and x.space.equals(self)

    # Overload for `vec in space` syntax
    __contains__ = contains

    # Error checking variant of methods
    def linComb(self, z, a, x, b=None, y=None):
        """ Linear combination of vectors

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

        space.linComb(x, 2, x, 3.14, x)

        is (mathematically) equivalent to

        x = x * (1 + 2 + 3.14)
        """

        if not self.contains(z):
            raise TypeError(errfmt('''
            Lincomb failed, z ({}) is not in space ({})'''.format(z, self)))

        if not self.field.contains(a):
            raise TypeError(errfmt('''
            Lincomb failed, a ({}) is not in field ({})
            '''.format(a, self.field)))

        if not self.contains(x):
            raise TypeError(errfmt('''
            Lincomb failed, x ({}) is not in space ({})'''.format(x, self)))

        if b is None:  # Single argument
            if y is not None:
                raise ValueError(errfmt('''
                Lincomb failed, y ({}) provided but not b'''.format(y)))

            # Call method
            return self.linCombImpl(z, a, x, 0, x)
        else:  # Two arguments
            if not self.field.contains(b):
                raise TypeError(errfmt('''
                Lincomb failed, b ({}) is not in field ({})
                '''.format(b, self.field)))
            if not self.contains(y):
                raise TypeError(errfmt('''
                Lincomb failed, y ({}) is not in space ({})
                '''.format(y, self)))

            # Call method
            return self.linCombImpl(z, a, x, b, y)

    class Vector(with_metaclass(ABCMeta, object)):
        """ Abstract vector, an element in the linear space
        """

        def __init__(self, space):
            """ Default initializer of vectors, must be called by all deriving
            classes to set space
            """
            self._space = space

        @property
        def space(self):
            """ Get the space this vector belongs to
            """
            return self._space

        # Convenience functions
        def assign(self, other):
            """ Assign the values of other to this vector
            """
            self.space.linComb(self, 1, other)

        def copy(self):
            """ Creates an identical (deep) copy of this vector
            """
            result = self.space.empty()
            result.assign(self)
            return result

        def linComb(self, a, x, b=None, y=None):
            """ Wrapper for space.linComb(self, a, x, b, y)
            """
            self.space.linComb(self, a, x, b, y)

        def setZero(self):
            """ Sets this vector to the zero vector
            """
            self.space.linComb(self, 0, self, 0, self)

        # Convenience operators
        def __iadd__(self, other):
            """Vector addition (self += other)
            """
            self.space.linComb(self, 1, self, 1, other)
            return self

        def __isub__(self, other):
            """Vector subtraction (self -= other)
            """
            self.space.linComb(self, 1, self, -1, other)
            return self

        def __imul__(self, scalar):
            """Vector multiplication by scalar (self *= scalar)
            """
            self.space.linComb(self, scalar, self)
            return self

        def __itruediv__(self, scalar):
            """Vector division by scalar (self *= (1/scalar))
            """
            return self.__imul__(1./scalar)

        __idiv__ = __itruediv__

        def __add__(self, other):
            """Vector addition (ret = self + other)
            """
            tmp = self.space.empty()
            self.space.linComb(tmp, 1, self, 1, other)
            return tmp

        def __sub__(self, other):
            """Vector subtraction (ret = self - other)
            """
            tmp = self.space.empty()
            self.space.linComb(tmp, 1, self, -1, other)
            return tmp

        def __mul__(self, scalar):
            """Scalar multiplication (ret = self * scalar)
            """
            tmp = self.space.empty()
            self.space.linComb(tmp, scalar, self)
            return tmp

        __rmul__ = __mul__

        def __truediv__(self, scalar):
            """ Scalar division (ret = self / scalar)
            """
            return self.__mul__(1.0 / scalar)

        __div__ = __truediv__

        def __neg__(self):
            """ Unary negation, used in assignments (ret = -self)
            """
            tmp = self.space.empty()
            self.space.linComb(tmp, -1.0, self)
            return tmp

        def __pos__(self):
            """ Unary plus (the identity operator), creates a copy of this
            object
            """
            return self.copy()

        def __str__(self):
            """ A default representation of the vector
            """
            return str(self.space) + "::Vector"

class MetricSpace(with_metaclass(ABCMeta, LinearSpace)):
    """ Abstract metric space
    """

    @abstractmethod
    def distImpl(self, vector):
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

        return float(self.normImpl(vector))

    class Vector(with_metaclass(ABCMeta, LinearSpace.Vector)):
        """ Abstract vector in a metric space
        """

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
            operations on a vector should be expected to break equality testing.
            For example

            >>> X = RN(1)
            >>> x = X.vector([0.1])
            >>> y = X.vector([0.3])
            >>> x+x+x == y
            False
            """
            if not isinstance(other, LinearSpace.Vector) or other.space != self.space: 
                # Cannot use (if other not in self.space) since this is not reflexive.
                return False
            elif other is self:
                 #Optimization for the most common case
                return True
            else:
                return self.dist(other) == 0

        def __eq__(self, other):
            return self.equals(other)

        def __ne__(self, other):
            return not self.equals(other)

class NormedSpace(with_metaclass(ABCMeta, MetricSpace)):
    """ Abstract normed space
    """

    @abstractmethod
    def normImpl(self, vector):
        """ implementation of norm
        """

    # Default implemented methods
    def norm(self, vector):
        """ Calculate the norm of a vector
        """
        if not self.contains(vector):
            raise TypeError('x ({}) is not in space ({})'.format(vector, self))

        return float(self.normImpl(vector))

    #Default implmentation
    def distImpl(self, x, y):
        """ The distance in Normed spaces is implicitly defined by the norm
        """
        return self.normImpl(x-y)

    class Vector(with_metaclass(ABCMeta, MetricSpace.Vector)):
        """ Abstract vector in a normed space
        """

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


class HilbertSpace(with_metaclass(ABCMeta, NormedSpace)):
    """ Abstract (pre)-Hilbert space or inner product space
    """

    @abstractmethod
    def innerImpl(self, x, y):
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

        return self.innerImpl(x, y)

    def normImpl(self, x):
        """ The norm in Hilbert spaces is implicitly defined by the inner
        product
        """
        return sqrt(self.innerImpl(x, x))

    class Vector(with_metaclass(ABCMeta, NormedSpace.Vector)):
        """ Abstract vector in a Hilbert-space
        """

        def inner(self, x):
            """ Shortcut for self.space.inner(self, x)

            Args:
                x:  Vector in same space as self
            """
            return self.space.inner(self, x)


class Algebra(with_metaclass(ABCMeta, LinearSpace)):
    """ Algebras, or Banach Algebras are linear spaces with multiplication
    defined
    """

    @abstractmethod
    def multiplyImpl(self, x, y):
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

        self.multiplyImpl(x, y)

    class Vector(with_metaclass(ABCMeta, LinearSpace.Vector)):

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
