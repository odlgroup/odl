# -*- coding: utf-8 -*-
"""
operator.py -- functional analytic operators

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future.builtins import object
from future import standard_library
standard_library.install_aliases()
from abc import ABCMeta, abstractmethod, abstractproperty
from math import sqrt
from RL.space.set import AbstractSet

class LinearSpace(AbstractSet):
    """ Abstract linear space
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def empty(self):
        """ Create an empty vector (of undefined state)
        """

    @abstractmethod
    def linCombImpl(self, z, a, x, b, y):
        """ Calculate z = ax + by. This method is intended to be private, public callers should resort to linComb which is type checked.
        """

    @abstractproperty
    def field(self):
        """ Get the underlying field
        """

    @abstractproperty
    def dimension(self):
        """ Get the dimension of the space
        """

    # Also equals(self,other) from set

    # Default implemented operators
    def zero(self):
        """ The zero vector of this space
        """
        #Default implementation using linComb
        tmp = self.empty()
        self.linCombImpl(tmp, 0, tmp, 0, tmp)
        return tmp

    def isMember(self, x):
        """ check vector for membership in space
        """
        return isinstance(x, LinearSpace.Vector) and x.space.equals(self)

    # Error checking variant of methods
    def linComb(self, z, a, x, b = None, y = None):
        """ Calculates 
        z = ax
        or
        z = ax + by 
        
        with error checking of types
        """
        
        if not self.isMember(z): 
            raise TypeError('z ({}) is not in space ({})'.format(z, self))

        if not self.field.isMember(a): 
            raise TypeError('a ({}) is not in field ({})'.format(a, self.field))
        if not self.isMember(x): 
            raise TypeError('x ({}) is not in space ({})'.format(x, self))

          
        if b is None:
            if y is not None:
                raise ValueError('y ({}) provided but not b'.format(y, self))

            return self.linCombImpl(z, a, x, 0, x)
        else:
            if not self.field.isMember(b): 
                raise TypeError('b ({}) is not in field ({})'.format(b, self.field))
            if not self.isMember(y): 
                raise TypeError('y ({}) is not in space ({})'.format(y, self))

            #Call method
            return self.linCombImpl(z, a, x, b, y)

    class Vector(object):
        """ Abstract vector
        """

        __metaclass__ = ABCMeta #Set as abstract

        def __init__(self, space):
            """ Default initializer of vectors, must be called by all deriving classes to set space
            """
            self._space = space

        @property
        def space(self):
            """ Get the space this vector belongs to
            """
            return self._space

        #Convenience functions
        def assign(self, other):
            """ Assign the values of other to this vector
            """
            self.space.linComb(self, 1, other)

        def copy(self):
            """ Creates an identical clone of this vector
            """
            result = self.space.empty()
            result.assign(self)
            return result
        
        def linComb(self, a, x, b=None, y=None):
            """ Wrapper for space.linComb(self, a, x, b, y)
            """
            self.space.linComb(self, a, x, b, y)

        #Convenience operators
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
            """ Unary negation, used in assignments:
            a = -b
            """
            tmp = self.space.empty()
            self.space.linComb(tmp, -1.0, self)
            return tmp

        def __pos__(self):
            """ Unary plus (the identity operator), creates a clone of this object
            """
            return self.copy()

        def __len__(self):
            return self.space.dimension

        def __str__(self):
            return str(self.space) + "::Vector"


class NormedSpace(LinearSpace):
    """ Abstract normed space
    """
    
    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def normSqImpl(self, vector):
        """ implementation of normSq
        """

    #Default implemented methods
    def normSq(self, vector):
        """ Calculate the squared norm of the vector
        """
        if not self.isMember(vector):
            raise TypeError('x ({}) is not in space ({})'.format(vector, self))

        return self.normSqImpl(vector)

    def norm(self, vector):
        """ The norm of the vector, default implementation uses the normSquared implementation
        """
        return sqrt(self.normSq(vector))

    class Vector(LinearSpace.Vector):

        __metaclass__ = ABCMeta #Set as abstract

        #Member variants of the space method
        def normSq(self):           
            return self.space.normSq(self)

        def norm(self):             
            return self.space.norm(self)

class HilbertSpace(NormedSpace):
    """ Abstract (pre)-Hilbert space or inner product space
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def innerImpl(self, x, y):
        """ Implementation of inner
        """
        
    #Default implemented methods
    def inner(self, x, y):
        """ Calculates the inner product of the vectors x and y
        """

        #Check spaces
        if not self.isMember(x): 
            raise TypeError('x ({}) is not in space ({})'.format(x, self))

        if not self.isMember(y):
            raise TypeError('y ({}) is not in space ({})'.format(y, self))

        return self.innerImpl(x, y)
    
    def normSqImpl(self, x):
        """ The norm in Hilbert spaces is implicitly defined by the inner product
        """
        return self.innerImpl(x, x)

    class Vector(NormedSpace.Vector):

        __metaclass__ = ABCMeta #Set as abstract

        def inner(self, x):         
            return self.space.inner(self, x)


class Algebra(LinearSpace):
    
    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def multiplyImpl(self, x, y):
        """ Implementation of multiply
        """

    def multiply(self, x, y):
        """ Calculates the pointwise product of x and y and assigns it to y
        y = x*y
        """
        #Check spaces
        if not self.isMember(x): 
            raise TypeError('x ({}) is in wrong space'.format(x))

        if not self.isMember(y): 
            raise TypeError('y ({}) is in wrong space'.format(y))

        self.multiplyImpl(x, y)

    class Vector(LinearSpace.Vector):
        
        __metaclass__ = ABCMeta #Set as abstract

        
        def multiply(self, other):
            self.space.multiply(other,self)

        def __imul__(self, other):
            """ Overloads the *= operator to mean pointwise multiplication if the other object is a vector
            """
            if isinstance(other, Algebra.Vector):
                self.multiply(other)
                return self
            else:
                return LinearSpace.Vector.__imul__(self, other)
