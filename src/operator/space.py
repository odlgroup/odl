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


class Field:
    Real, Complex = range(2)

class LinearSpace(object):
    """ Abstract linear space
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def zero(self):
        """ The zero element of the space
        """

    @abstractmethod
    def linComb(self,a,x,b,y):
        """ Calculate y=ax+by
        """

    @abstractmethod
    def __eq__(self, x):
        """ check spaces for equality
        """

    @abstractproperty
    def field(self):
        """ Get the underlying field
        """

    @abstractproperty
    def dimension(self):
        """ Get the dimension of the space
        """

    class Vector(object):
        """ Abstract vector
        """

        __metaclass__ = ABCMeta #Set as abstract

        def __init__(self,space):
            """ Default initializer of vectors, must be called by all deriving classes to set space
            """
            self.__space__ = space

        @abstractmethod
        def assign(self, other):
            """ Assign the values of other to this vector
            """

        @property
        def space(self):
            """ Get the space this vector belongs to
            """
            return self.__space__

        #Convenience operators
        def __iadd__(self,other):
            """Vector addition (self += other)
            """
            self.linComb(1.,other)
            return self

        def __isub__(self,other):
            """Vector subtraction (self -= other)
            """
            self.linComb(-1.,other)
            return self

        def __imul__(self,scalar):
            """Vector multiplication by scalar (self *= scalar)
            """
            self.space.linComb(0.,self,scalar,self)
            return self

        def __itruediv__(self,scalar):
            """Vector division by scalar (self *= (1/scalar))
            """
            return self.__imul__(1./scalar)

        __idiv__ = __itruediv__

        def __add__(self, other):
            """Vector addition (ret = self - other)
            """
            tmp = self.clone()
            tmp += other
            return tmp

        def __sub__(self, other):
            """Vector subtraction (ret = self - other)
            """
            tmp = self.clone()
            tmp -= other
            return tmp

        def __mul__(self, scalar):
            """Scalar multiplication (ret = self * scalar)
            """
            tmp = self.clone()
            tmp *= scalar
            return tmp
        
        __rmul__ = __mul__

        def __truediv__(self, scalar):
            """ Scalar division (ret = self * (1/scalar))
            """
            tmp = self.clone()
            tmp /= scalar
            return tmp

        __div__ = __truediv__

        def __neg__(self):
            """ Unary negation, used in assignments:
            a = -b
            """
            tmp = self.clone()
            tmp *= -1
            return tmp

        def __pos__(self):
            """ Unary plus (the identity operator), creates a clone of this object
            """
            return self.clone()

        def clone(self):
            """ Creates an identical clone of this vector
            """
            result = self.space.empty()
            result.assign(self)
            return result

        def linComb(self,a,x):     self.space.linComb(a,x,1.,self)

    #Default implemented operators
    def empty(self):
        """An empty vector (of undefined state) defaults to zero vector if no implementation is provided
        """
        return self.zero()

    #Implicitly defined operators
    def __ne__(self,other):         return not self == other

class NormedSpace(LinearSpace):
    """ Abstract normed space
    """
    
    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def normSquared(self,x):
        """ Calculate the squared norm of the vector x
        """

    class Vector(LinearSpace.Vector):
        #Member variants of the space method
        def normSquared(self):     return self.space.normSquared(self)
        def norm(self):            return self.space.norm(self)

    def norm(self,x):               
        """ The norm of the vector x, default implementation uses the normSquared implementation
        """
        return sqrt(self.normSquared(x))

class HilbertSpace(LinearSpace):
    """ Abstract (pre)-Hilbert space or inner product space
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def inner(self,x,y):
        """ Inner product of the vectors x,y
        """

    class Vector(LinearSpace.Vector):
        def inner(self,x):         return self.space.inner(self,x)
    
    def normSquared(self,x):
        """ The norm in Hilbert spaces is implicitly defined by the inner product
        """
        return self.inner(x,x)