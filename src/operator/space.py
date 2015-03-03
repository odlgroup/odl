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
import numpy as np

import RL.operator.operatorAlternative as OP
from RL.utility.utility import allEqual
from RL.operator.measure import *


class Field:
    Real, Complex = range(2)

class Space(object):
    """Abstract space
    """

    __metaclass__ = ABCMeta #Set as abstract

    class Vector(object):
        """Abstract vector
        """

        __metaclass__ = ABCMeta #Set as abstract

        def __init__(self,parent,*args, **kwargs):
            self.parent = parent

        @abstractmethod
        def clone(self):
            """Create a clone of this vector
            """

        def __add__(self, other):
            """Vector addition
            """
            tmp = self.clone()
            self.parent.linComb(1,other,1,tmp)
            return tmp

        def __mul__(self, other):
            """Scalar multiplication
            """
            tmp = self.clone()
            self.parent.linComb(other,self,0,tmp)
            return tmp

    @abstractmethod
    def zero(self):
        """The zero element of the space
        """
    
    @abstractmethod
    def inner(self,A,B):
        """Inner product
        """

    @abstractmethod
    def linComb(self,a,x,b,y):
        """Calculate y=ax+by
        """

    @abstractmethod
    def equals(self, x):
        """check spaces for equality
        """ 

    @abstractproperty
    def field(self):
        """ Get the underlying field
        """
        pass

    @abstractproperty
    def dimension(self):
        """ Get the dimension of the space
        """
        pass

    #Implicitly defined operators

    def normSquared(self,x):
        return self.inner(x,x)

    def norm(self,x):
        return sqrt(self.normSquared(x,x))

    def __eq__(self,other):
        return self.equals(other)

    def __ne__(self,other):
        return not self.equals(other)
    
class ProductSpace(Space):
    """Product space (X1 x X2 x ... x Xn)
    """

    def __init__(self,*spaces):
        if (len(spaces)==0):
            raise TypeError("Empty product not allowed")
        if (not allEqual(spaces, lambda x,y: x.field == y.field)):
            raise TypeError("All spaces must have the same field")

        self.spaces = spaces
        self._dimension = sum(space.dimension for space in self.spaces)
        self._field = spaces[0].field  #X_n has same field

    def zero(self):
        return self.makeVector(*[A.zero() for A in self.spaces])
    
    def inner(self,v1,v2):
        return sum(space.inner(v1p,v2p) for [space,v1p,v2p] in zip(self.spaces,v1.parts,v2.parts))

    def linComb(self,a,x,b,y):
        for [space,xp,yp] in zip(self.spaces,x.parts,y.parts):
            space.linComb(a,xp,b,yp)

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        return self._dimension

    def equals(self, other):
        return all(x.equals(y) for [x,y] in zip(self.spaces,other.spaces))
    
    def makeVector(self,*args):
        return ProductSpace.Vector(self,*args)

    def __getitem__(self,index):
        return self.spaces[index]

    class Vector(Space.Vector):
        def __init__(self,parent,*args):
            self.parent = parent
            if (not isinstance(args[0],Space.Vector)): #Delegate constructors
                self.parts = [space.makeVector(arg) for [arg,space] in zip(args,parent.spaces)]
            else: #Construct from existing tuple
                if any(part.parent != space for [part,space] in zip(args,parent.spaces)):
                    raise TypeError("The spaces of all parts must correspond to this space's parts")

                self.parts = args

        def clone(self):
            return self.parent.makeVector(*[part.clone() for part in self.parts])

        def __getitem__(self,index): #TODO should we have this?
            return self.parts[index]

        def __str__(self):
            return "[" + ",".join(str(part) for part in self.parts) + "]"


class Reals(Space):
    """The real numbers
    """

    def inner(self,x,y):
        return x.__val__ * y.__val__

    def linComb(self,a,x,b,y):
        y.__val__ = a * x.__val__ + b * y.__val__

    def zero(self):
        return self.makeVector(0.0)

    @property
    def field(self):
        return Field.Real

    @property
    def dimension(self):
        return 1
    
    def equals(self, other):
        return isinstance(other,Reals)

    def makeVector(self,value):
        return Reals.Vector(self,value)

    class Vector(Space.Vector):
        """Real vectors are floats
        """

        __val__ = None
        def __init__(self, parent, v):
            Space.Vector.__init__(self,parent)
            self.__val__ = v

        def clone(self):
            return self.parent.makeVector(self.__val__)

        #Need to duplicate methods since vectors are mutable but python floats are not
        #Source: https://gist.github.com/jheiv/6656349

        # Comparison Methods
        def __eq__(self, x):        return self.__val__.__eq__(x.__val__)
        def __ne__(self, x):        return self.__val__.__ne__(x.__val__)
        def __lt__(self, x):        return self.__val__.__lt__(x.__val__)
        def __gt__(self, x):        return self.__val__.__gt__(x.__val__)
        def __le__(self, x):        return self.__val__.__le__(x.__val__)
        def __ge__(self, x):        return self.__val__.__ge__(x.__val__)
        def __cmp__(self, x):       return 0 if self.__val__ == x.__val__ else 1 if self.__val__ > 0 else -1
        # Unary Ops
        def __pos__(self):          return self.__class__(self.parent,self.__val__.__pos__())
        def __neg__(self):          return self.__class__(self.parent,self.__val__.__neg__())
        def __abs__(self):          return self.__class__(self.parent,self.__val__.__abs__())
        # Arithmetic Binary Ops
        def __add__(self, x):       return self.__class__(self.parent,self.__val__.__add__(x.__val__))
        def __sub__(self, x):       return self.__class__(self.parent,self.__val__.__sub__(x.__val__))
        def __mul__(self, x):       return self.__class__(self.parent,self.__val__.__mul__(x.__val__))
        def __div__(self, x):       return self.__class__(self.parent,self.__val__.__div__(x.__val__))
        def __pow__(self, x):       return self.__class__(self.parent,self.__val__.__pow__(x.__val__))
        # Compound Assignment
        def __iadd__(self, x):      self.__val__.__iadd__(x.__val__); return self
        def __isub__(self, x):      self.__val__.__isub__(x.__val__); return self
        def __imul__(self, x):      self.__val__.__imul__(x.__val__); return self
        def __idiv__(self, x):      self.__val__.__idiv__(x.__val__); return self
        def __ipow__(self, x):      self.__val__.__ipow__(x.__val__); return self
        # Casts
        def __nonzero__(self):      return self.__val__.__nonzero__()
        def __float__(self):        return self.__val__.__float__()              # XXX
        # Conversions
        def __str__(self):          return self.__val__.__str__()               # XXX
        # Representation
        def __repr__(self):         return "%s(%d)" % (self.__class__.__name__, self.__val__)

        # Define set, a function that you can use to set the value of the instance
        def set(self, x):
            self.__val__ = x.__val__

        # Pass anything else along to self.__val__
        def __getattr__(self, attr):
            return getattr(self.__val__, attr)

    class MultiplyOp(OP.SelfAdjointOperator):    
        """Multiply with scalar
        """

        def __init__(self,a):
            self.a = a

        def apply(self,rhs):
            return self.a * rhs

    class AddOp(OP.SelfAdjointOperator):
        """Add scalar
        """

        def __init__(self,a):
            self.a = a

        def apply(self,rhs):
            return self.a + rhs


class RN(Space):
    """The real space R^n
    """

    def __init__(self,n):
        self.n = n

    def inner(self,x,y):
        return np.vdot(x,y)
    
    def linComb(self,a,x,b,y):
        y[:]=a*x+b*y

    def zero(self):
        return np.zeros(n)

    @property
    def field(self):
        return Field.Real

    @property
    def dimension(self):
        return self.n

    def equals(self, other):
        return isinstance(other,RN) and self.n == other.n

    def makeVector(self, *args, **kwargs):
        return RN.Vector(self,*args, **kwargs)

    class Vector(np.ndarray,Space.Vector):
        def __new__(cls, parent, *args, **kwargs):
            data = np.array(*args,**kwargs)
            return data.view(RN.Vector)

        def clone(self):
            return self.parent.makeVector(self, copy = True)

    class MultiplyOp(OP.Operator):
        """Multiply with matrix
        """

        def __init__(self,A):
            self.A = A

        def apply(self,rhs):
            return np.dot(self.A,rhs)

        def applyAdjoint(self,rhs):
            return np.dot(self.A.T,rhs)

#Example of a space:
class L2(Space):
    """The space of square integrable functions on some domain
    """

    def __init__(self,domain):
        self.domain = domain

    def inner(self,v1,v2):
        return self.domain.integrate(OP.PointwiseProduct(v1,v2))

    def linComb(self,a,b,A,B):
        return a*A+b*B #Use operator overloading

    def field(self):
        return Field.Real

    def dimension(self):
        return 1.0/0.0
    
    def equals(self, other):
        raise NotImplementedError("Todo")

    def zero(self):
        class ZeroFunction(L2.Vector):
            def apply(self,rhs):
                return 0.0
        return ZeroFunction(self)

    class Vector(OP.Operator,Space.Vector):
        """ L2 Vectors are operators from the domain onto R(C)
        """

        def clone(self):
            raise NotImplementedError("Todo")