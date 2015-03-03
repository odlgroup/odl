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
from abc import ABCMeta, abstractmethod
from itertools import izip

from math import sin,cos,sqrt
import numpy as np

import RL.operator.operatorAlternative as OP
from RL.utility.utility import allEqual


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

        def __add__(self, other):
            """Vector addition
            """
            from copy import copy
            tmp = copy(self)
            self.parent.linearComb(1,1,self,tmp)
            return tmp

        def __mul__(self, other):
            """Scalar multiplication
            """
            return self.parent.linearComb(other,0,self,self.parent.zero())

    @abstractmethod
    def zero(self):
        """The zero element of the space
        """
        pass
    
    @abstractmethod
    def inner(self,A,B):
        """Inner product
        """
        pass

    @abstractmethod
    def linearComb(self,a,x,b,y):
        """Calculate y=ax+by
        """
        pass

    @abstractmethod
    def field(self):
        """ Get the underlying field
        """
        pass

    @abstractmethod
    def dimension(self):
        """ Get the dimension of the space
        """
        pass

    def normSquared(self,x):
        return self.inner(x,x)

    def norm(self,x):
        return sqrt(self.normSquared(x,x))
    
class ProductSpace(Space):
    """Product space (X1 x X2 x ... x Xn)
    """

    def __init__(self,*spaces):
        if (not allEqual(spaces, lambda x,y: x.field() == y.field())):
            raise AttributeError("All spaces must have the same field")

        self.spaces = spaces

    def zero(self):
        return self.makeVector(*[A.zero() for A in self.spaces])
    
    def inner(self,v1,v2):
        return sum(space.inner(v1p,v2p) for [space,v1p,v2p] in zip(self.spaces,v1.parts,v2.parts))

    def linearComb(self,a,b,v1,v2):
        for [space,v1p,v2p] in zip(self.spaces,v1.parts,v2.parts):
            space.linearComb(a,b,v1p,v2p)

    def field(self):
        return self.spaces[0].field() #X_n has same field

    def dimension(self):
        return sum(space.dimension() for space in self.spaces)
    
    def makeVector(self,*args):
        return ProductSpace.Vector(self,*args)

    class Vector(Space.Vector):
        def __init__(self,parent,*parts):
            self.parent = parent
            self.parts = parts

        def __getitem__(self,index): #TODO should we have this?
            ind = 0
            for part in self.parts:
                if ind+part.parent.dimension()>index:
                    return part[index-ind]
                else:
                    ind += part.parent.dimension()
            #Todo complexity?
            #Todo out of range

        def __str__(self):
            return "[" + ",".join(str(part) for part in self.parts) + "]"   


class Reals(Space):
    """The real numbers
    """

    def inner(self,x,y):
        return x.__val__ * y.__val__

    def linearComb(self,a,x,b,y):
        y *= b
        y += a*x

    def zero(self):
        return self.makeVector(0.0)

    def field(self):
        return Field.Real

    def dimension(self):
        return 1

    def makeVector(self,value):
        return Reals.Vector(self,value)

    class Vector(Space.Vector):
        """Real vectors are floats
        """

        __val__ = None
        def __init__(self, parent, v):
            Space.Vector.__init__(self,parent)
            self.__val__ = v

        #Need to duplicate methods since vectors are mutable but floats are not
        #Source: https://gist.github.com/jheiv/6656349

        # Comparison Methods
        def __eq__(self, x):        return self.__val__ == x
        def __ne__(self, x):        return self.__val__ != x
        def __lt__(self, x):        return self.__val__ <  x
        def __gt__(self, x):        return self.__val__ >  x
        def __le__(self, x):        return self.__val__ <= x
        def __ge__(self, x):        return self.__val__ >= x
        def __cmp__(self, x):       return 0 if self.__val__ == x else 1 if self.__val__ > 0 else -1
        # Unary Ops
        def __pos__(self):          return self.__class__(self.parent,+self.__val__)
        def __neg__(self):          return self.__class__(self.parent,-self.__val__)
        def __abs__(self):          return self.__class__(self.parent,abs(self.__val__))
        # Bitwise Unary Ops
        def __invert__(self):       return self.__class__(self.parent,~self.__val__)
        # Arithmetic Binary Ops
        def __add__(self, x):       return self.__class__(self.parent,self.__val__ + x)
        def __sub__(self, x):       return self.__class__(self.parent,self.__val__ - x)
        def __mul__(self, x):       return self.__class__(self.parent,self.__val__ * x)
        def __div__(self, x):       return self.__class__(self.parent,self.__val__ / x)
        def __mod__(self, x):       return self.__class__(self.parent,self.__val__ % x)
        def __pow__(self, x):       return self.__class__(self.parent,self.__val__ ** x)
        def __floordiv__(self, x):  return self.__class__(self.parent,self.__val__ // x)
        def __divmod__(self, x):    return self.__class__(self.parent,divmod(self.__val__, x))
        def __truediv__(self, x):   return self.__class__(self.parent,self.__val__.__truediv__(x))
        # Reflected Arithmetic Binary Ops
        def __radd__(self, x):      return self.__class__(self.parent,x + self.__val__)
        def __rsub__(self, x):      return self.__class__(self.parent,x - self.__val__)
        def __rmul__(self, x):      return self.__class__(self.parent,x * self.__val__)
        def __rdiv__(self, x):      return self.__class__(self.parent,x / self.__val__)
        def __rmod__(self, x):      return self.__class__(self.parent,x % self.__val__)
        def __rpow__(self, x):      return self.__class__(self.parent,x ** self.__val__)
        def __rfloordiv__(self, x): return self.__class__(self.parent,x // self.__val__)
        def __rdivmod__(self, x):   return self.__class__(self.parent,divmod(x, self.__val__))
        def __rtruediv__(self, x):  return self.__class__(self.parent,x.__truediv__(self.__val__))
        # Bitwise Binary Ops
        def __and__(self, x):       return self.__class__(self.parent,self.__val__ & x)
        def __or__(self, x):        return self.__class__(self.parent,self.__val__ | x)
        def __xor__(self, x):       return self.__class__(self.parent,self.__val__ ^ x)
        def __lshift__(self, x):    return self.__class__(self.parent,self.__val__ << x)
        def __rshift__(self, x):    return self.__class__(self.parent,self.__val__ >> x)
        # Reflected Bitwise Binary Ops
        def __rand__(self, x):      return self.__class__(self.parent,x & self.__val__)
        def __ror__(self, x):       return self.__class__(self.parent,x | self.__val__)
        def __rxor__(self, x):      return self.__class__(self.parent,x ^ self.__val__)
        def __rlshift__(self, x):   return self.__class__(self.parent,x << self.__val__)
        def __rrshift__(self, x):   return self.__class__(self.parent,x >> self.__val__)
        # Compound Assignment
        def __iadd__(self, x):      self.__val__ += x; return self
        def __isub__(self, x):      self.__val__ -= x; return self
        def __imul__(self, x):      self.__val__ *= x; return self
        def __idiv__(self, x):      self.__val__ /= x; return self
        def __imod__(self, x):      self.__val__ %= x; return self
        def __ipow__(self, x):      self.__val__ **= x; return self
        # Casts
        def __nonzero__(self):      return self.__val__ != 0
        def __int__(self):          return self.__val__.__int__()               # XXX
        def __float__(self):        return self.__val__.__float__()             # XXX
        def __long__(self):         return self.__val__.__long__()              # XXX
        # Conversions
        def __oct__(self):          return self.__val__.__oct__()               # XXX
        def __hex__(self):          return self.__val__.__hex__()               # XXX
        def __str__(self):          return self.__val__.__str__()               # XXX
        # Random Ops
        def __index__(self):        return self.__val__.__index__()             # XXX
        def __trunc__(self):        return self.__val__.__trunc__()             # XXX
        def __coerce__(self, x):    return self.__val__.__coerce__(x)
        # Represenation
        def __repr__(self):         return "%s(%d)" % (self.__class__.__name__, self.__val__)

        # Define set, a function that you can use to set the value of the instance
        def set(self, x):
            if   isinstance(x, float): self.__val__ = x
            elif isinstance(x, self.__class__): self.__val__ = x.__val__
            else: raise TypeError("expected a numeric type")
        # Pass anything else along to self.__val__
        def __getattr__(self, attr):
            print("getattr: " + attr)
            return getattr(self.__val__, attr)

        def __getitem__(self,index): #TODO should we have this?
            if (index > 0):
                raise IndexError("Out of range")
            return self

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
    
    def linearComb(self,a,x,b,y):
        y[:]=a*x+b*y

    def zero(self):
        return np.zeros(n)

    def field(self):
        return Field.Real

    def dimension(self):
        return self.n

    def makeVector(self, *args, **kwargs):
        return RN.Vector(self,*args, **kwargs)

    class Vector(np.ndarray,Space.Vector):
        def __new__(cls, parent, *args, **kwargs):
            data = np.array(*args,**kwargs)
            return data.view(RN.Vector)

    class MultiplyOp(OP.Operator):
        """Multiply with matrix
        """

        def __init__(self,A):
            self.A = A

        def apply(self,rhs):
            return np.dot(self.A,rhs)

        def applyAdjoint(self,rhs):
            return np.dot(self.A.T,rhs)


class RNM(Space):
    """The space of real nxm matrices
    """

    def __init__(self,n,m):
        self.n = n
        self.m = m

    def inner(self,A,B):
        return np.vdot(self,A,B)
    
    def linearComb(self,a,b,A,B):
        return a*A+b*B

    def zero(self):
        return np.zeros(n,m)

class measureSpace(object):
    """A space where integration is defined
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def integrate(self,f):
        """Calculate the integral of f
        """
        pass

class Interval(measureSpace):
    def __init__(self,begin,end):
        self.begin = begin
        self.end = end

    def integrate(self,f):
        raise NotImplementedError("Cannot integrate without discretization") #TODO add abstract measure space?

class LinspaceDiscretization(measureSpace):
    def __init__(self,interval,n):
        self.interval = interval
        self.n = n

    def integrate(self,f):
        s = 0.0
        for x in np.linspace(self.interval.begin,self.interval.end,self.n):
            s += f(x)

        return s * (self.interval.end - self.interval.begin) / self.n

#Example of a space:
class L2(Space):
    """The space of square integrable functions on some domain
    """

    def __init__(self,domain):
        self.domain = domain

    def inner(self,v1,v2):
        return self.domain.integrate(OP.PointwiseProduct(v1,v2))

    def linearComb(self,a,b,A,B):
        return a*A+b*B

    def field(self):
        return Field.Real

    def dimension(self):
        return 1.0/0.0

    def zero(self):
        class ZeroFunction(L2.Vector):
            def apply(self,rhs):
                return 0.0
        return ZeroFunction(self)

    def sin(self):
        class SinFunction(L2.Vector):
            def apply(self,rhs):
                return sin(rhs)
        return SinFunction(self)

    def sin(self):
        class CosFunction(L2.Vector):
            def apply(self,rhs):
                return cos(rhs)
        return CosFunction(self)

    class Vector(OP.Operator,Space.Vector):
        """ L2 Vectors are operators from the domain onto R(C)
        """
        pass