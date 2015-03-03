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

    def normSquared(self,x):
        return self.inner(x,x)

    def norm(self,x):
        return sqrt(self.normSquared(x,x))
    
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

    def linearComb(self,a,b,v1,v2):
        for [space,v1p,v2p] in zip(self.spaces,v1.parts,v2.parts):
            space.linearComb(a,b,v1p,v2p)

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        return self._dimension
    
    def makeVector(self,*args):
        return ProductSpace.Vector(self,*args)

    class Vector(Space.Vector):
        def __init__(self,parent,*parts):
            self.parent = parent
            self.parts = parts

        def __getitem__(self,index): #TODO should we have this?
            return self.parts[index]

        def __str__(self):
            return "[" + ",".join(str(part) for part in self.parts) + "]"   


class Reals(Space):
    """The real numbers
    """

    def inner(self,x,y):
        return x.__val__ * y.__val__

    def linearComb(self,a,x,b,y):
        y.__val__ = a * x.__val__ + b * y.__val__

    def zero(self):
        return self.makeVector(0.0)

    @property
    def field(self):
        return Field.Real

    @property
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
        # Reflected Arithmetic Binary Ops
        def __radd__(self, x):      return self.__class__(self.parent,self.__val__.__radd__(x.__val__))
        def __rsub__(self, x):      return self.__class__(self.parent,self.__val__.__rsub__(x.__val__))
        def __rmul__(self, x):      return self.__class__(self.parent,self.__val__.__rmul__(x.__val__))
        def __rdiv__(self, x):      return self.__class__(self.parent,self.__val__.__rdiv__(x.__val__))
        def __rpow__(self, x):      return self.__class__(self.parent,self.__val__.__rpow__(x.__val__))
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
        def __oct__(self):          return self.__val__.__oct__()               # XXX
        def __hex__(self):          return self.__val__.__hex__()               # XXX
        def __str__(self):          return self.__val__.__str__()               # XXX
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

    @property
    def field(self):
        return Field.Real

    @property
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

class set(object):
    """ An arbitrary set
    """

    __metaclass__ = ABCMeta #Set as abstract

class measure(object):
    """ A measure on some set
    """

    __metaclass__ = ABCMeta #Set as abstract

    def __call__(self,set):
        return self.measure(set)

    @abstractmethod
    def measure(self,set):
        """Calculate the measure of set
        """

class measurableSets(object):
    """ Some measurable sets, subsets of some set
    """

    __metaclass__ = ABCMeta #Set as abstract


class measureSpace(object):
    """A space where integration is defined
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def integrate(self,f):
        """Calculate the integral of f
        """

class Interval(set):
    def __init__(self,begin,end):
        self.begin = begin
        self.end = end

    def midpoint(self):
        return (self.end+self.begin)/2.0

class borelMeasure(measure):
    def measure(self, interval):
        return interval.end-interval.begin

class discretization(measurableSets):
    @abstractmethod
    def __iter__(self):
        """Discrete spaces can be iterated over
        """

class LinspaceDiscretization(discretization):
    def __init__(self,interval,n):
        self.interval = interval
        self.n = n

    def __iter__(self):
        class intervalIter():
            def __init__(self,begin,step,n):
                self.cur = begin
                self.step = step
                self.n = n

            def next(self):
                if self.n>0:
                    i = Interval(self.cur,self.cur+self.step)
                    self.cur += self.step
                    self.n -= 1
                    return i
                else:
                    raise StopIteration()

        begin = self.interval.begin
        step = (self.interval.end - self.interval.begin)/self.n
        return intervalIter(begin,step,self.n)

class discreteMeaureSpace(measureSpace):
    def __init__(self,discretization,measure):
        self.discretization = discretization
        self.measure = measure

    def integrate(self,f):
        return sum(f.apply(inter.midpoint()) * self.measure(inter) for inter in self.discretization)

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

    class Vector(OP.Operator,Space.Vector):
        """ L2 Vectors are operators from the domain onto R(C)
        """