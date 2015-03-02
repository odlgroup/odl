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

from math import sin,cos,sqrt
import numpy as np

import RL.operator.operatorAlternative as OP

from abc import ABCMeta, abstractmethod

class Field:
    Real, Complex = range(2)

class Space(object):
    """Abstract space
    """

    __metaclass__ = ABCMeta #Set as abstract

    class Vector(object):
        def __add__(self, other):
            """Vector addition
            """
            return Space.linearComb(1,1,self,other)

        def __mul__(self, other):
            """Scalar multiplication
            """
            return Space.linearComb(other,0,self,Space.zero())

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
    def linearComb(self,a,b,A,B):
        """Calculate a*A+b*B
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

    def squaredNorm(self,x):
        return self.inner(x,x)

    def norm(self,x):
        return sqrt(self.squaredNorm(x,x))
    

class ProductSpace(Space):
    """Product space (A x B)
    """

    def __init__(self,A,B):
        if (A.field() is not B.field()):
            raise AttributeError("A and B have to be spaces over the same field")

        self.A = A
        self.B = B

    class Vector(Space.Vector):
        def __init__(self,parent,A,B):
            self.parent = parent
            self.vA = vA
            self.vB = vB

        def __getitem__(self,index): #TODO should we have this?
            if (index < self.parent.A.dimension()):
                return self.vA[index]
            else:
                return self.vB[index-self.parent.A.dimension()]

    def zero(self):
        return ProductSpace.Vector(self,self.A.zero(),self.B.zero())
    
    def inner(self,v1,v2):
        return self.A.inner(v1.vA,v2.vA) + self.B.inner(v1.vB,v2.vB)

    def linearComb(self,a,b,v1,v2):
        return ProductSpace.Vector(self,self.A.linearComb(a,b,v1.vA,v2.vA),self.B.linearComb(a,b,v1.vB,v2.vB))

    def field(self):
        return self.A.field() #A and B has same field

    def dimension(self):
        return self.A.dimension()+self.B.dimension()

#Example of a space:
class Reals(Space):
    """The real numbers
    """

    def inner(self,A,B):
        return A*B

    def linearComb(self,a,b,A,B):
        return a*A+b*B

    def zero(self):
        return 0.0

    def field(self):
        return Field.Real

    def dimension(self):
        return 1

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

#Example of a space:
class RN(Space):
    """The real numbers
    """

    def __init__(self,n):
        self.n = n

    def inner(self,A,B):
        return np.vdot(A,B)
    
    def linearComb(self,a,b,A,B):
        return a*A+b*B

    def zero(self):
        return np.zeros(n)

    def field(self):
        return Field.Real

    def dimension(self):
        return n

    class MultiplyOp(OP.Operator):    
        """Multiply with scalar
        """

        def __init__(self,A):
            self.A = A

        def apply(self,rhs):
            return np.dot(self.A,rhs)

        def applyAdjoint(self,rhs):
            return np.dot(self.A.T,rhs)


#Example of a space:
class RNM(Space):
    """The real numbers
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
    @abstractmethod
    def integrate(self,f):
        """Calculate the integral of f
        """
        pass

class unitInterval(measureSpace):
    def __init__(self,begin,end):
        self.begin = begin
        self.end = end

class linspaceDiscretization(unitInterval):
    def __init__(self,begin,end,n):
        unitInterval.__init__(self,begin,end)
        self.n = n

    def integrate(self,f):
        s = 0.0
        for x in np.linspace(self.begin,self.end,self.n):
            s += f(x)

        return s * (self.end - self.begin) / self.n

#Example of a space:
class L2(Space):
    """The real numbers
    """

    def __init__(self,domain):
        self.domain = domain

    def inner(self,v1,v2):
        return self.domain.integrate(L2.PointwiseProduct(v1,v2))

    def linearComb(self,a,b,A,B):
        return a*A+b*B

    def zero(self):
        class zeroFunction(OP.SelfAdjointOperator):
            def apply(self,rhs):
                return 0.0

        return zeroFunction()

    def field(self):
        return Field.Real

    def dimension(self):
        return 1.0/0.0

    class Sin(OP.Operator):
        def apply(self,rhs):
            return sin(rhs)

    class Cos(OP.Operator):
        def apply(self,rhs):
            return cos(rhs)

    class PointwiseProduct(OP.Operator):    
        """Multiply with scalar
        """

        def __init__(self,a,b):
            self.a = a
            self.b = b

        def apply(self,rhs):
            return self.a(rhs) * self.b(rhs)