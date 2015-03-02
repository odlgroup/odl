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

#TODO move this
class abstractstatic(staticmethod):
    """Decorator to enforce abstract static methods
    """
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

import numpy as np

class Operator(object):
    """Abstract operator
    """
    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def apply(self, rhs):
        """Apply the operator, abstract
        """
        pass

    def derivative(self, pos):
        """Calculate the derivative operator at some position
        """
        raise NotImplementedError("Derivative not implemented for this operator")

    def __call__(self, rhs):
        """Shorthand for self.apply(rhs)
        """
        return self.apply(rhs)

    def __add__(self, other):
        """Operator addition (pointwise)
        """

        if isinstance(other, Operator):  # Calculate sum
            return OperatorSum(self,other)
        else:
            raise TypeError('Expected an operator')

    def __mul__(self, other):
        """Composition of operators ((A*B)(x) == A(B(x)))
        or scalar multiplication
        """

        from numbers import Number

        if isinstance(other, Operator):  # Calculate sum
            return OperatorComposition(self,other)
        elif isinstance(other, Number):
            return OperatorScalarMultiplication(self,other)
        else:
            raise TypeError('Expected an operator or a scalar')

    def __rmul__(self,other):
        """Composition of operators ((A*B)(x) == A(B(x)))
        or scalar multiplication
        """

        from numbers import Number

        if isinstance(other, Operator):  # Calculate sum
            return OperatorComposition(other,self)
        elif isinstance(other, Number):
            return OperatorScalarMultiplication(self,other)
        else:
            raise TypeError('Expected an operator or a scalar')

class LinearOperator(Operator):
    """ Linear operator, satisfies A(ax+by)=aA(x)+bA(y)
    """
    
    @abstractmethod
    def applyAdjoint(self, rhs):
        """Apply the adjoint of the operator, abstract
        """
        pass

    def derivative(self):
        """Calculate the derivative operator at some position
        """
        return self

class SelfAdjointOperator(LinearOperator):
    """ Special case of self adjoint operators where A(x) = A.T(x)
    """
    def applyAdjoint(self, rhs):
        return self.apply(rhs)

class OperatorSum(Operator):
    """Expression type for the sum of operators
    """
    def __init__(self,left,right):
        self.left = left
        self.right = right

    def apply(self, rhs):
        return self.left(rhs) + self.right(rhs)

    def applyAdjoint(self, rhs):
        return self.left.applyAdjoint(rhs) + self.right.applyAdjoint(rhs)

    def derivative(self, pos):
        return self.left.derivative(pos) + self.right.derivative(pos)

class OperatorComposition(Operator):
    """Expression type for the composition of operators
    """

    def __init__(self,left,right):
        self.left = left
        self.right = right

    def apply(self, rhs):
        return self.left.apply(self.right.apply(rhs))
    
    def applyAdjoint(self, rhs):
        return self.right.applyAdjoint(self.left.applyAdjoint(rhs))

    def derivative(self, pos):
        #Use the chain rule
        return self.left.derivative(self.right.apply(pos)) * self.right.derivative(pos)

class OperatorScalarMultiplication(Operator):
    """Expression type for the multiplication of opeartors with scalars
    """

    def __init__(self,op,scalar):
        self.op = op
        self.scalar = scalar

    def apply(self, rhs):
        return scalar * self.op.apply(rhs)
    
    def applyAdjoint(self, rhs):
        return scalar * self.op.applyAdjoint(rhs)

    def derivative(self, pos):
        return scalar * self.op.derivative(pos)

class OperatorAdjoint(LinearOperator):
    """Expression type for the adjoint of an operator
    """

    def __init__(self,op):
        self.op = op

    def apply(self, rhs):
        return self.op.applyAdjoint(rhs)
    
    def applyAdjoint(self, rhs):
        return self.op.apply(rhs)

    def derivative(self, pos):
        return self.op.derivative(pos)

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
    

class ProductSpace(Space):
    """Product space (A x B)
    """

    def __init__(self,A,B):
        if (A.field() is not B.field()):
            raise AttributeError("A and B have to be spaces over the same field")

        self.A = A
        self.B = B

    class Vector(Space.Vector):
        def __init__(self,Apart,Bpart):
            self.Apart = Apart
            self.Bpart = Bpart

    def zero(self):
        return ProductSpace.Vector(self.A.zero(),self.B.zero())
    
    def inner(self,v1,v2):
        return self.A.inner(v1.Apart,v2.Apart) + self.B.inner(v1.Bpart,v2.Bpart)

    def linearComb(self,a,b,v1,v2):
        return ProductSpace.Vector(self.A.linearComb(a,b,v1.Apart,v2.Apart),self.B.linearComb(a,b,v1.Bpart,v2.Bpart))

    def field(self):
        return self.A.field()

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

    class MultiplyOp(SelfAdjointOperator):    
        """Multiply with scalar
        """

        def __init__(self,a):
            self.a = a

        def apply(self,rhs):
            return self.a * rhs

    class AddOp(SelfAdjointOperator):
        """Add scalar
        """

        def __init__(self,a):
            self.a = a

        def apply(self,rhs):
            return self.a + rhs

#Example of a space:
class R3(Space):
    """The real numbers
    """

    def inner(self,A,B):
        return np.vdot(A,B)
    
    def linearComb(self,a,b,A,B):
        return a*A+b*B

    def zero(self):
        return np.zeros(3)

    def field(self):
        return Field.Real

    def dimension(self):
        return 3

    class MultiplyOp(Operator):    
        """Multiply with scalar
        """

        def __init__(self,A):
            self.A = A

        def apply(self,rhs):
            return np.dot(self.A,rhs)

        def applyAdjoint(self,rhs):
            return np.dot(self.A.T,rhs)

#Example of a space:
class RN(Space):
    """The real numbers
    """

    def __init__(self,n):
        self.n = n

    def inner(selfA,B):
        return np.vdot(A,B)
    
    def linearComb(self,a,b,A,B):
        return a*A+b*B

    def zero(self):
        return np.zeros(n)

    def field(self):
        return Field.Real

    def dimension(self):
        return n

    class MultiplyOp(Operator):    
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

class voxelDiscretization(measureSpace):
    def __init__(self,begin,end,n):
        self.begin = begin
        self.end = end
        self.n = n

    def integrate(self,f):
        s = 0.0
        for x in np.linspace(self.begin,self.end,self.n):
            s += f(x)

#Example of a space:
class L2(Space):
    """The real numbers
    """

    def __init__(self,discretization):
        self.discretization = discretization

    def inner(self,v1,v2):
        return self.discretization.integrate(L2.PointwiseProduct(v1,v2))

    def linearComb(self,a,b,A,B):
        return a*A+b*B

    def zero(self):
        return 0.0

    def field(self):
        return Field.Real

    def dimension(self):
        return 1.0/0.0

    class Sin(Operator):
        def apply(self,rhs):
            return sin(rhs)

    class Cos(Operator):
        def apply(self,rhs):
            return sin(rhs)

    class PointwiseProduct(Operator):    
        """Multiply with scalar
        """

        def __init__(self,a,b):
            self.a = a
            self.b = b

        def apply(self,rhs):
            return self.a(rhs) * self.b(rhs)