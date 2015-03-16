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

from __future__ import unicode_literals, print_function, division, absolute_import
from future.builtins import object
from future import standard_library
standard_library.install_aliases()

from numbers import Number
from math import sin,cos,sqrt
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class Operator(object):
    """Abstract operator
    """
    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def apply(self, rhs, out):
        """Apply the operator, abstract
        """
        pass

    def __call__(self, rhs):
        """Shorthand for self.apply(rhs)
        """
        tmp = self.domain().empty()
        self.apply(rhs,tmp)
        return tmp

    def __add__(self, other):
        """Operator addition (pointwise)
        """

        if isinstance(other, Operator):  # Calculate sum
            return OperatorSum(self,other)
        else:
            raise TypeError('Expected an operator')

    def __mul__(self, other):
        """Pointwise multiplication of operators (A*B)(x) == A(x)*B(x)
        or scalar multiplication
        """

        if isinstance(other, Operator):
            return PointwiseProduct(self,other)
        elif isinstance(other, Number):
            return OperatorScalarMultiplication(self,other)
        else:
            raise TypeError('Expected an operator or a scalar')

    __rmul__ = __mul__

    @abstractmethod
    def domain(self):
        """Get the domain of the operator
        """

    @abstractmethod
    def range(self):
        """Get the range of the operator
        """

    @property
    def T(self):
        """Get the adjoint operator
        """
        return OperatorAdjoint(self)

class LinearOperator(Operator):
    """ Linear operator, satisfies A(ax+by)=aA(x)+bA(y)
    """
    
    @abstractmethod
    def applyAdjoint(self, rhs, out):
        """Apply the adjoint of the operator, abstract
        """
        pass

class SelfAdjointOperator(LinearOperator):
    """ Special case of self adjoint operators where A(x) = A.T(x)
    """
    def applyAdjoint(self, rhs, out):
        self.apply(rhs, out)

class OperatorSum(Operator):
    """Expression type for the sum of operators
    """
    def __init__(self,left,right):
        self.left = left
        self.right = right

    def apply(self, rhs, out):
        self.left(rhs, out) + self.right(rhs, out)

    def applyAdjoint(self, rhs, out):
        self.left.applyAdjoint(rhs, out) + self.right.applyAdjoint(rhs, out)

class OperatorComposition(Operator):
    """Expression type for the composition of operators
    """

    def __init__(self,left,right):
        if (right.range() != left.domain()):
            raise TypeError("Range and domain of operators do not fit")

        self.left = left
        self.right = right

    def apply(self, rhs,out):
        tmp = self.left.domain().empty()
        self.right.apply(rhs,tmp)
        self.left.apply(tmp,out)
    
    def applyAdjoint(self, rhs):
        tmp = self.right.domain().empty()
        self.left.applyAdjoint(rhs,tmp)
        self.right.applyAdjoint(tmp,out)

    def domain(self):
        return self.right.domain()

    def range(self):
        return self.left.range()

class PointwiseProduct(Operator):    
    """Pointwise multiplication of operators
    """

    def __init__(self,op1,op2):
        if (op1.range() != op2.range() or op1.domain() != op2.domain()):
            raise TypeError("Range and domain of operators do not fit")

        self.op1 = op1
        self.op2 = op2

    def apply(self,rhs):
        return self.op1(rhs) * self.op2(rhs)

class OperatorScalarMultiplication(Operator):
    """Expression type for the multiplication of operators with scalars
    """

    def __init__(self,op,scalar):
        self.op = op
        self.scalar = scalar

    def apply(self, rhs, out):
        self.op.apply(rhs, out)
        out*=scalar
    
    def applyAdjoint(self, rhs, out):
        self.op.applyAdjoint(rhs, out)
        out*=scalar

    def domain(self):
        return self.op.domain()

    def range(self):
        return self.op.range()

class OperatorAdjoint(LinearOperator):
    """Expression type for the adjoint of an operator
    """

    def __init__(self,op):
        self.op = op

    def apply(self, rhs, out):
        self.op.applyAdjoint(rhs, out)
    
    def applyAdjoint(self, rhs, out):
        self.op.apply(rhs, out)

    def domain(self):
        return self.op.range()

    def range(self):
        return self.op.domain()