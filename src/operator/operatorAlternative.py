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
from abc import ABCMeta, abstractmethod
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

        if isinstance(other, Operator):
            return OperatorComposition(self,other)
        elif isinstance(other, Number):
            return OperatorScalarMultiplication(self,other)
        else:
            raise TypeError('Expected an operator or a scalar')

    def __rmul__(self,other):
        """Composition of operators ((A*B)(x) == A(B(x)))
        or scalar multiplication
        """

        if isinstance(other, Operator):
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

class PointwiseProduct(Operator):    
    """Pointwise multiplication of operators
    """

    def __init__(self,op1,op2):
        self.op1 = op1
        self.op2 = op2

    def apply(self,rhs):
        return self.op1(rhs) * self.op2(rhs)

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

class OperatorAdjoint(LinearOperator):
    """Expression type for the adjoint of an operator
    """

    def __init__(self,op):
        self.op = op

    def apply(self, rhs):
        return self.op.applyAdjoint(rhs)
    
    def applyAdjoint(self, rhs):
        return self.op.apply(rhs)