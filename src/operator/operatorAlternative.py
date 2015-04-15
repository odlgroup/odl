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
from abc import ABCMeta, abstractmethod, abstractproperty


class Operator(object):
    """Abstract operator
    """
    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def applyImpl(self, rhs, out):
        """Apply the operator, abstract
        """

    @abstractproperty
    def domain(self):
        """Get the domain of the operator
        """

    @abstractproperty
    def range(self):
        """Get the range of the operator
        """

    #Implicitly defined operators
    def apply(self, rhs, out):
        if not self.domain.isMember(rhs): 
            raise TypeError('rhs ({}) is not in the domain of this operator ({})'.format(rhs, self))
        if not self.range.isMember(out): 
            raise TypeError('out ({}) is not in the range of this operator ({})'.format(out, self))

        self.applyImpl(rhs, out)

    def __call__(self, rhs):
        """Shorthand for self.apply(rhs)
        """
        tmp = self.range.empty()
        self.apply(rhs, tmp)
        return tmp

    def __add__(self, other):
        """Operator addition (pointwise)
        """

        if isinstance(other, Operator):  # Calculate sum
            return OperatorSum(self, other)
        else:
            raise TypeError('Expected an operator')

    def __rmul__(self, other):
        """Multiplication of operators with scalars (a*A)(x) = a*A(x)
        """

        if isinstance(other, Number):
            return OperatorScalarMultiplication(self, other)
        else:
            raise TypeError('Expected an operator or a scalar')

    __mul__ = __rmul__ #Should we have this?

    def __str__(self):
        return "Operator " + self.__class__.__name__ + ": " + str(self.domain) + "->" + str(self.range)


class OperatorSum(Operator):
    """Expression type for the sum of operators
    """
    def __init__(self, op1, op2):
        if op1.range != op2.range or op1.domain != op2.domain:
            raise TypeError("Range and domain of operators do not fit")

        self.op1 = op1
        self.op2 = op2

    def applyImpl(self, rhs, out):
        tmp = self.range.empty()
        self.op1.applyImpl(rhs, out)
        self.op2.applyImpl(rhs, tmp)
        out += tmp

    @property
    def domain(self):
        return self.op1.domain

    @property
    def range(self):
        return self.op1.range

class OperatorComposition(Operator):
    """Expression type for the composition of operators
    """

    def __init__(self, left, right):
        if right.range != left.domain:
            raise TypeError("Range and domain of operators do not fit")

        self.left = left
        self.right = right

    def applyImpl(self, rhs, out):
        tmp = self.right.range.empty()
        self.right.applyImpl(rhs, tmp)
        self.left.applyImpl(tmp, out)
        
    @property
    def domain(self):
        return self.right.domain

    @property
    def range(self):
        return self.left.range

class PointwiseProduct(Operator):    
    """Pointwise multiplication of operators
    """

    def __init__(self, op1, op2):
        if op1.range() != op2.range or op1.domain != op2.domain:
            raise TypeError("Range and domain of operators do not fit")

        self.op1 = op1
        self.op2 = op2

    def applyImpl(self, rhs, out):
        tmp = self.op2.range.empty()
        self.op1.applyImpl(rhs, out)
        self.op2.applyImpl(rhs, tmp)
        out *= tmp

    @property
    def domain(self):
        return self.op1.domain

    @property
    def range(self):
        return self.op1.range

class OperatorScalarMultiplication(Operator):
    """Expression type for the multiplication of operators with scalars
    """

    def __init__(self, op, scalar):
        self.operator = operator
        self.scalar = scalar

    def applyImpl(self, rhs, out):
        self.operator.applyImpl(rhs, out)
        out *= self.scalar

    @property
    def domain(self):
        return self.operator.domain

    @property
    def range(self):
        return self.operator.range

    
class LinearOperator(Operator):
    """ Linear operator, satisfies A(ax+by)=aA(x)+bA(y)
    """
    
    @abstractmethod
    def applyAdjointImpl(self, rhs, out):
        """Apply the adjoint of the operator, abstract
        """

    #Implicitly defined operators
    @property
    def T(self):
        return OperatorAdjoint(self)

    def applyAdjoint(self, rhs, out):
        if not self.range.isMember(rhs): 
            raise TypeError('rhs ({}) is not in the domain of this operators ({}) adjoint'.format(rhs,self))
        if not self.domain.isMember(out): 
            raise TypeError('out ({}) is not in the range of this operators ({}) adjoint'.format(out,self))

        self.applyAdjointImpl(rhs, out)

    def __add__(self, other):
        """Operator addition (pointwise)
        """

        if isinstance(other, LinearOperator): #Specialization if both are linear
            return LinearOperatorSum(self, other)
        else:
            return Operator.__add__(self, other)

    def __rmul__(self, other):
        """Multiplication of operators with scalars (a*A)(x) = a*A(x)
        """

        if isinstance(other, Number):
            return LinearOperatorScalarMultiplication(self, other)
        else:
            raise TypeError('Expected an operator or a scalar')

    __mul__ = __rmul__ #Should we have this?


class SelfAdjointOperator(LinearOperator):
    """ Special case of self adjoint operators where A(x) = A.T(x)
    """
    
    __metaclass__ = ABCMeta #Set as abstract

    def applyAdjointImpl(self, rhs, out):
        self.applyImpl(rhs, out)


class OperatorAdjoint(LinearOperator):
    """Expression type for the adjoint of an operator
    """

    def __init__(self, op):
        self.operator = op

    def applyImpl(self, rhs, out):
        self.operator.applyAdjointImpl(rhs, out)
    
    def applyAdjointImpl(self, rhs, out):
        self.operator.applyImpl(rhs, out)

    @property
    def domain(self):
        return self.operator.range

    @property
    def range(self):
        return self.operator.domain


class LinearOperatorSum(OperatorSum, LinearOperator):
    """Expression type for the sum of linear operators
    """
    def __init__(self, op1, op2):
        LinearOperator.__init__(self, op1, op2)

    def applyAdjointImpl(self, rhs, out):
        tmp = self.domain.empty()
        self.op1.applyAdjointImpl(rhs, out)
        self.op2.applyAdjointImpl(rhs, tmp)
        out += tmp


class LinearOperatorComposition(OperatorComposition, LinearOperator):
    """Expression type for the composition of operators
    """

    def __init__(self, left, right):
        OperatorComposition.__init__(self, left, right)
    
    def applyAdjointImpl(self, rhs, out):
        tmp = self.left.domain.empty()
        self.left.applyAdjoint(rhs, tmp)
        self.right.applyAdjoint(tmp, out)


class LinearOperatorScalarMultiplication(OperatorScalarMultiplication, LinearOperator):
    """Expression type for the multiplication of operators with scalars
    """

    def __init__(self, op, scalar):
        OperatorScalarMultiplication.__init__(self, op, scalar)
    
    def applyAdjointImpl(self, rhs, out):
        self.op.applyAdjointImpl(rhs, out)
        out *= self.scalar
