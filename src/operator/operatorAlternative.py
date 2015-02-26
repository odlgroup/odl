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

"""Abstract operator
"""
class Operator(object):
    __metaclass__ = ABCMeta

    #Apply the method, abstract
    @abstractmethod
    def apply(self, rhs):
        pass

    #Shorthand for self.apply(rhs)
    def __call__(self, rhs):
        return self.apply(rhs)

    #Operator addition (pointwise)
    def __add__(self, other):
        if isinstance(other, Operator):  # Calculate sum
            return OperatorSum(self,other)
        else:
            raise NameError('HiThere') #TODO

    #Composition of operators ((A*B)(x) == A(B(x)))
    def __mul__(self, other):
        if isinstance(other, Operator):  # Calculate sum
            return OperatorComposition(self,other)
        else:
            raise NameError('HiThere') #TODO

class OperatorSum(Operator):
    def __init__(self,left,right):
        self.left = left
        self.right = right

    def apply(self, rhs):
        return self.left(rhs)+self.right(rhs)

class OperatorComposition(Operator):
    def __init__(self,left,right):
        self.left = left
        self.right = right

    def apply(self, rhs):
        return self.left(self.right(rhs))

#Multiply with scalar
class MultiplyOp(Operator):
    def __init__(self,a):
        self.a = a

    def apply(self,rhs):
        return self.a * rhs

#Add a scalar
class AddOp(Operator):
    def __init__(self,a):
        self.a = a

    def apply(self,rhs):
        return self.a + rhs
